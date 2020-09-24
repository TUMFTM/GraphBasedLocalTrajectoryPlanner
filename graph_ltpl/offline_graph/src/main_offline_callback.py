import numpy as np
import configparser
import pickle
import os.path as osfuncs
import hashlib
import logging

# custom modules
import graph_ltpl


def main_offline_callback(globtraj_param_path: str,
                          graph_off_config_path: str,
                          graph_store_path: str,
                          graph_logging_path: str = None,
                          graph_id: str = None,
                          force_recalc=False) -> tuple:
    """
    The main function to be called once for the offline graph setup. The function tries to load an existing GraphBase
    object. If the object is not existent or not valid for the current parameter set, a new one is created. In this case
    the following steps are executed:

    * Load global race line and map
    * Calculate variable step-size along track depending on straight and curve segments
    * Init new GraphBase class instance
    * Setup node sceletion (spread nodes on normal vectors along reference line)
    * Generate edges between nodes
    * Prune graph (e.g. remove dead end edges and associated paths)
    * Calculate costs for each path segment
    * Store graph for later executions

    :param globtraj_param_path:    path pointing to data file holding all information about the global race line and map
    :param graph_off_config_path:  path pointing to the config file specifying the offline graph behavior / generation
    :param graph_store_path:       path pointing to location, where a copy of the setup graph should be stored/loaded
    :param graph_logging_path:     path pointing to location where the graph should be stored together with the logs
    :param graph_id:               unique graph identifier retrieved from a (eventually) loaded graph
    :param force_recalc:           flag, if set to "True" a new graph is calculated instead of loading from file
    :returns:
        * **graph_base** -         reference to the GraphBase object instance holding all graph relevant information
        * **new_base_generated** - status, whether a new graph base has been generated or not

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        28.09.2018

    """

    # ------------------------------------------------------------------------------------------------------------------
    # SETUP GRAPH ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    new_base_generated = False
    graph_base = None
    # get the MD5-hash of all config files (fused together, since we want to recalculate whenever any file changed)
    calculated_md5 = md5(globtraj_param_path) + md5(graph_off_config_path)

    # If legible, load graph from file (else generate)
    if not force_recalc and osfuncs.isfile(graph_store_path):
        f = open(graph_store_path, 'rb')
        graph_base = pickle.load(f)
        f.close()
        logging.getLogger("local_trajectory_logger").debug("Loaded database with " + str(len(graph_base.get_nodes()))
                                                           + " node and " + str(len(graph_base.get_edges()))
                                                           + " edges from file...")

    if force_recalc or graph_base is None or calculated_md5 != graph_base.md5_params:
        new_base_generated = True
        if force_recalc:
            print("Manually forced recalculation of graph! Skipped graph import from file!")
        if graph_base is not None and calculated_md5 is not graph_base.md5_params:
            print("MD5-Sum of any param-file does not match the one in the graph object! Triggered recalculation!")

        # load graph configuration
        graph_config = configparser.ConfigParser()
        if not graph_config.read(graph_off_config_path):
            raise ValueError('Specified graph config file does not exist or is empty!')

        # load data from csv files
        refline, t_width_right, t_width_left, normvec_normalized, alpha, length_rl, vel_rl, kappa_rl \
            = graph_ltpl.imp_global_traj.src.import_globtraj_csv.import_globtraj_csv(import_path=globtraj_param_path)

        # calculate closed race line parameters
        # s, x, y, kappa, vel
        s = np.concatenate(([0], np.cumsum(length_rl)))
        xy = refline + normvec_normalized * alpha[:, np.newaxis]
        raceline_params = np.column_stack((xy, kappa_rl, vel_rl))

        # determine if track is closed or unclosed (check if end and start-point are close together)
        closed = (np.hypot(xy[0, 0] - xy[-1, 0], xy[0, 1] - xy[-1, 1])
                  < graph_config.getfloat('LATTICE', 'closure_detection_dist'))
        if closed:
            logging.getLogger("local_trajectory_logger").debug("Input line is interpreted as closed track!")

            # close line
            glob_rl = np.column_stack((s, np.vstack((raceline_params, raceline_params[0, :]))))
        else:
            logging.getLogger("local_trajectory_logger").debug("Input line is interpreted as _unclosed_ track!")
            glob_rl = np.column_stack((s[:-1], raceline_params))

        # based on curvature get index array for selection of normal vectors and corresponding raceline parameters
        idx_array = graph_ltpl.imp_global_traj.src.variable_step_size. \
            variable_step_size(kappa=kappa_rl,
                               dist=length_rl,
                               d_curve=graph_config.getfloat('LATTICE', 'lon_curve_step'),
                               d_straight=graph_config.getfloat('LATTICE', 'lon_straight_step'),
                               curve_th=graph_config.getfloat('LATTICE', 'curve_thr'),
                               force_last=not closed)

        # extract values at determined positions
        refline = refline[idx_array, :]
        t_width_right = t_width_right[idx_array]
        t_width_left = t_width_left[idx_array]
        normvec_normalized = normvec_normalized[idx_array]
        alpha = alpha[idx_array]
        vel_rl = vel_rl[idx_array]
        s_raceline = s[idx_array]

        length_rl_tmp = []
        for idx_from, idx_to in zip(idx_array[:-1], idx_array[1:]):
            length_rl_tmp.append(np.sum(length_rl[idx_from:idx_to]))

        length_rl_tmp.append(0.0)
        length_rl = list(length_rl_tmp)

        # init graph base object
        graph_base = graph_ltpl.data_objects.GraphBase.\
            GraphBase(lat_offset=graph_config.getfloat('LATTICE', 'lat_offset'),
                      num_layers=np.size(alpha, axis=0),
                      refline=refline,
                      normvec_normalized=normvec_normalized,
                      track_width_right=t_width_right,
                      track_width_left=t_width_left,
                      alpha=alpha,
                      vel_raceline=vel_rl,
                      s_raceline=s_raceline,
                      lat_resolution=graph_config.getfloat('LATTICE', 'lat_resolution'),
                      sampled_resolution=graph_config.getfloat('SAMPLING', 'stepsize_approx'),
                      vel_decrease_lat=graph_config.getfloat('PLANNINGTARGET', 'vel_decrease_lat'),
                      veh_width=graph_config.getfloat('VEHICLE', 'veh_width'),
                      veh_length=graph_config.getfloat('VEHICLE', 'veh_length'),
                      veh_turn=graph_config.getfloat('VEHICLE', 'veh_turn'),
                      md5_params=calculated_md5,
                      graph_id=graph_id,
                      glob_rl=glob_rl,
                      virt_goal_node=graph_config.getboolean('LATTICE', 'virt_goal_n'),
                      virt_goal_node_cost=graph_config.getfloat('COST', 'w_virt_goal'),
                      min_plan_horizon=graph_config.getfloat('PLANNINGTARGET', 'min_plan_horizon'),
                      plan_horizon_mode=graph_config.get('PLANNINGTARGET', 'plan_horizon_mode'),
                      closed=closed)

        # set up state space
        state_pos = graph_ltpl.offline_graph.src.gen_node_skeleton. \
            gen_node_skeleton(graph_base=graph_base,
                              length_raceline=length_rl,
                              var_heading=graph_config.getboolean('LATTICE', 'variable_heading'))

        # convert to array of arrays
        state_pos_arr = np.empty(shape=(len(state_pos), 2), dtype=np.object)
        state_pos_arr[:] = state_pos

        # generate edges (polynomials and coordinate arrays)
        graph_ltpl.offline_graph.src.gen_edges.gen_edges(state_pos=state_pos_arr,
                                                         graph_base=graph_base,
                                                         stepsize_approx=graph_config.getfloat('SAMPLING',
                                                                                               'stepsize_approx'),
                                                         min_vel_race=graph_config.getfloat('LATTICE', 'min_vel_race'),
                                                         closed=closed)

        # prune graph (remove dead ends)
        graph_ltpl.offline_graph.src.prune_graph.prune_graph(graph_base=graph_base,
                                                             closed=closed)

        # generate cost
        graph_ltpl.offline_graph.src.gen_offline_cost.gen_offline_cost(graph_base=graph_base,
                                                                       cost_config_path=graph_off_config_path)

        # declare initialization as finished and initialize original filter
        graph_base.init_filtering()

        # store graph for later use
        f = open(graph_store_path, 'wb')
        pickle.dump(graph_base, f)
        f.close()
    else:
        if graph_logging_path is not None:
            # if existing graph object is valid, adapt logging file name according to stored id in graph object
            graph_logging_path = (graph_logging_path[:graph_logging_path.rfind('/Graph_Objects/') + 15]
                                  + str(graph_base.graph_id) + ".pckl")

    # log graph, if path provided and not existent
    if graph_logging_path is not None and not osfuncs.isfile(graph_logging_path):
        f = open(graph_logging_path, 'wb')
        pickle.dump(graph_base, f)
        f.close()

    return graph_base, new_base_generated


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
