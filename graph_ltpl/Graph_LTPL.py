import sys
import os
import numpy as np
import json
import logging
import datetime
import configparser
import copy

# own modules
mod_local_trajectory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mod_local_trajectory_path)
import graph_ltpl

# tries to load a previously computed graph set (paths -> offline-part), unless set to "True"
FORCE_RECALC = False

# BHPL action sets (do not modify!)
BP_SETS = ["straight", "left", "right"]

# required path dict entries
REQ_PATH_DICT_ENTRIES = ['globtraj_input_path', 'graph_store_path', 'ltpl_offline_param_path', 'ltpl_online_param_path',
                         'graph_log_id', 'log_path']


class Graph_LTPL(object):
    """
    Class providing all functions for a graph-based trajectory planner.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        19.03.2019
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 path_dict: dict,
                 visual_mode: bool = False,
                 log_to_file: bool = True) -> None:
        """
        :param path_dict:       dictionary holding strings to various configuration files, mandatory list below

                                * 'globtraj_input_path'     file holding track information (bounds and reference line)
                                * 'graph_store_path'        path the graph is (tried to be) loaded and stored to
                                * 'ltpl_offline_param_path' parameters specifying the offline generation of the graph
                                * 'ltpl_online_param_path'  parameters specifying the online handling of the graph
                                * 'graph_log_id'            (if log_to_file) unique id of the current graph (e.g. time)
                                * 'log_path'                (if log_to_file) loc. where all logged files are placed

        :param visual_mode:     enable / disable live-plot of track and vehicles (slows down execution of process)
        :param log_to_file:     enable / disable log-generation

        """

        # check if path dict holds all required entries
        path_dict_keys = list(path_dict.keys())

        for path_dict_entry in REQ_PATH_DICT_ENTRIES:
            if path_dict_entry not in path_dict_keys:
                # do not fire, if logging is not enabled and logging paths are missing
                if log_to_file or 'log' not in path_dict_entry:
                    raise ValueError('Missing path specification in path_dict (Missing entry: "' + path_dict_entry
                                     + '")!')

        # -- SETUP LOGGERS ---------------------------------------------------------------------------------------------

        if log_to_file:
            # add further paths specific paths
            path_dict['graph_log_path'] = path_dict['log_path'] + "Graph_Objects/" + path_dict['graph_log_id'] + ".pckl"
            if not os.path.exists(path_dict['log_path'] + "Graph_Objects/"):
                os.makedirs(path_dict['log_path'] + "Graph_Objects/")

            # create logging folder and write header
            fld_name = path_dict['log_path'] + datetime.datetime.now().strftime("%Y_%m_%d") + "/"
            file_prefix = datetime.datetime.now().strftime("%H_%M_%S")

            if not os.path.exists(fld_name):
                os.makedirs(fld_name)
            path_dict['graph_log_msgs_path'] = fld_name + file_prefix + "_msg.csv"
            path_dict['graph_log_data_path'] = fld_name + file_prefix + "_data.csv"

            with open(path_dict['graph_log_msgs_path'], "w+") as fh:
                header = "time;type;message\n"
                fh.write(header)

            # init logger
            log = logging.getLogger("local_trajectory_logger")

            # -- CONFIGURE CONSOLE OUTPUT ------------------------------------------------------------------------------
            # normal - stdout
            hdlr = logging.StreamHandler(sys.stdout)
            hdlr.setFormatter(logging.Formatter('%(levelname)s [%(asctime)s]: %(message)s', '%H:%M:%S'))
            hdlr.addFilter(lambda record: record.levelno < logging.CRITICAL)
            hdlr.setLevel(os.environ.get("LOGLEVEL", "INFO"))
            log.addHandler(hdlr)

            # error - stderr
            hdlr_e = logging.StreamHandler()
            hdlr_e.setFormatter(logging.Formatter('%(levelname)s [%(asctime)s]: %(message)s', '%H:%M:%S'))
            hdlr_e.setLevel(logging.CRITICAL)
            log.addHandler(hdlr_e)

            # -- CONFIGURE FILE OUTPUT ---------------------------------------------------------------------------------
            fhdlr = logging.FileHandler(path_dict['graph_log_msgs_path'])
            fhdlr.setFormatter(logging.Formatter('%(created)s;%(levelname)s;%(message)s'))
            fhdlr.setLevel(os.environ.get("LOGLEVEL", "INFO"))
            log.addHandler(fhdlr)

            # set the global logger level (should be the lowest of all individual streams --> leave at DEBUG!)
            log.setLevel(logging.DEBUG)

        # ## INIT CLASS MEMBER VARIABLES (executed for all ltpl approaches - only basic initialization here) ###########
        # -- SHARED ----------------------------------------------------------------------------------------------------
        # init logger handle
        self.__log = logging.getLogger("local_trajectory_logger")

        # basic configuration parameters
        self.__path_dict = path_dict
        self.__visual_mode = visual_mode
        self.__log_to_file = log_to_file

        # object and zone related interfaces
        self.__obj_list_handler = None
        self.__obj_veh = None      # list of vehicle objects -> for class definition refer to "ObjectListInterface.py"
        self.__obj_zone = []       # list of zone objects    -> for class definition refer to "ObjectListInterface.py"

        # trajectory ID and time-stamp
        self.__action_set = None
        self.__action_set_id = None
        self.__traj_time = 0.0

        # graph log object instance
        self.__graph_log_handler = None

        # plot object instance
        self.__graph_plot_handler = None
        self.__cost_dep_color = None

        self.__graph_base = None

        # online trajectory handler object instance
        self.__oth = None

        # iterative trajectory storage
        self.__pos_est = None
        self.__prev_action_id = None
        self.__prev_traj_idx = None
        self.__local_trajectories = None

        # required for logging and visualization only
        self.__plan_start_node = None
        self.__node_list = None
        self.__const_path_seg = None
        self.__cut_index_pos = None

        # (NOTE: second "graph_ltpl" import required here)
        import graph_ltpl

        if self.__visual_mode:
            # import plot handler when visualization is requested (avoid errors on devices with no connected screen)
            import graph_ltpl.visualization.src.PlotHandler

        online_param = configparser.ConfigParser()
        online_param.read(path_dict['ltpl_online_param_path'])

        self.__cost_dep_color = json.loads(online_param.get('GENERAL', 'cost_dep_color'))
        self.__max_heading_offset = json.loads(online_param.get('GENERAL', 'max_heading_offset'))
        self.__nmbr_export_points = json.loads(online_param.get('EXPORT', 'nmbr_export_points'))

        # init object-list
        self.__obj_list_handler = graph_ltpl.data_objects.ObjectListInterface.ObjectListInterface()

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __del__(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # CLASS METHODS ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def graph_init(self,
                   veh_param_dyn_model_exp: float = 1.0,
                   veh_param_dragcoeff: float = 0.85,
                   veh_param_mass: float = 1000.0) -> None:
        """
        Init offline part of graph.

        :param veh_param_dyn_model_exp:  vehicle dynamics model exponent (range [1.0, 2.0])
        :param veh_param_dragcoeff:      drag coefficient -> 0.5 * rho_air * c_w_A in m2*kg/m3
        :param veh_param_mass:           vehicle mass in kg

        """

        self.__graph_base, new_base_gen = graph_ltpl.offline_graph.src.main_offline_callback. \
            main_offline_callback(globtraj_param_path=self.__path_dict['globtraj_input_path'],
                                  graph_off_config_path=self.__path_dict['ltpl_offline_param_path'],
                                  graph_store_path=self.__path_dict['graph_store_path'],
                                  graph_logging_path=self.__path_dict.get('graph_log_path', None),
                                  graph_id=self.__path_dict.get('graph_log_id', None),
                                  force_recalc=FORCE_RECALC)

        # -- PLOTTING --------------------------------------------------------------------------------------------------
        if self.__visual_mode and new_base_gen:
            # Init plot class
            self.__graph_plot_handler = graph_ltpl.visualization.src.PlotHandler.\
                PlotHandler(plot_title="Local Trajectory - Offline Graph")

            # Plot major components
            self.__graph_plot_handler.plot_graph_base(graph_base=self.__graph_base,
                                                      cost_dep_color=self.__cost_dep_color)

            self.__graph_plot_handler.show_plot()

        # -- INIT ONLINE TRAJECTORY HANDLER ----------------------------------------------------------------------------
        self.__oth = graph_ltpl.online_graph.src.OnlineTrajectoryHandler.OnlineTrajectoryHandler(
            graph_base=self.__graph_base,
            graph_online_config_path=self.__path_dict['ltpl_online_param_path'],
            graph_offline_config_path=self.__path_dict['ltpl_offline_param_path'],
            veh_param_dyn_model_exp=veh_param_dyn_model_exp,
            veh_param_dragcoeff=veh_param_dragcoeff,
            veh_param_mass=veh_param_mass)

        # -- INIT OBJECT LIST INTERFACE --------------------------------------------------------------------------------
        self.__obj_list_handler.set_track_data(refline=self.__graph_base.refline,
                                               normvec_normalized=self.__graph_base.normvec_normalized,
                                               w_left=self.__graph_base.track_width_left,
                                               w_right=self.__graph_base.track_width_right)

        # Calculate zones for given graph (reduce online computation)
        # load config file
        graph_config = configparser.ConfigParser()
        if not graph_config.read(self.__path_dict['ltpl_online_param_path']):
            raise ValueError('Specified cost config file does not exist or is empty!')

        # -- INIT PLOT HANDLER -----------------------------------------------------------------------------------------
        if self.__visual_mode:
            # Re-Init plot class
            self.__graph_plot_handler = graph_ltpl.visualization.src.PlotHandler.\
                PlotHandler(plot_title="Local Trajectory - Online Graph")

            # Plot major components
            self.__graph_plot_handler.plot_graph_base(graph_base=self.__graph_base,
                                                      cost_dep_color=self.__cost_dep_color,
                                                      plot_edges=False)

        # -- INIT LOGGING MODULE ---------------------------------------------------------------------------------------
        if self.__log_to_file:
            self.__graph_log_handler = graph_ltpl.helper_funcs.src.Logging. \
                Logging(graph_id=self.__graph_base.graph_id,
                        log_path=self.__path_dict['graph_log_data_path'])

    # ------------------------------------------------------------------------------------------------------------------

    def set_startpos(self,
                     pos_est: np.ndarray,
                     heading_est: float,
                     vel_est: float = 0.0) -> bool:
        """
        Set the start position for the graph.

        :param pos_est:             position estimate of the vehicle in map-coordinates [x, y]
        :param heading_est:         heading estimate of the vehicle in map-frame
        :param vel_est:             velocity estimate of the vehicle at the start position (allow flying start)
        :returns:
            * **out_of_track** -    boolean flag, 'True' if requested position does not reside on the track as intended

        """

        # check if initialized already
        if self.__oth is None:
            raise ValueError("Could not set start position, since graph is not initialized yet. "
                             "Call graph_init() first!")

        self.__pos_est = pos_est

        # reset last action set
        self.__action_set = {"straight": []}

        # set initial pose for graph and check if within track
        in_track, cor_heading = self.__oth.set_initial_pose(start_pos=self.__pos_est,
                                                            start_heading=heading_est,
                                                            start_vel=vel_est,
                                                            max_heading_offset=self.__max_heading_offset)

        # if out of track or wrong heading, jump to start of loop
        out_of_track = not in_track or not cor_heading

        return out_of_track

    # ------------------------------------------------------------------------------------------------------------------

    def calc_paths(self,
                   prev_action_id: str,
                   prev_traj_idx: int = 0,
                   object_list: list = None,
                   blocked_zones: dict = None) -> dict:
        """
        Calculate paths for current time-step.

        :param prev_action_id:  trajectory set pursued in previous iteration ['straight', 'follow', 'left', 'right']
        :param prev_traj_idx:   trajectory index folloed in previous iteration (if multiple per set)
        :param object_list:     list of dicts, each dict describing an object with keys ['type', 'X', 'Y', 'theta', ...]
        :param blocked_zones:   dict, each key holding the zone ID and each value descr. a zone list with the values:
                                [blocked layer nbrs, blocked node nbrs, left bound of region, right bound of region]
        :returns:
            * **path_dict** -   dict holding paths for each of the available action sets {'straight': np([...]), ...}

        """

        self.__prev_action_id = prev_action_id
        self.__prev_traj_idx = prev_traj_idx

        # update internal object handles
        self.__obj_veh = self.__obj_list_handler.process_object_list(object_list=object_list)

        # update zones
        if blocked_zones is not None:
            for blocked_zone_id in blocked_zones.keys():
                self.__obj_zone = self.__obj_list_handler.update_zone(zone_id=blocked_zone_id,
                                                                      zone_data=blocked_zones[blocked_zone_id],
                                                                      zone_type='nodes')

        # update (clear and set new) obstacles in the scene
        self.__oth.update_objects(obj_veh=self.__obj_veh,
                                  obj_zone=self.__obj_zone)

        # trigger local trajectory generation
        path_dict, self.__plan_start_node, self.__node_list, self.__const_path_seg = \
            self.__oth.calc_paths(action_id_sel=self.__prev_action_id,
                                  idx_sel_traj=self.__prev_traj_idx)

        return path_dict

    # ------------------------------------------------------------------------------------------------------------------

    def calc_vel_profile(self,
                         pos_est: np.ndarray,
                         vel_est: float,
                         vel_max: float = 100.0,
                         gg_scale: np.ndarray = 1.0,
                         local_gg: dict = (5.0, 5.0),
                         ax_max_machines: np.ndarray = np.atleast_2d([100.0, 5.0]),
                         safety_d: float = 30.0,
                         incl_emerg_traj: bool = False) -> tuple:
        """
        Calculate velocity profile for current given paths and trim to closest point on trajectory to "pos_est".

        :param pos_est:          position estimate of the vehicle in map-coordinates [x, y]
        :param vel_est:          velocity estimate of the vehicle (vel-profile at current pos will start at this value)
        :param vel_max:          maximum allowed velocity, if vehicle is currently faster, slowdown will be initiated
        :param gg_scale:         scaling factor to be applied on the gg-limits
        :param local_gg:         two available options:

                                 * LOCATION DEPENDENT FRICTION: dict of lat/long acceleration limits along the paths,
                                   each path coordinate [self.__last_action_set_path_param] must be represented by a row
                                   in the local_gg, with two columns (ax, ay)
                                 * CONSTANT FRICTION: provide a tuple with maximum allowed accelerations (ax, ay)

        :param ax_max_machines:  velocity dependent maximum acceleration provided by motor, columns [vx, ax]
        :param safety_d:         safety distance to be maintained to any lead vehicle
        :param incl_emerg_traj:  if set to 'true', a simple emerg. profile will be returned in the set (key 'emergency')
        :returns:
            * **action_set** -   dict holding list of trajectories for each action set {'straight': [np(...)], ...}
            * **action_set_id** - dict holding unique trajectory id for each action set {'straight': 123, ...}
            * **traj_time** -    time of trajectory generation (time.time())

        """

        self.__pos_est = pos_est

        # -- DETERMINE CUT INDEX BASED ON ACTUAL POSITION AND LAST PLANNED TRAJECTORY ----------------------------------
        self.__cut_index_pos, cut_layer, vel_plan, vel_course, acc_plan = \
            self.__oth.get_ref_idx(action_id_sel=self.__prev_action_id,
                                   idx_sel_traj=self.__prev_traj_idx,
                                   pos_est=self.__pos_est)

        # -- PREPARE TRAJECTORIES FOR EXPORT (trim to pos and calculate velocity profile) ------------------------------
        self.__action_set, self.__action_set_id, self.__traj_time, self.__local_trajectories = \
            self.__oth.calc_vel_profile(cut_index_pos=self.__cut_index_pos,
                                        cut_layer=cut_layer,
                                        vel_plan=vel_plan,
                                        acc_plan=acc_plan,
                                        vel_course=vel_course,
                                        vel_est=vel_est,
                                        vel_max=vel_max,
                                        gg_scale=gg_scale,
                                        local_gg=local_gg,
                                        ax_max_machines=ax_max_machines,
                                        safety_d=safety_d,
                                        incl_emerg_traj=incl_emerg_traj)

        # trim trajectory to number of export points (optimization based vel-planner is fixed to certain amount of vals)
        for action_id in self.__action_set.keys():
            # for every trajectory in action set
            for i in range(len(self.__action_set[action_id])):
                # write data to dict structure
                # NOTE: cut exported trajectory to specified length (e.g. determined by planned velocity volume)
                self.__action_set[action_id][i] = self.__action_set[action_id][i][:self.__nmbr_export_points, :]

        return self.__action_set, self.__action_set_id, self.__traj_time

    # ------------------------------------------------------------------------------------------------------------------

    def log(self) -> None:
        """
        Log graph relevant data to file (if parameterized).

        """

        if self.__log_to_file:
            # prepare data (extract required data, e.g. velocities for each individual trajectory)
            pos_list = copy.deepcopy(self.__action_set)
            vel_list = copy.deepcopy(self.__action_set)
            a_list = copy.deepcopy(self.__action_set)
            kappa_list = copy.deepcopy(self.__action_set)
            s_list = copy.deepcopy(self.__action_set)
            psi_list = copy.deepcopy(self.__action_set)
            for key in vel_list.keys():
                for temp_i, parameters in enumerate(vel_list[key]):
                    s_list[key][temp_i] = parameters[:, 0]
                    pos_list[key][temp_i] = parameters[:, 1:3]
                    vel_list[key][temp_i] = parameters[:, 5]
                    a_list[key][temp_i] = parameters[:, 6]
                    kappa_list[key][temp_i] = parameters[:, 4]
                    psi_list[key][temp_i] = parameters[:, 3]

            # get global s coordinate
            s_ego, _ = graph_ltpl.helper_funcs.src.get_s_coord. \
                get_s_coord(ref_line=self.__graph_base.raceline,
                            pos=tuple(self.__pos_est),
                            s_array=self.__graph_base.s_raceline,
                            closed=True)

            if self.__const_path_seg is not None:
                self.__const_path_seg = self.__const_path_seg[self.__cut_index_pos:, :]

            self.__graph_log_handler.log_onlinegraph(time=self.__traj_time,
                                                     s_coord=s_ego,
                                                     start_node=self.__plan_start_node,
                                                     obj_veh=self.__obj_veh,
                                                     obj_zone=self.__obj_zone,
                                                     nodes_list=self.__node_list,
                                                     s_list=s_list,
                                                     pos_list=pos_list,
                                                     vel_list=vel_list,
                                                     a_list=a_list,
                                                     psi_list=psi_list,
                                                     kappa_list=kappa_list,
                                                     traj_id=self.__action_set_id,
                                                     clip_pos=list(self.__pos_est),
                                                     action_id_prev=self.__prev_action_id,
                                                     traj_id_prev=self.__prev_traj_idx,
                                                     const_path_seg=self.__const_path_seg)

    # ------------------------------------------------------------------------------------------------------------------

    def visual(self) -> None:
        """
        Visualize graph update (if parameterized).

        """

        if self.__visual_mode:
            # plot extracted local path
            self.__graph_plot_handler.highlight_lines(self.__local_trajectories,
                                                      id_in="Local Path")

            # plot predictions
            self.__graph_plot_handler.update_obstacles(obstacle_pos_list=[obj.get_prediction()[-1, :] for obj in
                                                                          self.__obj_veh],
                                                       obstacle_radius_list=[obj.get_radius() for obj in
                                                                             self.__obj_veh],
                                                       object_id='Prediction',
                                                       color='grey')

            # plot obstacles
            self.__graph_plot_handler.update_obstacles(obstacle_pos_list=[x.get_pos() for x in self.__obj_veh],
                                                       obstacle_radius_list=[x.get_radius() for x in self.__obj_veh],
                                                       object_id='Objects')

            # plot patches for overtaking zones
            patch_xy_pos_list = []
            for obj in self.__obj_zone:
                bound_l, bound_r = obj.get_bound_coords()
                patch = np.vstack((bound_l, np.flipud(bound_r)))

                patch_xy_pos_list.append(patch)

            self.__graph_plot_handler.highlight_patch(patch_xy_pos_list=patch_xy_pos_list)

            # euclidean distances to all objects
            text_str = ""
            for i, vehicle in enumerate(self.__obj_veh):
                eucl_dist = np.linalg.norm(np.array(self.__pos_est) - np.array(vehicle.get_pos()))
                text_str += "Obj. " + str(i) + ": " + "%.2fm\n" % eucl_dist
            self.__graph_plot_handler.update_text_field(text_str=text_str,
                                                        text_field_id=2)

            # print selected action id
            self.__graph_plot_handler.update_text_field(text_str=self.__prev_action_id,
                                                        color_str='r')

            # highlight ego pos
            self.__graph_plot_handler.plot_vehicle(pos=self.__pos_est,
                                                   heading=next(iter(self.__action_set.values()))[0][0, 3],
                                                   width=2.0,
                                                   length=self.__graph_base.veh_length,
                                                   zorder=100,
                                                   color_str='darkorange')

            # highlight start node of planning phase
            try:
                s_pos = self.__graph_base.get_node_info(layer=self.__plan_start_node[0],
                                                        node_number=self.__plan_start_node[1],
                                                        active_filter=None)[0]
                self.__graph_plot_handler.highlight_pos(pos_coords=s_pos,
                                                        color_str='c',
                                                        zorder=5,
                                                        radius=2,
                                                        id_in='Start Node')
            except ValueError:
                pass

            self.__graph_plot_handler.show_plot(non_blocking=True)
