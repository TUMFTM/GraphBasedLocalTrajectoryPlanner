import numpy as np
import time
import json
import configparser
import logging

# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph

# mapping from action IDs to ints
ACTION_ID_MAP = {"straight": 0,
                 "follow": 1,
                 "left": 2,
                 "right": 3}


class OnlineTrajectoryHandler(object):
    """
    Class for online manipulation of trajectories. Temporarily stores data from previous iterations
    (allow warm start, etc.) and provides simple method interfaces.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        13.11.2018
    """

    def __init__(self,
                 graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                 graph_online_config_path: str,
                 graph_offline_config_path: str,
                 veh_param_dyn_model_exp: float = 1.0,
                 veh_param_dragcoeff: float = 0.85,
                 veh_param_mass: float = 1000.0) -> None:
        """
        :param graph_base:                  reference to GraphBase object instance holding relevant parameters
        :param graph_online_config_path:    string-path pointing to the online parameter file
        :param graph_offline_config_path:   string-path pointing to the offline parameter file
        :param veh_param_dyn_model_exp:     vehicle dynamics model exponent (range [1.0, 2.0])
        :param veh_param_dragcoeff:         drag coefficient -> 0.5 * rho_air * c_w_A in m2*kg/m3
        :param veh_param_mass:              vehicle mass in kg

        """

        # init logger handle
        self.__log = logging.getLogger("local_trajectory_logger")

        # load config files
        graph_config = configparser.ConfigParser()
        if not graph_config.read(graph_online_config_path):
            raise ValueError('Specified cost config file does not exist or is empty!')

        graph_config_offline = configparser.ConfigParser()
        if not graph_config_offline.read(graph_offline_config_path):
            raise ValueError('Specified offline config file does not exist or is empty!')

        # moving average calculation time
        self.__calc_buffer = []

        # initialize iterative memory
        self.__traj_base_id = 0
        self.__start_node = None
        self.__last_action_set_nodes = None
        self.__last_action_set_node_idx = None
        self.__last_action_set_coeff = None
        self.__last_action_set_path_param = None
        self.__last_action_set_path_gg = None
        self.__last_action_set_red_len = None
        self.__last_bp_action_set = None
        self.__last_path_timestamp = None
        self.__last_cut_idx = 0

        self.__pos_est = None

        # action id for the path used to generate the emergency profile
        self.__em_base_id = None

        # initialize backup plan
        self.__backup_nodes = None
        self.__backup_node_idx = None
        self.__backup_coeff = None
        self.__backup_path_param = None
        self.__backup_path_gg = None

        # keep graph_base reference
        self.__graph_base = graph_base

        # init object container
        self.__obj_veh = []
        self.__obj_zone = []

        self.__closest_obj_index = None

        # trajectory quantity parameters
        self.__max_solutions = graph_config.getint('ACTIONSET', 'max_solutions')
        self.__max_cost_diff = graph_config.getfloat('ACTIONSET', 'max_cost_diff')

        self.__v_max_offset = graph_config.getfloat('ACTIONSET', 'v_max_offset')

        self.__v_start = 0.0
        self.__action_id_forced = None

        self.__vx_filt_window = graph_config.getint('SMOOTHING', 'filt_window_width')

        self.__w_last_edges = json.loads(graph_config.get('COST', 'w_last_edges'))

        # follow mode controller parameters
        self.__follow_control_type = graph_config.get('FOLLOW', 'controller_type')
        self.__follow_control_params = \
            json.loads(graph_config.get('FOLLOW', 'control_params_' + self.__follow_control_type))

        # delay compensation
        self.__delaycomp = graph_config.getfloat('DELAY', 'delaycomp')

        # calculation time related
        self.__calc_time_warn_thr = graph_config.getfloat('CALC_TIME', 'calc_time_warn_threshold')
        self.__calc_time_safety = graph_config.getfloat('CALC_TIME', 'calc_time_safety')
        self.__calc_time_buffer_len = graph_config.getint('CALC_TIME', 'calc_time_buffer_len')

        # Init velocity planners
        vp_type = graph_config.get('VP', 'vp_type')

        # if import of qp-vel-planner failed force fb
        if vp_type == "sqp" and graph_ltpl.online_graph.src.VpSQP.get_import_failure_status():
            self.__log.warning("QP-based velocity planner import failed! Forced forward-backward planner.")
            vp_type = "fb"

        self.__vp_fb = None
        self.__vp_sqp = None

        if vp_type == "fb":
            # forward-backward velocity planner
            self.__vp_fb = graph_ltpl.online_graph.src.VpForwardBackward. \
                VpForwardBackward(dyn_model_exp=veh_param_dyn_model_exp,
                                  drag_coeff=veh_param_dragcoeff,
                                  m_veh=veh_param_mass,
                                  len_veh=self.__graph_base.veh_length,
                                  follow_control_type=self.__follow_control_type,
                                  follow_control_params=self.__follow_control_params,
                                  glob_rl=self.__graph_base.glob_rl)

        elif vp_type == "sqp":
            # --- SQP velocity planner
            self.__vp_sqp = graph_ltpl.online_graph.src.VpSQP. \
                VpSQP(nmbr_export_points=graph_config.getint('EXPORT', 'nmbr_export_points'),
                      stepsize_approx=graph_config_offline.getfloat('SAMPLING', 'stepsize_approx'),
                      veh_turn=graph_config_offline.getfloat('VEHICLE', 'veh_turn'),
                      glob_rl=graph_base.glob_rl,
                      delaycomp=self.__delaycomp)

        else:
            raise ValueError('No valid velocity planner specified!')

    def __del__(self):
        pass

    def reinit_iterative_memory(self) -> None:
        """
        Reset / initialize the iterative memory (must be executed, when initializing to a new pose).

        """

        # initialize iterative memory
        self.__start_node = None
        self.__last_action_set_nodes = None
        self.__last_action_set_node_idx = None
        self.__last_action_set_coeff = None
        self.__last_action_set_path_param = None
        self.__last_action_set_path_gg = None
        self.__last_action_set_red_len = None
        self.__last_bp_action_set = None
        self.__last_path_timestamp = None
        self.__last_cut_idx = 0

        self.__pos_est = None

    def set_initial_pose(self,
                         start_pos: list,
                         start_heading: float,
                         start_vel: float = 0.0,
                         max_heading_offset: float = np.pi / 4) -> tuple:
        """
        Set an initial pose for the trajectory planner and plan a trajectory leading into the grid.

        :param start_pos:           list holding x and y coordinate of the current position of the vehicle
        :param start_heading:       current orientation of the vehicle
        :param start_vel:           start velocity of the vehicle (to be initialized to each time)
        :param max_heading_offset:  maximum allowed heading offset of the vehicle in relation to the track layout
        :returns:
            * **in_track** -        boolean flag, 'True' if the provided start_pos is within the track bounds
            * **cor_heading** -     boolean flag, 'True' if the provided start_heading points in the correct direction

        """

        self.__v_start = start_vel

        in_track = True
        cor_heading = True

        self.reinit_iterative_memory()

        # -- CHECK IF WITHIN TRACK -------------------------------------------------------------------------------------
        # get bounds
        bound1 = (self.__graph_base.refline + self.__graph_base.normvec_normalized
                  * np.expand_dims(self.__graph_base.track_width_right, 1))
        bound2 = (self.__graph_base.refline - self.__graph_base.normvec_normalized
                  * np.expand_dims(self.__graph_base.track_width_left, 1))

        # Validate start position
        if not graph_ltpl.online_graph.src.check_inside_bounds.check_inside_bounds(bound1=bound1,
                                                                                   bound2=bound2,
                                                                                   pos=start_pos):
            self.__log.warning("Vehicle is out of track, check if correct reference line is provided!")
            in_track = False
            return in_track, cor_heading
            # raise ValueError("VEHICLE SEEMS TO BE OUT OF TRACK!")

        # -- SELECT INITIAL PLANNING NODE ------------------------------------------------------------------------------
        closest_nodes, distance = self.__graph_base.get_closest_nodes(pos=start_pos, limit=1)

        # determine goal node two layers ahead
        goal_layer = (closest_nodes[0][0] + 2) % (self.__graph_base.num_layers - 1)
        goal_node = self.__graph_base.raceline_index[goal_layer]

        self.__start_node = [goal_layer, goal_node]

        # extract cartesian information of goal node
        end_pos, end_heading, _, _, _ = self.__graph_base.get_node_info(layer=goal_layer,
                                                                        node_number=goal_node)
        heading_diff = abs(start_heading - end_heading)
        if heading_diff > np.pi:
            heading_diff = abs(2 * np.pi - heading_diff)
        if heading_diff > max_heading_offset:
            self.__log.warning("Heading mismatch between vehicle and track grid, check if vehicle oriented correctly!")
            cor_heading = False
            return in_track, cor_heading
            # raise ValueError("VEHICLE HEADING MISMATCH (TRACK <-> VEHICLE)!")

        # calculate spline to start node
        x_coeff, y_coeff, _, _ = tph.calc_splines.calc_splines(path=np.vstack((start_pos, end_pos)),
                                                               psi_s=start_heading,
                                                               psi_e=end_heading)

        path, inds, t_values, _ = tph.interp_splines.\
            interp_splines(coeffs_x=x_coeff,
                           coeffs_y=y_coeff,
                           stepsize_approx=self.__graph_base.sampled_resolution,
                           incl_last_point=True)

        psi, kappa = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=x_coeff,
                                                             coeffs_y=y_coeff,
                                                             ind_spls=inds,
                                                             t_spls=t_values)

        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))

        # set passed variables
        act_id = "straight"
        self.__action_id_forced = act_id

        self.__last_action_set_coeff = {act_id: [np.hstack((x_coeff, y_coeff))]}
        self.__last_action_set_path_param = {act_id: [np.column_stack((path, psi, kappa, np.append(el_lengths, 0)))]}
        self.__last_action_set_nodes = {act_id: [[[None, None], self.__start_node]]}
        self.__last_action_set_node_idx = {act_id: [[0, path.shape[0] - 1]]}

        return in_track, cor_heading

    def update_objects(self,
                       obj_veh: list,
                       obj_zone: list) -> None:
        """
        Updates the classes internal object representation.

        :param obj_veh:    list of objects of type "car", with info "pos", "ids", "vel", "radius"
        :param obj_zone:   list of objects of type "overtaking_zone_***", with "ids", "ref_pos", "norm_vec", "bound_l"

        """

        self.__obj_veh = obj_veh
        self.__obj_zone = obj_zone

        # reset closest object index
        self.__closest_obj_index = None

    def calc_paths(self,
                   action_id_sel: str,
                   idx_sel_traj: int) -> tuple:
        """
        Determine a proper split point on the last generated trajectory. The part of the trajectory until the split
        point is kept constant, since the vehicle continues to travel on the path during calculation of the next point.
        The split point is then used as "start_node" for the graph search evaluating the local plan. Finally, the path
        is reassembled (const. part) and returned.

        :param action_id_sel:       last executed action identifier string (e.g. "straight")
        :param idx_sel_traj:        last executed trajectory id within the chosen action set (e.g. "0")
        :returns:
            * **path_dict** -       dict holding paths for each of the available action sets {'straight': np[], ...}
            * **start_node** -      start node in the grid from which the path-search is executed (lies in front of veh)
            * **node_dict** -       dict holding node sequences in the graph for each available action set
            * **const_path_seg** -  path segment from the previous solution to the start node (kept constant)

        """

        # if action id is emergency, translate to action id the emergency profile was based on
        if action_id_sel == 'emergency':
            action_id_sel = self.__em_base_id

        # if action id is forced (e.g. init)
        if self.__action_id_forced is not None:
            action_id_sel = self.__action_id_forced
            self.__action_id_forced = None

        # --------------------------------------------------------------------------------------------------------------
        # - Status identifiers -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        const_path_seg_exists = (self.__last_action_set_path_param is not None
                                 and action_id_sel in self.__last_action_set_path_param.keys())
        planned_once = self.__last_path_timestamp is not None
        valid_solution_last_step = (planned_once and const_path_seg_exists
                                    and self.__last_bp_action_set[action_id_sel][idx_sel_traj].shape[0] > 2)

        # --------------------------------------------------------------------------------------------------------------
        # - Store last straight or follow trajectory as backup plan ----------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        if valid_solution_last_step:
            temp_id = "straight"
            if "follow" in self.__last_action_set_nodes.keys():
                temp_id = "follow"

            self.__backup_coeff = self.__last_action_set_coeff[temp_id][0]
            self.__backup_node_idx = self.__last_action_set_node_idx[temp_id][0]
            self.__backup_nodes = self.__last_action_set_nodes[temp_id][0]
            self.__backup_path_param = self.__last_action_set_path_param[temp_id][0]
            self.__backup_path_gg = self.__last_action_set_path_gg[temp_id][0]
        else:
            self.__backup_coeff = None
            self.__backup_node_idx = None
            self.__backup_nodes = None
            self.__backup_path_param = None
            self.__backup_path_gg = None

        # --------------------------------------------------------------------------------------------------------------
        # - Determine start node for graph search and fix first trajectory part (calculation time) ---------------------
        # --------------------------------------------------------------------------------------------------------------
        # If already generated (and found) a local trajectory in a previous time-step, estimate the expected pose on it
        # (using last calc duration) -> select next node as start for graph-search and leave initial part constant
        if planned_once and valid_solution_last_step:
            # extract calculation time for last iteration
            calc_time = time.time() - self.__last_path_timestamp
            self.__last_path_timestamp = time.time()

            # warn if calc time exceeds threshold
            if calc_time > self.__calc_time_warn_thr:
                self.__log.warning("Warning: One trajectory generation iteration took more than %.3fs (Actual "
                                   "calculation time: %.3fs)" % (self.__calc_time_warn_thr, calc_time))
            self.__log.debug("Update frequency: %.2f Hz" % (1.0 / max(calc_time, 0.001)))

            # get moving average of calc time (smooth out some outliers)
            if len(self.__calc_buffer) >= self.__calc_time_buffer_len:
                self.__calc_buffer.pop(0)
            self.__calc_buffer.append(calc_time)
            calc_time_avg = float(np.sum(self.__calc_buffer) / len(self.__calc_buffer))

            # get index of pose on last trajectory based on the previous calculation time and vel profile
            # divide element lengths by the corresponding velocity in order to obtain a time approximation
            s_past = np.diff(self.__last_bp_action_set[action_id_sel][idx_sel_traj][1:, 0])
            v_past = self.__last_bp_action_set[action_id_sel][idx_sel_traj][1:-1, 5]
            t_approx = np.divide(s_past, v_past, out=np.full(v_past.shape[0], np.inf), where=v_past != 0)

            # force constant trajectory for a certain amount of time (here: upper bounded calculation time)
            t_const = min(calc_time_avg * self.__calc_time_safety, 0.5)

            # find cumulative time value larger than estimated calculation time
            next_idx = (np.cumsum(t_approx) <= t_const).argmin() + 1

            # Get first node after "next_idx_corr"
            last_node_idx = self.__last_action_set_node_idx[action_id_sel][idx_sel_traj]
            node_coords = self.__last_action_set_path_param[action_id_sel][idx_sel_traj][last_node_idx, 0:2]
            predicted_pos = self.__last_bp_action_set[action_id_sel][idx_sel_traj][next_idx, 1:3]
            start_node_idx = graph_ltpl.helper_funcs.src.get_s_coord.get_s_coord(ref_line=node_coords,
                                                                                 pos=predicted_pos,
                                                                                 only_index=True)[1][1]

            loc_path_start_idx = self.__last_action_set_node_idx[action_id_sel][idx_sel_traj][start_node_idx]

            self.__start_node = self.__last_action_set_nodes[action_id_sel][idx_sel_traj][start_node_idx]

            # get nodes of last solution (used to reduce the cost on those segments)
            last_solution_nodes = self.__last_action_set_nodes[action_id_sel][idx_sel_traj][start_node_idx:]
        else:
            self.__last_path_timestamp = time.time()
            last_solution_nodes = None

            if const_path_seg_exists and self.__start_node in self.__last_action_set_nodes[action_id_sel][idx_sel_traj]:
                start_node_pos = self.__graph_base.get_node_info(layer=self.__start_node[0],
                                                                 node_number=self.__start_node[1])[0]
                loc_path_start_idx = graph_ltpl.helper_funcs.src.closest_path_index.\
                    closest_path_index(path=self.__last_action_set_path_param[action_id_sel][idx_sel_traj][:, 0:2],
                                       pos=start_node_pos)[0][0]
                start_node_idx = self.__last_action_set_nodes[action_id_sel][idx_sel_traj].index(self.__start_node)
            else:
                loc_path_start_idx = 0
                start_node_idx = 0

        # --------------------------------------------------------------------------------------------------------------
        # - Main online callback for trajectory planning (extract and prepare graph segment and exec. graph search) ----
        # --------------------------------------------------------------------------------------------------------------
        const_path_seg = None
        if const_path_seg_exists:
            const_path_seg = self.__last_action_set_path_param[action_id_sel][idx_sel_traj][:loc_path_start_idx + 1, :]

        action_set_nodes, action_set_node_idx, action_set_coeff, action_set_path_param, action_set_red_len, \
            self.__closest_obj_index = (graph_ltpl.online_graph.src.main_online_path_gen.
                                        main_online_path_gen(graph_base=self.__graph_base,
                                                             start_node=self.__start_node,
                                                             obj_veh=self.__obj_veh,
                                                             obj_zone=self.__obj_zone,
                                                             last_action_id=action_id_sel,
                                                             max_solutions=self.__max_solutions,
                                                             const_path_seg=const_path_seg,
                                                             pos_est=self.__pos_est,
                                                             last_solution_nodes=last_solution_nodes,
                                                             w_last_edges=self.__w_last_edges))

        # --------------------------------------------------------------------------------------------------------------
        # - Reassemble planned paths with constant trajectory part (not subjected to graph search) ---------------------
        # --------------------------------------------------------------------------------------------------------------
        # loop through all action sets
        for action_id in action_set_nodes.keys():
            # if the action set contains data
            if action_set_nodes[action_id]:
                # if a solution is found (else keep member variables from last iteration)
                if const_path_seg_exists:
                    # for every path (if multiple path solutions activated)
                    for i in range(len(action_set_nodes[action_id])):
                        if loc_path_start_idx > 0:
                            # path parameters
                            action_set_path_param[action_id][i] = np.concatenate((
                                self.__last_action_set_path_param[action_id_sel][idx_sel_traj][:loc_path_start_idx, :],
                                action_set_path_param[action_id][i]))

                            # element length
                            # Edge case: if cut index exactly at end of last planned action set member (length from
                            #            terminal coordinate to new spline missing - here set to "0" -> calc dist)
                            if np.size(self.__last_action_set_path_param[action_id_sel][idx_sel_traj], axis=0) == \
                                    loc_path_start_idx:
                                j = loc_path_start_idx - 1
                                action_set_path_param[action_id][i][j, 4] = np.sqrt(
                                    np.power(np.diff(action_set_path_param[action_id][i][j:j + 2, 0]), 2)
                                    + np.power(np.diff(action_set_path_param[action_id][i][j:j + 2, 1]), 2))

                        # Update node index reference (add the indexes of the nodes cut away beforehand and correct
                        # numbers according to segment added in the line above)
                        action_set_node_idx[action_id][i] = np.concatenate(
                            (np.array(self.__last_action_set_node_idx[action_id_sel][idx_sel_traj][:start_node_idx]),
                             (np.array(action_set_node_idx[action_id][i]) + loc_path_start_idx)))

                        # Update nodes and spline coefficients
                        if start_node_idx > 0:
                            # nodes
                            action_set_nodes[action_id][i] = np.concatenate(
                                (self.__last_action_set_nodes[action_id_sel][idx_sel_traj][:start_node_idx],
                                 action_set_nodes[action_id][i])).tolist()

                            # spline coefficients
                            action_set_coeff[action_id][i] = np.concatenate(
                                (self.__last_action_set_coeff[action_id_sel][idx_sel_traj][:start_node_idx],
                                 action_set_coeff[action_id][i]))

        # If all action sets are empty, print warning
        if not bool([a for a in action_set_nodes.values() if a != []]):
            self.__log.critical(
                "Could not find a path solution for any of the points in the given destination layer! "
                "Track useems to be blocked.")

            # if no path solution found, add constant path segment, if available
            if const_path_seg_exists and const_path_seg.shape[0] > 2:
                # increment loc_path_start_idx and start_node_idx, since we want to include the node in the path
                loc_path_start_idx += 1
                start_node_idx += 1

                if loc_path_start_idx > 0:
                    # path parameters
                    action_set_path_param[action_id_sel] = \
                        [self.__last_action_set_path_param[action_id_sel][idx_sel_traj][:loc_path_start_idx, :]]

                # Update node index reference (add the indexes of the nodes cut away beforehand and correct
                # numbers according to segment added in the line above)
                action_set_node_idx[action_id_sel] = \
                    [np.array(self.__last_action_set_node_idx[action_id_sel][idx_sel_traj][:start_node_idx])]

                # Update nodes and spline coefficients
                if start_node_idx > 0:
                    # nodes
                    action_set_nodes[action_id_sel] = \
                        [self.__last_action_set_nodes[action_id_sel][idx_sel_traj][:start_node_idx]]

                    # spline coefficients
                    action_set_coeff[action_id_sel] = \
                        [self.__last_action_set_coeff[action_id_sel][idx_sel_traj][:start_node_idx]]

                action_set_red_len[action_id_sel] = [True]

        # After combination of old and new calculations -> write "new" calculation to "old" memory
        self.__last_action_set_nodes = action_set_nodes
        self.__last_action_set_node_idx = action_set_node_idx
        self.__last_action_set_coeff = action_set_coeff
        self.__last_action_set_path_param = action_set_path_param
        self.__last_action_set_red_len = action_set_red_len

        # return characteristic parameters (e.g. for logging purposes)
        return self.__last_action_set_path_param, self.__start_node, self.__last_action_set_nodes, const_path_seg

    def get_ref_idx(self,
                    action_id_sel: str,
                    idx_sel_traj: int,
                    pos_est: tuple) -> tuple:
        """
        Determines the index of the current position in the last planned trajectory and extracts the planned velocity
        there.

        :param action_id_sel:       last executed action identifier string (e.g. "straight")
        :param idx_sel_traj:        last executed trajectory id within the chosen action set (e.g. "0")
        :param pos_est:             estimated position of the ego-vehicle
        :returns:
            * **cut_index_pos** -   index of estimated vehicle position in the generated trajectories
            * **cut_layer** -       index of the layer to be the first in the returned trajectory
            * **vel_plan** -        velocity of the ego vehicle according to the last planned and executed trajectory
            * **vel_course** -      segment of velocity course of last traj., which should stay constant (comp. delay)
            * **acc_plan** -        acceleration of the ego vehicle according to the last planned and executed traj.

        """

        self.__pos_est = pos_est

        # --------------------------------------------------------------------------------------------------------------
        # - Status identifiers -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        planned_once = self.__last_bp_action_set is not None
        valid_solution_last_step = (planned_once and action_id_sel in self.__last_bp_action_set.keys()
                                    and np.size(self.__last_bp_action_set[action_id_sel][idx_sel_traj], axis=0) > 0)
        valid_solution_this_step = len(list(self.__last_action_set_node_idx.keys())) > 0

        # --------------------------------------------------------------------------------------------------------------
        # - Get cut index and extract velocity -------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        if planned_once and valid_solution_last_step:
            last_path = self.__last_bp_action_set[action_id_sel][idx_sel_traj][:, 1:3]
            s_last_path = self.__last_bp_action_set[action_id_sel][idx_sel_traj][:, 0]

            # get cut index for selected trajectory (2 neighboring points on trajectory candidate rel. to "pos_est")
            idx_nb = graph_ltpl.helper_funcs.src.get_s_coord. \
                get_s_coord(ref_line=last_path,
                            pos=pos_est,
                            s_array=s_last_path,
                            only_index=True)[1]
            cut_index = idx_nb[0]

            # -- calculate velocity cut index (including delay compensation) -------------------------------------------
            # get last index below delay estimate
            s_past = np.diff(self.__last_bp_action_set[action_id_sel][idx_sel_traj][cut_index:, 0])
            v_past = self.__last_bp_action_set[action_id_sel][idx_sel_traj][cut_index:-1, 5]
            t_approx = np.divide(s_past, v_past, out=np.full(v_past.shape[0], np.inf), where=v_past != 0)

            # find last cumulative time value being below delay estimate
            vel_idx = min((np.cumsum(t_approx) <= self.__delaycomp).argmin() + 1, v_past.shape[0] - 1)

            vel_plan = self.__last_bp_action_set[action_id_sel][idx_sel_traj][cut_index + vel_idx, 5]
            acc_plan = self.__last_bp_action_set[action_id_sel][idx_sel_traj][cut_index + vel_idx, 6]
            vel_course = self.__last_bp_action_set[action_id_sel][idx_sel_traj][cut_index:cut_index + vel_idx, 5]

            # add to last cut index, since bp_action_sets started at actual position of vehicle
            cut_index_pos = self.__last_cut_idx + cut_index

            # Check on which spline segment position estimate is located (layer_index)
            if valid_solution_this_step:
                action_id_tmp = list(self.__last_action_set_node_idx.keys())[0]
                cut_layer = max(np.argmin(np.array(
                    self.__last_action_set_node_idx[action_id_tmp][0]) < cut_index_pos) - 2, 0)

                # Get corresponding coordinate cut index
                cut_index_layer = self.__last_action_set_node_idx[action_id_tmp][0][cut_layer]
            else:
                cut_layer = 0
                cut_index_layer = 0
        else:
            cut_index_pos = 0
            cut_layer = 0
            cut_index_layer = 0
            vel_course = np.array([])
            vel_plan = self.__v_start
            acc_plan = 0.0

        # update cut index memory
        self.__last_cut_idx = cut_index_pos - cut_index_layer

        return cut_index_pos, cut_layer, vel_plan, vel_course, acc_plan

    def calc_vel_profile(self,
                         cut_index_pos: int,
                         cut_layer: int,
                         vel_plan: float,
                         acc_plan: float,
                         vel_course: np.ndarray,
                         vel_est: float,
                         vel_max: float,
                         ax_max_machines: np.ndarray,
                         safety_d: float,
                         gg_scale: float,
                         local_gg: dict = (5.0, 5.0),
                         incl_emerg_traj: bool = False) -> tuple:
        """
        Trims and returns the previously generated trajectory "__last_path_coord" and returns a relevant segment
        starting near the estimated pose.

        ..Note:: The trim process is also important for the next iteration of trajectory generation (determination of
            the part of the trajectory that should stay constant)

        :param cut_index_pos:   index of estimated vehicle position in the generated trajectories
        :param cut_layer:       index of the layer to be the first in the returned trajectory
        :param vel_plan:        velocity of the ego vehicle according to the last plan
        :param acc_plan:        acceleration of the ego vehicle according to the last plan
        :param vel_course:      velocity course of last trajectory, which should stay constant (comp. delay till exec.)
        :param vel_est:         estimated velocity of the ego-vehicle
        :param vel_max:         maximum velocity allowed by the behavior planner
        :param ax_max_machines: velocity dependent maximum acceleration allowed by the machine (given by BHPL)
        :param safety_d:        safety distance in meters to a (potential) lead vehicle (bumper to bumper)
        :param gg_scale:        gg-scale in range [0.0, 1.0] applied to [ax_max, ay_max]
        :param local_gg:        two available options:

                                * LOCATION DEPENDENT FRICTION: dict of lat/long acceleration limits along the paths,
                                  each path coordinate [self.__last_action_set_path_param] must be represented by a row
                                  in the local_gg, with two columns (ax, ay)
                                * CONSTANT FRICTION: provide a tuple with maximum allowed accelerations (ax, ay)
        :param incl_emerg_traj: if set to 'true', a simple emerg. profile will be returned in the set (key 'emergency')
        :returns:
            * **action_set** -          returned action set - dict with each key holding an action primitive, each value
              holds a list of one or multiple trajectories each as numpy array with columns s, x, y, psi, kappa, vx, ax
            * **action_set_path_id** -  dict holding a unique trajectory id for each action set
            * **traj_time_stamp** -     trajectory generation time-stamp
            * **path_coord_list** -     list of path coordinates for each trajectory (e.g. for path visualization)

        """

        # - Handle local gg input --------------------------------------------------------------------------------------
        # if constant acceleration provided
        if type(local_gg) is not dict:
            if type(local_gg) is not tuple or len(local_gg) != 2:
                raise ValueError("Provided local_gg does not satisfy requested format! Read parameter documentation.")

            # convert to path gg's
            gg_bounds = tuple(local_gg)
            local_gg = dict()
            # for all available action sets
            for action_id in self.__last_action_set_path_param.keys():

                local_gg[action_id] = []

                # for every trajectory in action set
                for i in range(len(self.__last_action_set_path_param[action_id])):
                    local_gg[action_id].append(np.ones((self.__last_action_set_path_param[action_id][i].shape[0], 2))
                                               * gg_bounds)

        # update trajectory base-ID
        self.__traj_base_id += 10

        # trajectory generation time-stamp (used across all modules)
        traj_time_stamp = time.time()

        # provide updated dynamic vehicle parameters to velocity planner classes
        if self.__vp_fb is not None:
            self.__vp_fb.update_dyn_parameters(vel_max=vel_max,
                                               gg_scale=gg_scale,
                                               ax_max_machines=ax_max_machines)
        elif self.__vp_sqp is not None:
            self.__vp_sqp.update_dyn_parameters(vel_max=vel_max)

        self.__last_bp_action_set = dict()
        action_set_path_param_vel = dict()
        action_set_path_param_gg = dict()
        self.__last_action_set_path_gg = dict()
        action_set_path_id = dict()

        for action_id in list(self.__last_action_set_path_param.keys()):
            # init action set in dict
            self.__last_bp_action_set[action_id] = []
            action_set_path_param_vel[action_id] = []
            action_set_path_param_gg[action_id] = []
            self.__last_action_set_path_gg[action_id] = []

            # get ID for current action set
            action_set_path_id[action_id] = (self.__traj_base_id
                                             + ACTION_ID_MAP.get(action_id, 9))

            # for every trajectory in action set
            for i in range(len(self.__last_action_set_path_param[action_id])):
                # ------------------------------------------------------------------------------------------------------
                # - Cut trajectories based on given pose estimate (passed spline segments) -----------------------------
                # ------------------------------------------------------------------------------------------------------

                # cut path parameters and local gg for velocity planner (at current pose)
                action_set_path_param_vel[action_id].append(
                    self.__last_action_set_path_param[action_id][i][cut_index_pos:, :])

                action_set_path_param_gg[action_id].append(local_gg[action_id][i][cut_index_pos:, :])

                # Get coordinate cut index from cut_layer
                cut_index_layer = self.__last_action_set_node_idx[action_id][i][cut_layer]

                # Update node index reference (reduce values by amount of cut number)
                self.__last_action_set_node_idx[action_id][i] = \
                    np.array(self.__last_action_set_node_idx[action_id][i][cut_layer:]) - cut_index_layer

                # Cut path parameters for next iteration (aligned with nodes)
                self.__last_action_set_path_param[action_id][i] = \
                    self.__last_action_set_path_param[action_id][i][cut_index_layer:, :]

                # store backup of local gg
                self.__last_action_set_path_gg[action_id].append(local_gg[action_id][i][cut_index_layer:, :])

                # Cut coeffs (aligned with nodes)
                self.__last_action_set_coeff[action_id][i] = \
                    self.__last_action_set_coeff[action_id][i][cut_layer:, :]

                # Cut nodes
                self.__last_action_set_nodes[action_id][i] = \
                    self.__last_action_set_nodes[action_id][i][cut_layer:]

                # ------------------------------------------------------------------------------------------------------
                # - Calculate velocity profile (if set is not empty) ---------------------------------------------------
                # ------------------------------------------------------------------------------------------------------
                bp_out = []
                vel_bound = True
                if np.size(action_set_path_param_vel[action_id][i], axis=0) > 0:
                    # get velocity profile offset based on delay
                    vel_idx = vel_course.shape[0]

                    # calculate s coordinates along given path
                    s = np.concatenate(([0], np.cumsum(action_set_path_param_vel[action_id][i][:-1, 4])))

                    # -- DETERMINE VELOCITY PROFILE PREFIX -------------------------------------------------------------
                    if self.__vp_fb is not None:
                        vx_prefix, pref_idx_add, vel_start = self.__vp_fb.\
                            check_brake_prefix(vel_plan=vel_plan,
                                               vel_course=vel_course,
                                               kappa=action_set_path_param_vel[action_id][i][vel_idx:, 3],
                                               el_lengths=action_set_path_param_vel[action_id][i][vel_idx:-1, 4],
                                               loc_gg=action_set_path_param_gg[action_id][i][vel_idx:, :])

                        pref_idx = vel_idx + pref_idx_add
                    else:
                        # No prefix required for other planners, copy plain values
                        # vx_prefix = vel_course
                        pref_idx = vel_idx
                        vel_start = vel_plan
                        # v_max = vel_max

                    # -- CALC VELOCITY PROFILE FOR FOLLOW MODE ---------------------------------------------------------
                    if action_id == "follow":
                        # check for valid "closest_obj_index" (if "None" in follow mode, sth. went wrong!)
                        if self.__closest_obj_index is None:
                            obj_dist = 0.0  # brake immediately
                            c_obj_vel = 0.0
                        else:
                            # find coordinate within path array, which is closest to upcoming object
                            c_obj_pos = self.__obj_veh[self.__closest_obj_index].get_pos()
                            c_obj_vel = self.__obj_veh[self.__closest_obj_index].get_vel()

                            # calculate distance to leading vehicle
                            s_obj, _ = graph_ltpl.helper_funcs.src.get_s_coord.\
                                get_s_coord(ref_line=action_set_path_param_vel[action_id][i][:, 0:2],
                                            pos=c_obj_pos,
                                            s_array=np.cumsum(action_set_path_param_vel[action_id][i][:, 4]))

                            s_start, _ = graph_ltpl.helper_funcs.src.get_s_coord.\
                                get_s_coord(ref_line=action_set_path_param_vel[action_id][i][:, 0:2],
                                            pos=self.__pos_est,
                                            s_array=np.cumsum(action_set_path_param_vel[action_id][i][:, 4]))

                            obj_dist = s_obj - s_start

                        # get velocity profile for follow mode
                        if self.__vp_fb is not None:
                            # calculate follow profile with fb solver
                            vx, too_close, vel_bound = self.__vp_fb.calc_vel_profile_follow(
                                kappa=action_set_path_param_vel[action_id][i][pref_idx:, 3],
                                el_lengths=action_set_path_param_vel[action_id][i][pref_idx:, 4],
                                loc_gg=action_set_path_param_gg[action_id][i][pref_idx:, :],
                                v_start=vel_start,
                                v_ego=vel_est,
                                v_obj=c_obj_vel,
                                safety_d=safety_d,
                                obj_dist=obj_dist,
                                obj_pos=c_obj_pos)

                        elif self.__vp_sqp is not None:
                            # calculate follow profile with sqp solver
                            vx, too_close, vel_bound = self.__vp_sqp.calc_vel_profile_follow(
                                action_id=action_id,
                                s_glob=(graph_ltpl.helper_funcs.src.get_s_coord.
                                        get_s_coord(ref_line=self.__graph_base.raceline,
                                                    pos=action_set_path_param_vel[action_id][i][0, 0:2])[0]),
                                kappa=action_set_path_param_vel[action_id][i][pref_idx:, 3],
                                el_lengths=action_set_path_param_vel[action_id][i][pref_idx:, 4],
                                loc_gg=action_set_path_param_gg[action_id][i][pref_idx:, :],
                                v_obj=c_obj_vel,
                                vel_plan=vel_plan,
                                acc_plan=acc_plan,
                                safety_d=safety_d,
                                obj_dist=obj_dist,
                                veh_length=self.__graph_base.veh_length,
                                v_max_offset=self.__v_max_offset)

                        else:
                            raise ValueError("Requested velocity planner not defined!")

                        if too_close:
                            self.__log.warning("Too close to object! Entering safety distance... [Follow-Mode]")

                        x_y_psi_kappa = action_set_path_param_vel[action_id][i][:, 0:4]
                        vx = np.concatenate((vel_course, vx))

                        # if kappa-profile was too short, cut too many entries in optimized v-profile
                        if vx.shape[0] > s.shape[0]:
                            vx = vx[0:len(s)]
                        bp_out = np.column_stack((s, x_y_psi_kappa, vx))

                    # -- CALC VEL PROFILE FOR ALL ACTION PRIMITIVES EXCEPT FOLLOW --------------------------------------
                    # all action profiles except follow (or follow and reduced planning horizon -> take min. of both)
                    if action_id != "follow" or \
                            (action_id == "follow" and self.__last_action_set_red_len[action_id][i]):
                        # --- get end velocity ---
                        # check lateral distance of goal point to raceline
                        end_node = self.__last_action_set_nodes[action_id][i][-1]
                        num_el = len(action_set_path_param_vel[action_id][i][:, 4])

                        raceline_index = self.__graph_base.raceline_index[end_node[0]]

                        raceline_offset = abs(end_node[1] - raceline_index) * self.__graph_base.lat_offset

                        # determine end velocity
                        if self.__last_action_set_red_len[action_id][i]:
                            # with reduced planning horizon, set end velocity to zero
                            v_end = 0.0

                            # force zero velocity profile on last 5 meters
                            spl_len = np.sum(action_set_path_param_vel[action_id][i][:-1, 4])

                            # find last index fulfilling 5.0m distance to end constraint
                            v_idx = np.argmin(np.cumsum(action_set_path_param_vel[action_id][i][:-1, 4])
                                              < (spl_len - 5.0)) + 1

                            # if no valid solution found
                            if v_idx == 1 and num_el > 1:
                                v_idx = num_el
                        else:
                            # extract velocity form raceline profile
                            v_end = self.__graph_base.vel_raceline[end_node[0]]

                            # calculate reduction due to lateral displacement
                            v_end -= min(v_end * self.__graph_base.vel_decrease_lat * raceline_offset, v_end)

                            v_idx = num_el

                        # calculate velocity profile
                        if v_idx - pref_idx > 1:
                            if self.__vp_fb is not None:
                                vx = self.__vp_fb.calc_vel_profile(
                                    kappa=action_set_path_param_vel[action_id][i][pref_idx:v_idx, 3],
                                    el_lengths=action_set_path_param_vel[action_id][i][pref_idx:v_idx - 1, 4],
                                    loc_gg=action_set_path_param_gg[action_id][i][pref_idx:v_idx, :],
                                    v_start=vel_start,
                                    v_end=v_end
                                )

                            elif self.__vp_sqp is not None:
                                vx = self.__vp_sqp.calc_vel_profile(
                                    action_id=action_id,
                                    s_glob=(graph_ltpl.helper_funcs.src.get_s_coord.
                                            get_s_coord(ref_line=self.__graph_base.raceline,
                                                        pos=action_set_path_param_vel[action_id][i][0, 0:2])[0]),
                                    s=s,
                                    kappa=action_set_path_param_vel[action_id][i][pref_idx:v_idx, 3],
                                    el_lengths=action_set_path_param_vel[action_id][i][pref_idx:v_idx - 1, 4],
                                    loc_gg=action_set_path_param_gg[action_id][i][pref_idx:v_idx, :],
                                    vel_plan=vel_plan,
                                    acc_plan=acc_plan
                                )

                            else:
                                raise ValueError("Requested velocity planner not defined!")

                        else:
                            vx = [0.0]

                        # Append zeros if reduced planning horizon
                        if v_idx != num_el or v_idx <= 2:
                            print("Modification of the vx-profile due to reduced planning horizon!")
                            vx = np.append(vx, [0.0] * (num_el - v_idx))

                        # check if velocity profile is a valid solution
                        vel_bound = True
                        if not abs(vx[0] - vel_plan) < self.__v_max_offset:
                            self.__log.warning("Velocity profile generation did not match the inserted boundary "
                                               "conditions! (Action Set: " + action_id + ", "
                                               "Offset: %.2fm/s)" % (vx[0] - vel_plan))
                            vel_bound = False

                        # Assemble proper trajectory format
                        x_y_psi_kappa = action_set_path_param_vel[action_id][i][:, 0:4]

                        vx = np.concatenate((vel_course, vx))[:num_el]

                        if action_id != "follow":
                            bp_out = np.column_stack((s, x_y_psi_kappa, vx))
                        else:
                            bp_out2 = np.column_stack((s, x_y_psi_kappa, vx))

                            bp_out = np.where(bp_out[5, :] < bp_out2[5, :], bp_out, bp_out2)

                    # -- FINALIZE VEL PROFILE AND CALCULATE ACCELERATION -----------------------------------------------
                    # filter vx profile
                    if self.__vp_fb is not None:
                        vx_f = tph.conv_filt.conv_filt(signal=bp_out[:, 5],
                                                       filt_window=self.__vx_filt_window,
                                                       closed=False)
                    else:
                        vx_f = vx

                    # calculate ax profile
                    ax_f = tph.calc_ax_profile.calc_ax_profile(vx_profile=vx_f,
                                                               el_lengths=np.diff(bp_out[:, 0]))

                    # set negative longitudinal acceleration for vx_profile == 0.0
                    ax_f[np.logical_and(np.isclose(vx_f[:-1], 0.0), np.isclose(ax_f, 0.0))] = -5.0

                    bp_out = np.column_stack((bp_out[:, :-1], vx_f, np.append(ax_f, [0.0])))

                # -- ASSEMBLE ACTION SET -------------------------------------------------------------------------------
                # do not add to action set, if velocity bounds not satisfied
                if vel_bound or action_id in ["follow", "straight"]:

                    if vel_bound or self.__backup_nodes is None:
                        self.__last_bp_action_set[action_id].append(bp_out)
                    else:
                        # -- HANDLE RECURSIVE INFEASIBILITY ------------------------------------------------------------
                        # if velocity constraints not fulfilled in straight action set
                        # -> stick to old path with full deceleration profile
                        self.__log.warning("Detected iterative infeasibility and triggered deceleration on old path!")

                        # copy backup profile
                        self.__last_action_set_node_idx[action_id][i] = \
                            np.array(self.__backup_node_idx[cut_layer:]) - cut_index_layer
                        self.__last_action_set_path_param[action_id][i] = self.__backup_path_param[cut_index_layer:, :]
                        self.__last_action_set_path_gg[action_id][i] = self.__backup_path_gg[cut_index_layer:, :]
                        self.__last_action_set_coeff[action_id][i] = self.__backup_coeff[cut_layer:, :]
                        self.__last_action_set_nodes[action_id][i] = self.__backup_nodes[cut_layer:]

                        # calculate brake profile
                        i = vel_course.shape[0]
                        if self.__vp_fb is not None:
                            # calculate brake profile on backup path (prev. solution) with fb solver
                            vx = self.__vp_fb.calc_vel_brake_em(
                                loc_gg=self.__backup_path_gg[cut_index_pos + i:, :],
                                kappa=self.__backup_path_param[cut_index_pos + i:, 3],
                                el_lengths=self.__backup_path_param[(cut_index_pos + i):-1, 4],
                                v_start=vel_plan,
                            )
                        elif self.__vp_sqp is not None:
                            # calculate brake profile on backup path (prev. solution) with sqp solver
                            vx = self.__vp_sqp.calc_vel_brake_em(
                                loc_gg=self.__backup_path_gg[cut_index_pos + i:, :],
                                kappa=self.__backup_path_param[cut_index_pos + i:, 3],
                                el_lengths=self.__backup_path_param[(cut_index_pos + i):-1, 4],
                                v_start=vel_plan,
                            )

                        # Assemble proper trajectory format
                        x_y_psi_kappa = self.__backup_path_param[cut_index_pos:, 0:4]
                        vx = np.concatenate((vel_course, vx))

                        if self.__vp_fb is not None:
                            # filter vx profile
                            vx_f = tph.conv_filt.conv_filt(signal=vx,
                                                           filt_window=self.__vx_filt_window,
                                                           closed=False)

                        else:
                            vx_f = vx

                        # calculate ax profile
                        ax_f = tph.calc_ax_profile. \
                            calc_ax_profile(vx_profile=vx_f,
                                            el_lengths=self.__backup_path_param[cut_index_pos:-1, 4])

                        # set negative longitudinal acceleration for vx_profile == 0.0
                        ax_f[np.logical_and(np.isclose(vx_f[:-1], 0.0), np.isclose(ax_f, 0.0))] = -5.0

                        s = np.concatenate(([0], np.cumsum(self.__backup_path_param[cut_index_pos:-1, 4])))
                        bp_out = np.column_stack((s, x_y_psi_kappa, vx_f, np.append(ax_f, [0.0])))

                        self.__last_bp_action_set[action_id].append(bp_out)
                else:
                    self.__log.warning("Removed action set, since vel constraints were "
                                       "broken! (Action Set: " + action_id + ")")
                    # self.__last_bp_action_set[action_id].append([]) (do not append for BP)
                    self.__last_action_set_coeff[action_id][i] = []
                    self.__last_action_set_path_param[action_id][i] = []
                    self.__last_action_set_path_gg[action_id][i] = []
                    self.__last_action_set_nodes[action_id][i] = []
                    self.__last_action_set_node_idx[action_id][i] = []

            # remove action id from dicts, if no solution present anymore
            if not any([bool(traj) for traj in self.__last_action_set_nodes[action_id]]):
                self.__last_action_set_coeff.pop(action_id)
                self.__last_action_set_path_param.pop(action_id)
                self.__last_action_set_path_gg.pop(action_id)
                self.__last_action_set_nodes.pop(action_id)
                self.__last_action_set_node_idx.pop(action_id)
                self.__last_action_set_red_len.pop(action_id)
                self.__last_bp_action_set.pop(action_id)

        # append emergency trajectory, if requested
        if incl_emerg_traj:
            self.__em_base_id = list(self.__last_bp_action_set.keys())[0]
            self.__last_bp_action_set['emergency'] = [
                (graph_ltpl.helper_funcs.src.calc_brake_emergency.
                 calc_brake_emergency(traj=self.__last_bp_action_set[self.__em_base_id][0],
                                      loc_gg=action_set_path_param_gg[self.__em_base_id][0]))]
            action_set_path_id['emergency'] = action_set_path_id[self.__em_base_id]

        # return list of all coordinate arrays (visualization)
        path_coord_list = \
            [item[:, 1:3] for sublist in list(self.__last_bp_action_set.values()) for item in sublist]

        return self.__last_bp_action_set, action_set_path_id, traj_time_stamp, path_coord_list
