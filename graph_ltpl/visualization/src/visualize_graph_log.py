import copy
import os
import io
import glob
import pickle
import json
import sys
import os.path as osfuncs
import warnings
import numpy as np
import datetime
import matplotlib.pyplot as plt

# custom packages
# add path (if called outside of project)
mod_local_traj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(mod_local_traj_path)
import graph_ltpl.visualization.src.PlotHandler


# -- MODIFIED UNPICKLE -------------------------------------------------------------------------------------------------
# provide support for old log files by converting to the new structure
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if "data_objects" in module and "graph_ltpl" not in module:
            renamed_module = module.replace("data_objects", "graph_ltpl.data_objects")

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


"""
Script which allows to visualize online graph log-files recorded beforehand during an online run. In order to visualize
a log, simply call this script with the path to a log file as argument. When the run was executed locally, a call of the
script without any argument will display the latest run.

:Authors:
    * Tim Stahl <tim.stahl@tum.de>

:Created on:
    22.11.2018
"""

# alter the color of the edges based on their cost
COST_DEPENDENT_COLOR = True

# plot all existing edges in the graph (if set to "False" only edges at the mouse pointer are highlighted)
VISUALIZE_ALL_EDGES = False

# recalculate solution based on log environment and validate against log information
RECALC_VALIDATION = False

# compatible graph version (print warning, if restored graph version does not match this version)
GRAPH_VERSION_SUPPORT = 0.2


def get_data_from_line(file_path_in: str,
                       line_num: int):
    # skip header
    line_num = line_num + 1

    # extract a certain line number (based on time_stamp)
    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)

        # get header (":-1" in order to remove tailing newline character)
        file.readline()
        header = file.readline()[:-1]

        # extract line
        line = ""
        for _ in range(line_num):
            line = file.readline()

        # parse the data objects we want to retrieve from that line
        data = dict(zip(header.split(";"), line.split(";")))

        # decode
        # start_node;obstacle_pos_list;obstacle_radius_list;nodes_list;clip_pos
        start_node = json.loads(data['start_node'])
        obj_veh_data = json.loads(data['obj_veh'])

        if 'obj_virt' in data.keys():
            obj_virt_data = json.loads(data['obj_virt'])
        else:
            obj_virt_data = []

        obj_zone_data = json.loads(data['obj_zone'])
        nodes_list = json.loads(data['nodes_list'])
        clip_pos = json.loads(data['clip_pos'])

        s_list = json.loads(data['s_list'])
        if 'pos_list' in data.keys():
            pos_list = json.loads(data['pos_list'])
        else:
            pos_list = None

        vel_list = json.loads(data['vel_list'])

        if 'a_list' in data.keys():
            a_list = json.loads(data['a_list'])
        else:
            a_list = None
        psi_list = json.loads(data['psi_list'])
        kappa_list = json.loads(data['kappa_list'])

        if 'traj_id' in data.keys():
            traj_id = json.loads(data['traj_id'])
        else:
            traj_id = dict()

        # read action id and trajectory id (just in case we will reach the end of file
        action_id_prev = json.loads(data['action_id_prev'])
        traj_sel_idx = json.loads(data['traj_id_prev'])

        const_path_seg = np.array(json.loads(data['const_path_seg']))

        # get action selector
        line = file.readline()
        data = dict(zip(header.split(";"), line.split(";")))

        if not line == '':
            action_id = json.loads(data['action_id_prev'])
            traj_sel_idx = json.loads(data['traj_id_prev'])
        else:
            action_id = action_id_prev

    return (start_node, obj_veh_data, obj_virt_data, obj_zone_data, nodes_list, s_list, pos_list, vel_list, a_list,
            psi_list, kappa_list, traj_id, clip_pos, action_id, action_id_prev, traj_sel_idx, const_path_seg)


class DebugHandler(object):

    def __init__(self,
                 time_stamps_data: list,
                 time_stamps_msgs: list,
                 time_msgs_types: list,
                 time_msgs_content: list):
        self.__n_store = None
        self.__working = False

        self.__time_stamps_data = time_stamps_data
        self.__time_stamps_msgs = time_stamps_msgs
        self.__time_msgs_types = time_msgs_types
        self.__time_msgs_content = time_msgs_content

        # calculate thresholds to set the maximum allowed cursor offset
        self.__time_threshold_data = np.mean(np.diff(self.__time_stamps_data)) * 2
        self.__time_threshold_msgs = np.mean(np.diff(self.__time_stamps_msgs)) * 2

    def plot_timestamp_n(self,
                         plot_handler_in,
                         graph_base_in,
                         n: int):
        if not self.__n_store == n:
            self.__n_store = n
            (start_node, obj_veh_data, obj_virt_data, obj_zone_data, nodes_list, s_list, pos_list, vel_list, a_list,
             psi_list, kappa_list, traj_id, clip_pos, action_id, action_id_prev, traj_sel_idx, const_path_seg) = \
                get_data_from_line(file_path_data, n)

            # check if object zone is provided, else use time reference
            if obj_zone_data and obj_zone_data[0] == "no update since":
                # search line matching the timestamp
                obj_zone_data = get_data_from_line(file_path_data,
                                                   self.__time_stamps_data.index(float(obj_zone_data[1])))[3]

            # reconstruct object lists
            obj_zone = []
            for obj_data in obj_zone_data:
                zone_obj = graph_ltpl.data_objects.ObjectListInterface.ZoneObject(id_in="0",
                                                                                  blocked_layer_ids_in=obj_data[0][0],
                                                                                  blocked_node_ids_in=obj_data[0][1],
                                                                                  bound_l_coord_in=obj_data[1][0],
                                                                                  bound_r_coord_in=obj_data[1][1])
                zone_obj.set_fixed()
                obj_zone.append(zone_obj)

            obj_veh = []
            for obj_data in obj_veh_data:
                veh_obj = graph_ltpl.data_objects.ObjectListInterface.\
                    VehObject(id_in=obj_data[0],
                              pos_in=obj_data[1],
                              psi_in=obj_data[2],
                              radius_in=obj_data[3],
                              vel_in=obj_data[4],
                              prediction_in=np.atleast_2d(obj_data[5]))
                obj_veh.append(veh_obj)

            # calculate time axis for path dependent velocity profile
            t_list = copy.deepcopy(s_list)
            for key in s_list.keys():
                for temp_i, parameters in enumerate(s_list[key]):
                    temp_vel = np.array(vel_list[key][temp_i])
                    vel_avg = 0.5 * (temp_vel[1:] + temp_vel[:-1])
                    t_list[key][temp_i] = np.cumsum(
                        np.concatenate(([0], np.diff(s_list[key][temp_i])
                                        / np.where(np.abs(vel_avg) < 0.001, 0.001, vel_avg))))

            # -- rerun online search and compare against logs ----------------------------------------------------------
            if RECALC_VALIDATION or pos_list is None:
                # execute online process (masking graph and introducing obstacles)
                loc_path_nodes_list, _, _, loc_path_param_list, _, _ = graph_ltpl.online_graph.src.\
                    main_online_path_gen.main_online_path_gen(graph_base=graph_base_in,
                                                              start_node=start_node,
                                                              obj_veh=obj_veh,
                                                              obj_virt=[],
                                                              obj_zone=obj_zone,
                                                              last_action_id=action_id_prev,
                                                              max_solutions=1)

                # if pos_list is not in log file, use generated one
                if pos_list is None:
                    pos_list = copy.deepcopy(loc_path_param_list)
                    for key in loc_path_param_list.keys():
                        for temp_i, parameters in enumerate(loc_path_param_list[key]):
                            pos_list[key][temp_i] = parameters[:, 0:2]

                # verify that stored nodes list equals the calculated one
                if not np.array_equal(nodes_list, loc_path_nodes_list):
                    print("Could not restore the exact same trajectory as stored in the logs!")
                    print("Commonly the stored node list has an additional node in the beginning (const. path seg.).")
                    print(" -> Start node in the logs: " + str(start_node))
                    print(" -> Node list in the logs:  " + str(nodes_list))
                    print(" -> Generated node list:    " + str(loc_path_nodes_list))

            # -- get coordinates of nodes in log -----------------------------------------------------------------------
            nodes_coords_log = []
            for nodes in nodes_list.values():
                temp_nodes_coords = np.zeros((len(nodes[0]), 2))
                for n, node in enumerate(nodes[0]):
                    if None not in node:
                        temp_nodes_coords[n, :] = graph_base_in.get_node_info(layer=node[0],
                                                                              node_number=node[1],
                                                                              active_filter=None)[0]
                    else:
                        temp_nodes_coords[n, :] = [None, None]

                nodes_coords_log.append(temp_nodes_coords)

            # -- plot gained information -------------------------------------------------------------------------------
            vel_info_list = [np.column_stack((t_item, v_item))
                             for v_sublist, t_sublist in zip(list(vel_list.values()), list(t_list.values()))
                             for v_item, t_item in zip(v_sublist, t_sublist)]
            kappa_info_list = [np.column_stack((t_item, k_item))
                               for k_sublist, t_sublist in zip(list(kappa_list.values()), list(t_list.values()))
                               for k_item, t_item in zip(k_sublist, t_sublist)]
            psi_info_list = [np.column_stack((t_item, np.array(k_item) / (10 * np.pi)))
                             for k_sublist, t_sublist in zip(list(psi_list.values()), list(t_list.values()))
                             for k_item, t_item in zip(k_sublist, t_sublist)]
            plot_handler_in.plot_time_rel_line(line_coords_list=[vel_info_list, kappa_info_list, psi_info_list])

            # plot calculated paths
            plot_handler_in.highlight_lines(line_coords_list=nodes_coords_log,
                                            id_in='Node Solution',
                                            color_base=[1, 1, 0])

            plot_handler_in.highlight_lines(line_coords_list=list(pos_list.values())[0],
                                            id_in='Local Path')

            # plot prediction
            plot_handler_in.update_obstacles(obstacle_pos_list=[obj.get_prediction()[-1, :] for obj in obj_veh],
                                             obstacle_radius_list=[obj.get_radius() for obj in obj_veh],
                                             object_id='Prediction',
                                             color='grey')

            # plot regular objects
            plot_handler_in.update_obstacles(obstacle_pos_list=[x.get_pos() for x in obj_veh],
                                             obstacle_radius_list=[x.get_radius() for x in obj_veh],
                                             object_id='Objects')

            # plot virtual objects
            plot_handler_in.update_obstacles(obstacle_pos_list=[x[1] for x in obj_virt_data],
                                             obstacle_radius_list=[x[2] for x in obj_virt_data],
                                             object_id='virtual',
                                             color='TUM_grey_dark')

            if len(const_path_seg) > 0:
                plot_handler.highlight_pos(pos_coords=const_path_seg[-1, :],
                                           color_str='c',
                                           zorder=5,
                                           radius=2,
                                           id_in='Start Node')

            # plot pose of cut (near to actual pose)
            plot_handler_in.highlight_pos(pos_coords=clip_pos,
                                          id_in="Ego Position",
                                          color_str="r")

            # plot selected action id
            action_ids = str(list(vel_list.keys()))
            traj_ids = str(traj_id)
            plot_handler.update_text_field(text_str="Available actions: " + action_ids + "\n"
                                                    "Trajectory IDs:    " + traj_ids + "\n"
                                                    "Chosen action:     " + action_id,
                                           color_str='r')

            # plot object distances
            text_str = ""
            for i, vehicle in enumerate(obj_veh):
                eucl_dist = np.linalg.norm(np.array(clip_pos) - np.array(vehicle.get_pos()))
                text_str += "Obj. " + str(i) + ": " + "%.2fm\n" % eucl_dist
            plot_handler.update_text_field(text_str=text_str,
                                           text_field_id=2)

            # plot patches for overtaking zones
            patch_xy_pos_list = []
            for obj in obj_zone:
                bound_l, bound_r = obj.get_bound_coords()
                patch = np.vstack((bound_l, np.flipud(bound_r)))

                patch_xy_pos_list.append(patch)

            plot_handler.highlight_patch(patch_xy_pos_list=patch_xy_pos_list)

            self.__working = False
            plot_handler_in.show_plot(non_blocking=True)

        self.__working = False

    def get_closest_timestamp(self,
                              plot_handler_in,
                              graph_base_in,
                              time_stamp_in: float):
        # check if currently still processing (avoid lags)
        if not self.__working:
            self.__working = True

            dists = abs(np.array(self.__time_stamps_msgs) - time_stamp_in)

            # if cursor is in range of valid measurements
            if np.min(dists) < self.__time_threshold_msgs:
                # get index of closest time stamp
                _, idx = min((val, idx) for (idx, val) in enumerate(dists))

                # trigger plot update
                plot_handler_in.highlight_timeline(time_stamp=self.__time_stamps_msgs[idx],
                                                   type_in=self.__time_msgs_types[idx],
                                                   message=self.__time_msgs_content[idx])

            dists = abs(np.array(self.__time_stamps_data) - time_stamp_in)

            # if cursor is in range of valid measurements
            if np.min(dists) < self.__time_threshold_data:
                # get index of closest time stamp
                _, idx = min((val, idx) for (idx, val) in enumerate(dists))

                # trigger plot update
                self.plot_timestamp_n(plot_handler_in,
                                      graph_base_in,
                                      idx)
            else:
                self.__working = False


# ----------------------------------------------------------------------------------------------------------------------
# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    toppath = os.path.dirname(os.path.realpath(__file__ + "/../../../"))
    sys.path.append(toppath)

    # ----- get log-file name (via arguments or most recent one) -----
    if len(sys.argv) == 2:
        # if one argument provided, assume provided file name
        file_path = sys.argv[1]  # First argument
    else:
        # use most recent file if no arguments provided
        list_of_files = glob.glob(
            os.path.expanduser(toppath + '/logs/graph_ltpl/'
                               + datetime.datetime.now().strftime("%Y_%m_%d") + '/*_data.csv'))
        if list_of_files:
            file_path = max(list_of_files, key=os.path.getctime)
        else:
            raise ValueError("Could not find any logs in the specified folder! Please provide a file path argument.")

        # specific file
        # file_path = os.path.expanduser(toppath + '/logs/2019_03_22/11_00_47_data.csv')

    # extract common file parent
    file_path = file_path[:file_path.rfind("_")]

    # get file paths for all subfiles
    file_path_data = file_path + "_data.csv"
    file_path_msg = file_path + "_msg.csv"
    file_path_follow = file_path + "_follow.csv"

    # ----- FOLLOW MODE DEBUG PLOT -------------------------------------------------------------------------------------
    if os.path.isfile(file_path_follow):
        time_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['time']
        obj_dist_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['obj_dist']
        control_dist_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['control_dist']
        v_control_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['v_control']
        v_target_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['v_target']
        v_ego_f = np.genfromtxt(file_path_follow, delimiter=';', skip_header=0, names=True)['v_ego']

        if isinstance(time_f, (list, np.ndarray)) and len(time_f) > 0:
            # prepare data (when jumps occur, add a "None")
            time_f = list(time_f)
            control_dist_f = list(control_dist_f)
            obj_dist_f = list(obj_dist_f)
            v_control_f = list(v_control_f)
            v_target_f = list(v_target_f)
            v_ego_f = list(v_ego_f)

            last_time = time_f[0]
            for idx in range(len(time_f)):
                time_diff = abs(last_time - time_f[idx])
                last_time = time_f[idx]

                if time_diff > 1.0:
                    time_f[idx] = None
                    control_dist_f[idx] = None
                    obj_dist_f[idx] = None
                    v_control_f[idx] = None
                    v_target_f[idx] = None
                    v_ego_f[idx] = None

            plt.figure("Follow Mode Debug")
            ax_1 = plt.subplot(2, 1, 1)
            ax_1.set_title("Distances")
            ax_1.set_xlabel('$t$ in s')
            ax_1.set_ylabel('dist in m')
            ax_1.grid()

            ax_1.plot(time_f, control_dist_f)
            ax_1.plot(time_f, obj_dist_f)
            ax_1.legend(['$dist_{control}$', '$dist_{obj}$'])

            ax_2 = plt.subplot(2, 1, 2, sharex=ax_1)
            ax_2.set_title("Velocities")
            ax_2.set_xlabel('$t$ in s')
            ax_2.set_ylabel('$v$ in m/s')
            ax_2.grid()

            ax_2.plot(time_f, v_control_f)
            ax_2.plot(time_f, v_target_f)
            ax_2.plot(time_f, v_ego_f)
            ax_2.legend(['$v_{control}$', '$v_{target}$', '$v_{ego}$'])

    # -- get path to relevant graph-base object (for now: assuming the one in the repo) --------------------------------
    # get time stamps from file
    time_stamps = list(np.genfromtxt(file_path_data, delimiter=';', skip_header=1, names=True)['time'])

    # get message data (if available)
    if os.path.isfile(file_path_msg):
        time_stamps_msgs = list(np.genfromtxt(file_path_msg, delimiter=';', skip_header=0, names=True)['time'])
        msgs_type = list(np.genfromtxt(
            file_path_msg, delimiter=';', skip_header=0, dtype=None, encoding=None, names=True)['type'].astype(str))
        msgs_content = list(np.genfromtxt(
            file_path_msg, delimiter=';', skip_header=0, dtype=None, encoding=None, names=True)['message'].astype(str))

    else:
        time_stamps_msgs = []
        msgs_type = []
        msgs_content = []

    # extract a certain line number (based on time_stamp)
    with open(file_path_data) as file:
        # get to top of file
        file.seek(0)

        # get header (":-1" in order to remove tailing newline character)
        graph_file = file.readline()[1:-1]
        header = file.readline()[:-1]
        content = file.readlines()

    kappa_course = []
    vel_course = []
    action_id = None
    for line in content:
        data = dict(zip(header.split(";"), line.split(";")))
        if action_id is None or action_id not in json.loads(data['kappa_list']).keys():
            action_id = 'straight'
            if 'straight' not in json.loads(data['kappa_list']).keys():
                action_id = 'follow'
        try:
            kappa_course.append(json.loads(data['kappa_list'])[action_id][0][0])
            vel_course.append(json.loads(data['vel_list'])[action_id][0][0])
        except ValueError:
            kappa_course.append(0)
            vel_course.append(0)

        action_id = json.loads(data['action_id_prev'])

    kappa_course = np.column_stack((time_stamps, np.array(kappa_course) * 250 + 25))
    vel_course = np.column_stack((time_stamps, vel_course))

    # init debug handler
    dh = DebugHandler(time_stamps_data=time_stamps,
                      time_stamps_msgs=time_stamps_msgs,
                      time_msgs_types=msgs_type,
                      time_msgs_content=msgs_content)

    # -- load graph base object ----------------------------------------------------------------------------------------
    graph_str_path = os.path.expanduser(os.path.dirname(file_path) + '/../Graph_Objects/') + graph_file + '.pckl'

    if osfuncs.isfile(graph_str_path):
        f = open(graph_str_path, 'rb')
        graph_base = renamed_load(f)
        f.close()
    else:
        raise ValueError("Could not load graph-base object! Please double-check the provided path ("
                         + graph_str_path + ").")

    # check version compatibility of graph object and visualization tool
    msg = None
    try:
        if graph_base.VERSION != GRAPH_VERSION_SUPPORT:
            if graph_base.VERSION < GRAPH_VERSION_SUPPORT:
                msg = ("The loaded log was recorded with an older version of the SW-stack (V%.2f vs V%.2f)!\n"
                       "If you experience issues, consider reverting to an older SW-base for log review!"
                       % (graph_base.VERSION, GRAPH_VERSION_SUPPORT))
            else:
                msg = ("The loaded log was recorded with a newer version of the SW-stack (V%.2f vs V%.2f)!\n"
                       "If you experience issues, consider updating to a newer SW-base for log review!"
                       % (graph_base.VERSION, GRAPH_VERSION_SUPPORT))
    except AttributeError:
        mgs = ("The loaded log was recorded with an older version of the SW-stack!\n"
               "If you experience issues, consider reverting to an older SW-base for log review!")

    if msg is not None:
        warnings.warn("\n###########\n" + msg + "\n###########\n")

    # -- initialize plot -----------------------------------------------------------------------------------------------

    # plot time stamps
    plot_handler = graph_ltpl.visualization.src.PlotHandler.\
        PlotHandler(plot_title="Graph Log Visualization", include_timeline=True)

    # plot major components
    plot_handler.plot_graph_base(graph_base=graph_base,
                                 cost_dep_color=COST_DEPENDENT_COLOR,
                                 plot_edges=VISUALIZE_ALL_EDGES)

    # get combined time stamps
    time_stamps_comb = time_stamps + time_stamps_msgs
    types_comb = (["DATA"] * len(time_stamps)) + msgs_type

    # plot time line
    plot_handler.plot_timeline_stamps(time_stamps=time_stamps_comb,
                                      types=types_comb,
                                      lambda_fct=lambda time_stamp: dh.get_closest_timestamp(plot_handler, graph_base,
                                                                                             time_stamp))

    plot_handler.plot_timeline_course(line_coords_list=[vel_course, kappa_course])

    # initialize track plot with first entry in log
    dh.plot_timestamp_n(plot_handler, graph_base, 1)

    plot_handler.show_plot()
