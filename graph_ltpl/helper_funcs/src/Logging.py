import json
import numpy as np


class Logging:
    """
    Logging class that handles the setup and data-flow in order to write a log for the graph-planner in an iterative
    manner.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Alexander Heilmeier

    :Created on:
        23.01.2019

    """

    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self,
                 graph_id: str,
                 log_path: str) -> None:
        """
        :param graph_id:    string holding the unique graph object id in order to keep matching pairs (pickled graph
                            object and log)
        :param log_path:    string holding path the logging file should be created in

        """

        # write header to logging file
        self.__fp_log = log_path
        with open(self.__fp_log, "w+") as fh:
            header = ("#" + str(graph_id) + "\n"
                      + "time;s_coord;start_node;obj_veh;obj_zone;nodes_list;s_list;pos_list;vel_list;"
                      + "a_list;psi_list;kappa_list;traj_id;clip_pos;action_id_prev;traj_id_prev;const_path_seg")
            fh.write(header)

        self.__obj_zone_ids = None
        self.__obj_zone_timestamp = None

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def log_onlinegraph(self,
                        time: float,
                        s_coord: float,
                        start_node: tuple,
                        obj_veh: list,
                        obj_zone: list,
                        nodes_list: dict,
                        s_list: dict,
                        pos_list: dict,
                        vel_list: dict,
                        a_list: dict,
                        psi_list: dict,
                        kappa_list: dict,
                        traj_id: dict,
                        clip_pos: list,
                        action_id_prev: str,
                        traj_id_prev: int,
                        const_path_seg: np.ndarray) -> None:
        """
        Write one line to the log file.

        :param time:            current time stamp (float time)
        :param s_coord:         current global s-coordinate on the track
        :param start_node:      current start node of the graph planning problem (i.e. node a bit upfront the vehicle)
        :param obj_veh:         list of object vehicles
        :param obj_zone:        list of zone objects
        :param nodes_list:      dict holding the nodes along the action set's solutions (each key an action set)
        :param s_list:          dict holding the s-coordinates along the action set's solutions
        :param pos_list:        dict holding positions along the action set's solutions
        :param vel_list:        dict holding velocity values along the action set's solutions
        :param a_list:          dict holding acceleration values along the action set's solutions
        :param psi_list:        dict holding heading values along the action set's solutions
        :param kappa_list:      dict holding curvature values along the action set's solutions
        :param traj_id:         dict holding trajectory IDs for each action set
        :param clip_pos:        position where the trajectory starts (current position of ego vehicle)
        :param action_id_prev:  previously chosen / executed action set
        :param traj_id_prev:    previousle chosen / executed trajectory number within the action set (if multiple)
        :param const_path_seg:  coordinates of the constant path segment (between ego-position and start_node)

        """

        # check if obj_zone changed
        if self.__obj_zone_ids is None or not self.__obj_zone_ids == [x.id for x in obj_zone]:
            # extract relevant information from object zone
            obj_zone_log = []
            for obj in obj_zone:
                obj_zone_log.append([obj.get_blocked_nodes(graph_base=None), obj.get_bound_coords()])

            self.__obj_zone_ids = [x.id for x in obj_zone]
            self.__obj_zone_timestamp = str(time)
        else:
            obj_zone_log = ["no update since", self.__obj_zone_timestamp]

        obj_veh_log = []
        for obj in obj_veh:
            obj_veh_log.append([obj.id, obj.get_pos(), obj.get_psi(), obj.get_radius(), obj.get_vel(),
                                obj.get_prediction()])

        if const_path_seg is not None:
            const_path_seg = const_path_seg[:, 0:2]

        with open(self.__fp_log, "a") as fh:
            fh.write("\n"
                     + str(time) + ";"
                     + str(s_coord) + ";"
                     + json.dumps(start_node, default=default) + ";"
                     + json.dumps(obj_veh_log, default=default) + ";"
                     + json.dumps(obj_zone_log, default=default) + ";"
                     + json.dumps(nodes_list, default=default) + ";"
                     + json.dumps(s_list, default=default) + ";"
                     + json.dumps(pos_list, default=default) + ";"
                     + json.dumps(vel_list, default=default) + ";"
                     + json.dumps(a_list, default=default) + ";"
                     + json.dumps(psi_list, default=default) + ";"
                     + json.dumps(kappa_list, default=default) + ";"
                     + json.dumps(traj_id, default=default) + ";"
                     + json.dumps(clip_pos, default=default) + ";"
                     + json.dumps(action_id_prev, default=default) + ";"
                     + json.dumps(traj_id_prev, default=default) + ";"
                     + json.dumps(const_path_seg, default=default))


def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError('Not serializable (type: ' + str(type(obj)) + ')')


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
