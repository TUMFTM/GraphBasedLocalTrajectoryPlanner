import numpy as np
import time
import logging

# custom modules
import graph_ltpl

# Known object types
KNOWN_OBJ_TYPES = ["physical"]

# Allowed time in seconds to pass without object update
TIME_WARNING = 0.5


class ObjectListInterface(object):
    """
    Interfaces the object-list module, extracts and prepares the latest object information via the ZMQ interface.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        03.12.2018

    """

    def __init__(self) -> None:

        # init logger handle
        self.__log = logging.getLogger("local_trajectory_logger")

        # init object containers
        self.__object_vehicles = []
        self.__object_zones = []

        # bounds
        self.__refline = None
        self.__normvec_normalized = None
        self.__w_left = None
        self.__w_right = None
        self.__bound1 = None
        self.__bound2 = None

        # time variable
        self.__last_timestamp = 0.0

    def __del__(self) -> None:
        pass

    def set_track_data(self,
                       refline: np.ndarray,
                       normvec_normalized: np.ndarray,
                       w_left: np.ndarray,
                       w_right: np.ndarray) -> None:
        """
        Set track data. The track data is used to only add vehicles within the track-bounds to the object-list and for
        virtual collect obstacles.

        :param refline:             reference line (numpy array with columns x and y)
        :param normvec_normalized:  normal vectors based on every point in the reference line (numpy array w. c. x & y)
        :param w_left:              track width left measured from the reference line coords along the normal vectors
        :param w_right:             track width right measured from the reference line coords along the normal vectors

        """

        self.__refline = refline
        self.__normvec_normalized = normvec_normalized
        self.__w_left = w_left
        self.__w_right = w_right

        # calculate bound coordinates
        self.__bound1 = refline + normvec_normalized * np.expand_dims(w_right, 1)
        self.__bound2 = refline - normvec_normalized * np.expand_dims(w_left, 1)

    def process_object_list(self,
                            object_list: list) -> list:
        """
        Process object-list. This function provides the following features:

        * Track new and lost objects (e.g. avoid iterative collision checks for stationary objects - only calc. once)
        * Process objects depending on their object-type (e.g. virtual objects increase the cost in the graph, edges
          intersecting real vehicles are removed from the graph, ...)

        :param object_list:             list of dicts, each dict describing an object with keys ['id', 'type', 'X', 'Y',
                                        'theta', 'v', 'length', 'width' ...]
        :returns:
            * **object_vehicles** -     list of vehicle object instances of type 'VehObject'

        """

        # process all objects in list
        if object_list is not None:
            self.__last_timestamp = time.time()

            # re-init object containers
            new_vehicle_objects = []

            for object_el in object_list:
                if object_el['type'] in KNOWN_OBJ_TYPES:

                    # check for car objects
                    if object_el['type'] == "physical":

                        # check if vehicle within bounds / on track (ignore objects outside)
                        on_track = True
                        if self.__bound1 is not None and self.__bound2 is not None:
                            on_track = graph_ltpl.online_graph.src.check_inside_bounds.\
                                check_inside_bounds(bound1=self.__bound1,
                                                    bound2=self.__bound2,
                                                    pos=[object_el['X'], object_el['Y']])

                        if on_track:
                            # add prediction
                            # NOTE: since the spatial and temporal domain are decoupled, the selected nodes here are
                            #       blocked for the complete planning horizon (all temporal states)

                            if 'prediction' in object_el.keys():
                                # extract prediction
                                pred = object_el['prediction']

                            else:
                                # if no prediction provided, (at least) calculate (simple) prediction for delay comp.

                                dt = 0.2  # 200ms prediction horizon
                                pred = np.zeros((1, 2))
                                pred[0, 0] = object_el['X'] - np.sin(object_el['theta']) * object_el['v'] * dt
                                pred[0, 1] = object_el['Y'] + np.cos(object_el['theta']) * object_el['v'] * dt

                            # for now: circle contains whole vehicle
                            veh_obj = VehObject(id_in=object_el['id'],
                                                pos_in=[object_el['X'], object_el['Y']],
                                                psi_in=object_el['theta'],
                                                radius_in=(object_el['length'] / 2.0),
                                                vel_in=object_el['v'],
                                                prediction_in=pred)
                            new_vehicle_objects.append(veh_obj)

                else:
                    self.__log.warning("Found non-supported object of type '%s' in object list!" % object_el['type'])

            self.__object_vehicles = new_vehicle_objects

        else:
            if time.time() - self.__last_timestamp > TIME_WARNING:
                if self.__last_timestamp == 0.0:
                    time_str = "so far"
                else:
                    time_str = "in the last %.2fs" % (time.time() - self.__last_timestamp)

                # warn user about connection issues and keep old objects
                self.__log.warning("Did not receive an object list " + time_str + "! Check coms!")

        return self.__object_vehicles

    def update_zone(self,
                    zone_id: str,
                    zone_data: list or np.ndarray,
                    zone_type: str = 'normals') -> list:
        """
        Updates the internal representation of zones. Zones are used to block several nodes in the graph. That way it is
        possible to temporally block large regions in the driving space (e.g. a pit-lane or currently impassable track
        regions).

        :param zone_id:             unique id specifier for the given zone
        :param zone_data:           zone data specifying the blocked nodes (either via node ids or normal vectors):

                                    * **nodes**: list with the following members:
                                        * blocked layer numbers - pairwise with blocked node numbers
                                        * blocked node numbers - pairwise with blocked layer numbers
                                        * numpy array holding coordinates of left bound of region (columns x and y)
                                        * numpy array holding coordinates of right bound of region (columns x and y)
                                    * **normals**: numpy array with the following columns:
                                        * x coordinate of reference line points
                                        * y coordinate of reference line points
                                        * x coordinate of normal vector (matching the reference line points)
                                        * y coordinate of normal vector (matching the reference line points)
                                        * distance to left bound of zone (along each of the normal vectors) [in m]
                                        * distance to right bound of zone (along each of the normal vectors) [in m]

        :param zone_type:           string selecting the two input methods ("normals" or "nodes")
        :returns:
            * **object_zones** -    list of zone object instances of type 'ZoneObject'

        """

        # re-init object container
        new_zone_objects = []

        # get existing object zones
        last_ids = [x.id for x in self.__object_zones]

        if zone_id is not None:
            if zone_id in last_ids:
                last_ids_idx = last_ids.index(zone_id)
                new_zone_objects.append(self.__object_zones[last_ids_idx])

                # remove zone from last ids (mark as processed)
                last_ids[last_ids_idx] = None
            else:
                if zone_type == 'normals':
                    # reshape array
                    zone_info = np.reshape(zone_data, (-1, 6))

                    zone_obj = ZoneObject(id_in=zone_id,
                                          ref_pos_in=zone_info[:, 0:2],
                                          norm_vec_in=zone_info[:, 2:4],
                                          bound_l_in=zone_info[:, 4],
                                          bound_r_in=zone_info[:, 5])
                elif zone_type == 'nodes':
                    zone_obj = ZoneObject(id_in=zone_id,
                                          blocked_layer_ids_in=zone_data[0],
                                          blocked_node_ids_in=zone_data[1],
                                          bound_l_coord_in=zone_data[2],
                                          bound_r_coord_in=zone_data[3])

                else:
                    raise ValueError("Type specifier " + zone_type + " is not supported!")

                # add zone object
                new_zone_objects.append(zone_obj)

                self.__log.info("Received new zone object with ID " + zone_id + "!")

        # check if a zone was removed -> keep in list but add removal flag
        for zone_id in last_ids:
            if zone_id is not None:
                removed_zone_idx = last_ids.index(zone_id)

                # if zone still contains nodes, keep it
                if self.__object_zones[removed_zone_idx].get_blocked_nodes()[0]:
                    self.__object_zones[removed_zone_idx].set_disabled()
                    self.__object_zones[removed_zone_idx].id = self.__object_zones[removed_zone_idx].id + "rmv"
                    new_zone_objects.append(self.__object_zones[removed_zone_idx])

        self.__object_zones = new_zone_objects

        return self.__object_zones


class VehObject(object):
    """
    Vehicle object class storing relevant data for a vehicle.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        25.02.2019

    """

    def __init__(self,
                 id_in: int,
                 pos_in: list,
                 psi_in: float,
                 radius_in: float,
                 vel_in: float = None,
                 prediction_in: np.ndarray = None):

        # public properties
        self.id = id_in

        # private properties
        self.__pos = pos_in
        self.__psi = psi_in
        self.__radius = radius_in
        self.__vel = vel_in
        self.__prediction = prediction_in

    def get_pos(self):
        return self.__pos

    def get_psi(self):
        return self.__psi

    def get_prediction(self):
        return self.__prediction

    def get_radius(self):
        return self.__radius

    def get_vel(self):
        return self.__vel

    def update_properties(self,
                          pos_in: np.ndarray,
                          psi_in: float,
                          radius_in: np.ndarray,
                          vel_in: float = None,
                          prediction_in: np.ndarray = None):
        self.__pos = pos_in
        self.__psi = psi_in
        self.__radius = radius_in
        self.__vel = vel_in
        self.__prediction = prediction_in


class ZoneObject(object):
    """
    Zone object class storing relevant data for zone objects. Furthermore it offers functions to retrieve
    intersecting nodes from an graph. This process is only triggered once for a zone object and then stored in the
    object in order to reduce computation load.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        25.02.2019

    """

    def __init__(self,
                 id_in: str,
                 ref_pos_in: np.ndarray = None,
                 norm_vec_in: np.ndarray = None,
                 bound_l_in: np.ndarray = None,
                 bound_r_in: np.ndarray = None,
                 blocked_layer_ids_in: list = None,
                 blocked_node_ids_in: list = None,
                 bound_l_coord_in: list = None,
                 bound_r_coord_in: list = None):

        # public properties
        self.id = id_in
        self.processed = False
        self.disabled = False
        self.fixed = False

        # private properties
        self.__ref_pos = ref_pos_in
        self.__norm_vec = norm_vec_in
        self.__bound_l = bound_l_in
        self.__bound_r = bound_r_in

        self.__blocked_layer_ids = blocked_layer_ids_in
        self.__blocked_node_ids = blocked_node_ids_in

        if ref_pos_in is not None and norm_vec_in is not None and bound_l_in is not None and bound_r_in is not None:
            self.__bound_l_coord = ref_pos_in + norm_vec_in * np.expand_dims(bound_l_in, 1)
            self.__bound_r_coord = ref_pos_in + norm_vec_in * np.expand_dims(bound_r_in, 1)
        elif blocked_layer_ids_in is not None and blocked_node_ids_in is not None and bound_l_coord_in is not None and \
                bound_r_coord_in is not None:
            self.__bound_l_coord = bound_l_coord_in
            self.__bound_r_coord = bound_r_coord_in
        else:
            raise ValueError("No matching set of initialization variables was provided!")

    def get_blocked_nodes(self,
                          graph_base=None):

        if self.__blocked_layer_ids is None and graph_base is not None:
            # get blocked layers
            self.__blocked_layer_ids, self.__blocked_node_ids, succ_match = \
                graph_ltpl.online_graph.src.get_zone_nodes.get_zone_nodes(graph_base=graph_base,
                                                                          ref_pos=self.__ref_pos,
                                                                          norm_vec=self.__norm_vec,
                                                                          bound_l=self.__bound_l,
                                                                          bound_r=self.__bound_r)

            if not succ_match:
                logging.getLogger("local_trajectory_logger").critical(
                    "Provided zone object '" + str(self.id) + "' does not share ANY common normal"
                    " vectors with internal representation! Zone is ignored!")
                raise ValueError("Provided zone object is not supported (details above)!")

        return self.__blocked_layer_ids, self.__blocked_node_ids

    def update_blocked_nodes(self,
                             layer_ids: list,
                             node_ids: list):
        self.__blocked_layer_ids = layer_ids
        self.__blocked_node_ids = node_ids

    def get_bound_coords(self):
        return self.__bound_l_coord, self.__bound_r_coord

    def update_bound_coords(self,
                            bound_l_coord: np.ndarray,
                            bound_r_coord: np.ndarray):
        self.__bound_l_coord = bound_l_coord
        self.__bound_r_coord = bound_r_coord

    def set_processed(self):
        self.processed = True

    def set_disabled(self):
        self.disabled = True

    def set_fixed(self):
        self.fixed = True
