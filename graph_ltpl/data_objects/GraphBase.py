import numpy as np
import itertools

# before loading igraph, configure to block all warnings regarding reachability of vertices
import warnings
warnings.filterwarnings('ignore', r"Couldn't reach some vertices")

import igraph


class GraphBase(object):
    """
    This class serves as a custom (planning graph specific) wrapper for the igraph library. The following main features
    are implemented:

    * Interfacing the igraph class (providing fast c++ operations) - all the contact with this library is managed within
      this wrapper class.
    * Providing methods to store and access nodes as well as edges (together with their trajectory specific information
      / properties).
    * The class should host all information required for the online execution (a class instance is pickled for
      pre-planning and logging purposes), therefore relevant environment parameters are stored as class members
      (ref. line, ...). Once the offline graph (spatial lattice) has been laid out, the class instance can be stored.
      When executing the planner the next time, it is checked if any parameter files changed, if not, the last graph
      instance can be loaded and executed directly (skips time-demanding spatial lattice generation).
    * Online methods, e.g. graph search and filtering (obstacles), can be triggered with class member functions.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        28.09.2018

    """

    def __init__(self,
                 lat_offset: float,
                 num_layers: int,
                 refline: np.ndarray,
                 normvec_normalized: np.ndarray,
                 track_width_right: np.ndarray,
                 track_width_left: np.ndarray,
                 alpha: np.ndarray,
                 vel_raceline: np.ndarray,
                 s_raceline: np.ndarray,
                 lat_resolution: float,
                 sampled_resolution: float,
                 vel_decrease_lat: float,
                 veh_width: float,
                 veh_length: float,
                 veh_turn: float,
                 md5_params: str,
                 graph_id: str,
                 glob_rl: np.ndarray,
                 virt_goal_node: bool = True,
                 virt_goal_node_cost: float = 200.0,
                 min_plan_horizon: int or float = 200.0,
                 plan_horizon_mode: str = 'distance',
                 closed: bool = True) -> None:
        """
        :param lat_offset:          allowed lateral offset parallel to race line per (long.) travelled meter [in m]
        :param num_layers:          total number of layers in the graph (along the whole track)
        :param refline:             x and y coordinates of reference line (numpy array with columns x and y)
        :param normvec_normalized:  norm. normal vectors at each point in the refline (numpy array with columns x and y)
        :param track_width_right:   width of track meas. from each point in the refline in dir. of normal vectors [in m]
        :param track_width_left:    width of track meas. from each point in the refline in dir. of normal vectors [in m]
        :param alpha:               displacement of race line at each point in refline in dir. of normal vectors [in m]
        :param vel_raceline:        velocity at each point in the race line [in mps]
        :param s_raceline:          s-coordinate along the race line (starting at '0.0' for the first point) [in m]
        :param lat_resolution:      lateral discretization of graph-nodes across the track [in m]
        :param sampled_resolution:  approximated step-size for all generated splines / trajectories
        :param vel_decrease_lat:    goal velocity reduction per meter of lateral displacement to the race line [in mps]
        :param veh_width:           ego vehicle width [in m]
        :param veh_length:          ego vehicle length [in m]
        :param veh_turn:            vehicle turn radius [in m]
        :param md5_params:          md5 sum of all relevant parameter files (check whether class instance is up to date)
        :param graph_id:            unique string ID for this graph
        :param glob_rl:             global race line with fine resolution - independent of graph discretization (numpy
                                    array with columns: s, x, y, curvature, velocity)
        :param virt_goal_node:      boolean specifying whether a virtual goal node for the graph search should be used
        :param virt_goal_node_cost: cost for the vertices from virtual goal node to the nodes in the goal layer per
                                    meter lateral displacement to race line node
        :param min_plan_horizon:    minimum planning horizon - either in meters or a fixed number of layers (see mode)
        :param plan_horizon_mode:   string specifying distance ('distance') or fixed number of layers ('layers') based
                                    planning horizon calculation
        :param closed:              boolean specifying whether the track is closed or not

        """

        # Version number (Increases, when new options are added / changed)
        self.VERSION = 0.2

        # general (public) parameters
        self.lat_offset = lat_offset
        self.num_layers = num_layers
        self.refline = refline
        self.track_width_right = track_width_right
        self.track_width_left = track_width_left
        self.lat_resolution = lat_resolution
        self.normvec_normalized = normvec_normalized
        self.alpha = alpha
        self.s_raceline = s_raceline
        self.sampled_resolution = sampled_resolution
        self.vel_raceline = vel_raceline
        self.vel_decrease_lat = vel_decrease_lat
        self.veh_width = veh_width
        self.veh_length = veh_length
        self.veh_turn = veh_turn
        self.raceline_index = None
        self.md5_params = md5_params
        self.graph_id = graph_id
        self.glob_rl = glob_rl
        self.virt_goal_node = virt_goal_node
        self.virt_goal_node_cost = virt_goal_node_cost
        self.min_plan_horizon = min_plan_horizon
        self.plan_horizon_mode = plan_horizon_mode
        self.closed = closed

        # calculate raceline
        self.raceline = refline + normvec_normalized * alpha[:, np.newaxis]

        # initialize graph-tool object
        self.__g = igraph.Graph()
        self.__g.to_directed()

        # copy of original graph (for filtering)
        self.__g_orig = None

        # set of filter graphs
        self.__g_filter = dict()

        # dictionary holding all nodes and corresponding information
        self.__virtual_layer_node = dict()

        # number of nodes per layer
        self.nodes_in_layer = dict()

    # ------------------------------------------------------------------------------------------------------------------
    # BASIC NODE FUNCTIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def add_node(self,
                 layer: int,
                 node_number: int,
                 pos: np.ndarray,
                 psi: float,
                 raceline_index: int = None) -> None:
        """
        Stores a node which will later hold all relevant information.

        :param layer:           layer number the node should be added to
        :param node_number:     number of the node in the layer
        :param pos:             coordinate of the node in the local map
        :param psi:             orientation of the node in range [-pi, pi] with 0.0 being north
        :param raceline_index:  number of the node which is passed by the race line in this layer ('layer')

        """

        # if plain (non-filtered) graph is not active, switch to it
        if self.__g_orig is not None:
            self.__g = self.__g_orig

        # create vertex in graph-tool object and store related information / properties
        self.__g.add_vertex(name=str((layer, node_number)),
                            position=pos,
                            psi=psi,
                            raceline=(raceline_index == node_number),
                            node_id=node_number,
                            layer_id=layer)

        # update the max node index for the given layer (if the value is larger than a possibly existing stored one)
        self.nodes_in_layer[layer] = max(self.nodes_in_layer.get(layer, 0), node_number + 1)

        # if virtual layer node activated
        if self.virt_goal_node:
            # check if virtual layer node exists
            if layer not in self.__virtual_layer_node.keys():
                # add virtual goal node for given layer
                vv = "v_l" + str(layer)
                self.__g.add_vertex(name=vv,
                                    layer_id=layer)

                # store reference in dict
                self.__virtual_layer_node[layer] = vv
            else:
                vv = self.__virtual_layer_node[layer]

            # add edge between virtual and generated node
            offline_cost = abs(raceline_index - node_number) * self.lat_resolution * self.virt_goal_node_cost

            self.__g.add_edge(source=str((layer, node_number)),
                              target=vv,
                              virtual=1,
                              start_layer=layer,
                              offline_cost=offline_cost)

    def add_layer(self,
                  layer: int,
                  pos_multi: np.ndarray,
                  psi: np.ndarray,
                  raceline_index: int) -> None:
        """
        Add several nodes belonging to one layer (several positions provided via "pos_multi". no children assumed.

        :param layer:           layer number the nodes should be added to
        :param pos_multi:       numpy array holding multiple positions
        :param psi:             numpy array holding the heading for each node and position
        :param raceline_index:  number of the node which is passed by the race line in this layer ('layer')

        """

        for i in range(len(pos_multi)):
            self.add_node(layer=layer,
                          node_number=i,
                          pos=pos_multi[i],
                          psi=psi[i],
                          raceline_index=raceline_index)

        # update the max node index for the given layer (if the value is larger than a possibly existing stored one)
        self.nodes_in_layer[layer] = max(self.nodes_in_layer.get(layer, 0), len(pos_multi))

    def get_node_info(self,
                      layer: int,
                      node_number: int,
                      return_child: bool = False,
                      return_parent: bool = False,
                      active_filter: str = "current") -> tuple:
        """
        Return information stored for a specific node.

        :param layer:           layer number of the node to be returned
        :param node_number:     node number in the layer to be returned
        :param return_child:    boolean specifying whether the IDs of the child nodes should be returned ('None'
                                returned else)
        :param return_parent:   boolean specifying whether the IDs of the parent nodes should be returned ('None'
                                returned else)
        :param active_filter:   string specifying the filter of the graph the node(s) should be retrieved from
        :returns:
            * **pos** -         position of the node
            * **psi** -         orientation of the node in range [-pi, pi] with 0.0 being north
            * **raceline** -    index of the raceline node in this layer
            * **children** -    list of children nodes, each specified by a tuple holding layer and node number [(l, n)]
            * **parents** -     list of parent nodes, each specified by a tuple holding layer and node number [(l, n)]

        """

        if active_filter is None:
            g = self.__g_orig
        elif active_filter == "current":
            g = self.__g
        else:
            g = self.__g_filter[active_filter]

        # Check for invalid ID
        try:
            node = g.vs.find(str((layer, node_number)))
        except ValueError as e:
            print("KeyError - Could not find requested node ID! " + str(e))
            return None, None, None, None, None

        pos = node['position']
        psi = node['psi']
        raceline = node['raceline']

        # retrieve nodes children
        if return_child:
            idx_children = g.successors(node.index)
            children = [(g.vs[v]["layer_id"], g.vs[v]["node_id"]) for v in idx_children
                        if g.vs[v]["node_id"] is not None]
        else:
            children = None

        # retrieve nodes parents
        if return_parent:
            idx_parents = g.predecessors(node.index)
            parents = [(g.vs[v]["layer_id"], g.vs[v]["node_id"]) for v in idx_parents
                       if g.vs[v]["node_id"] is not None]
        else:
            parents = None

        return pos, psi, raceline, children, parents

    def get_layer_info(self,
                       layer: int) -> tuple:
        """
        Return the information for all nodes in the specified layer.

        :param layer:               layer number of the nodes to be returned
        :returns:
            * **pos_list** -        list of positions of the nodes
            * **psi_list** -        list of orientations of the nodes
            * **raceline_list** -   list of ints indicating the node number of the race line in this layer
            * **children_list** -   list of lists holding children nodes, each specified by a tuple holding layer and
              node number [(l, n)]

        """

        pos_list = []
        psi_list = []
        raceline_list = []
        children_list = []

        for i in range(self.nodes_in_layer[layer]):
            pos, psi, raceline, children, _ = self.get_node_info(layer=layer,
                                                                 node_number=i,
                                                                 return_child=True,
                                                                 active_filter=None)

            if pos is not None:
                pos_list.append(pos)
                psi_list.append(psi)
                raceline_list.append(raceline)
                children_list.append(children)

        return pos_list, psi_list, raceline_list, children_list

    def get_closest_nodes(self,
                          pos: np.ndarray,
                          limit: int or float = 1,
                          fixed_amount: bool = True) -> tuple:
        """
        Searches for the n closest points in the graph for a given coordinate (x, y), with two modes as follows:

        * fixed_amount=True:  return the n closest points (amount specified by "limit")
        * fixed_amount=False: return _all_ points within a radius specified by "limit" in meters

        :param pos:             position of interest
        :param limit:           limit returned nodes by radius [in m] or amount of nodes (int) - see 'fixed_amount'
        :param fixed_amount:    boolean specifying whether to return fixed number of nodes or nodes within radius (see
                                description)
        :returns:
            * **nodes** -       list of nodes, each specified by a tuple holding layer and node number [(l, n), (), ...]
            * **distances** -   list of distances to each of the returned nodes [in m]

        """

        # extract coordinates and calculate squared distances
        node_indexes, node_positions = \
            zip(*[(node_idx, node_pos) for node_idx, node_pos in enumerate(self.__g.vs['position'])
                  if node_pos is not None])
        node_positions = np.vstack(node_positions)
        distances2 = np.power(node_positions[:, 0] - pos[0], 2) + np.power(node_positions[:, 1] - pos[1], 2)

        if fixed_amount:
            # find k indexes holding minimum squared distances
            idx = np.argpartition(distances2, limit)[:limit]
        else:
            # get all elements within radius "limit"
            ref_radius = limit * limit
            idx = np.argwhere(distances2 < ref_radius).reshape(1, -1)[0]

        # get actual distances of those k nodes
        distances = np.sqrt(distances2[idx])

        # return matching node ids (layer and node number)
        nodes = [(self.__g.vs[node_indexes[v]]["layer_id"], self.__g.vs[node_indexes[v]]["node_id"]) for v in idx
                 if self.__g.vs[node_indexes[v]]["node_id"] is not None]
        return nodes, distances

    def get_nodes(self) -> list:
        """
        Returns a list of all stored nodes in the graph.

        :returns:
            * **nodes** -       list of nodes, each specified by a tuple holding layer and node number [(l, n), (), ...]

        """

        return [(v["layer_id"], v["node_id"]) for v in self.__g.vs if v["node_id"] is not None]

    # ------------------------------------------------------------------------------------------------------------------
    # BASIC EDGE FUNCTIONS ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def add_edge(self,
                 start_layer: int,
                 start_node: int,
                 end_layer: int,
                 end_node: int,
                 **kwargs) -> None:  # Optionally pass argument "spline_coeff", "spline_coord", "offline_cost"
        """
        Initializes or updates (if already existing) an edge between the specified nodes. The method bay be called with
        the following optional arguments:

        * **spline_coeff** -    coefficients of the cubic spline associated with this edge
        * **spline_coord** -    sampled coordinates along the spline associated with this edge
        * **offline_cost** -    offline cost associated with the edge

        ..note:: If no optional parameters are provided, the connection between the two specified nodes is set up in the
            graph framework. The parameters can be added or updated later by calling the function again with the same
            node identifiers and providing one or multiple of the above mentioned arguments.

        :param start_layer:     layer number of the node the edge originates from
        :param start_node:      number of the node in the layer the edge originates from
        :param end_layer:       layer number of the node the edge destinates in
        :param end_node:        number of the node in the layer the edge destinates in

        """

        # if plain (non-filtered) graph is not active, switch to it
        if self.__g_orig is not None:
            self.__g = self.__g_orig

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        # create edge, if not existent yet
        edge_id = self.__g.get_eid(sn, en, error=False)
        if edge_id == -1:
            self.__g.add_edge(start_layer=start_layer,
                              virtual=0,
                              source=sn,
                              target=en,
                              spline_coeff=None,
                              spline_length=None,
                              spline_param=None,
                              offline_cost=None)
            edge_id = self.__g.get_eid(sn, en, error=False)

        # check for optional arguments
        if 'spline_coeff' in kwargs:
            self.__g.es[edge_id]['spline_coeff'] = kwargs.get('spline_coeff')

        if 'spline_x_y_psi_kappa' in kwargs:
            # calculate length
            el_lengths = np.sqrt(np.sum(np.power(
                np.diff(kwargs.get('spline_x_y_psi_kappa')[:, 0:2], axis=0), 2), axis=1))

            # store total spline length
            self.__g.es[edge_id]['spline_length'] = np.sum(el_lengths)

            #  append zero to el length array, in order to reach equal array length
            el_lengths = np.append(el_lengths, 0)

            # generate proper formatted numpy array containing spline data
            self.__g.es[edge_id]['spline_param'] = \
                np.column_stack((kwargs.get('spline_x_y_psi_kappa'), el_lengths))

        if 'offline_cost' in kwargs:
            self.__g.es[edge_id]['offline_cost'] = kwargs.get('offline_cost')

    # "update_edge" equals the function "add_edge"
    update_edge = add_edge

    def get_edge(self,
                 start_layer: int,
                 start_node: int,
                 end_layer: int,
                 end_node: int) -> tuple:
        """
        Retrieve the information stored for a specified edge.

        :param start_layer:         layer number of the node the edge originates from
        :param start_node:          number of the node in the layer the edge originates from
        :param end_layer:           layer number of the node the edge destinates in
        :param end_node:            number of the node in the layer the edge destinates in
        :returns:
            * **spline_coeff** -    coefficients of the cubic spline associated with this edge
            * **spline_param** -    sampled parameters along the spline associated with this edge (numpy array with
              columns x, y, heading, curvature)
            * **offline_cost** -    offline cost associated with the edge
            * **spline_length** -   length of the spline

        """

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        edge_id = self.__g.get_eid(sn, en)  # if we want to catch errors, add "error=False" and check for returned "-1"
        edge = self.__g.es(edge_id)

        spline_coeff = edge['spline_coeff'][0]
        spline_param = edge['spline_param'][0]
        offline_cost = edge['offline_cost'][0]
        spline_length = edge['spline_length'][0]

        return spline_coeff, spline_param, offline_cost, spline_length

    def factor_edge_cost(self,
                         start_layer: int,
                         start_node: int,
                         end_layer: int,
                         end_node: int,
                         cost_factor: float,
                         active_filter: str = "current") -> None:
        """
        Update the cost of a specified edge (multiply current cost with 'cost_factor').

        :param start_layer:         layer number of the node the edge originates from
        :param start_node:          number of the node in the layer the edge originates from
        :param end_layer:           layer number of the node the edge destinates in
        :param end_node:            number of the node in the layer the edge destinates in
        :param cost_factor:         factor to be multiplied with the current cost of the edge
        :param active_filter:       string specifying the filter of the graph the edge cost should be updated in

        """

        if active_filter is None:
            g = self.__g_orig
        elif active_filter == "current":
            g = self.__g
        else:
            g = self.__g_filter[active_filter]

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        try:
            edge_id = g.get_eid(sn, en)
            g.es[edge_id]['offline_cost'] *= cost_factor
        except ValueError:
            # catch errors, when nodes are not present anymore (e.g. filtering)
            pass

    def increase_edge_cost(self,
                           cost_value: float,
                           edge_list: list,
                           active_filter: str = "current") -> None:
        """
        Increase the edge cost of specified edges by a specified offset.

        ..note:: The specified cost_value is only added to edges holding a cost lower than 'cost_value'.

        :param cost_value:       float value specifying the cost to be added to the edge
        :param edge_list:        list of node pairs [[str((start_layer, start_node)), str((end_layer, end_node))], ...]
        :param active_filter:    string specifying the filter of the graph the edge cost should be updated in

        """

        if active_filter is None:
            g = self.__g_orig
        elif active_filter == "current":
            g = self.__g
        else:
            g = self.__g_filter[active_filter]

        for (start_node, end_node) in edge_list:
            try:
                edge_id = g.get_eid(start_node, end_node)

                if g.es[edge_id]['offline_cost'] <= cost_value:
                    g.es[edge_id]['offline_cost'] += cost_value
            except ValueError:
                # catch errors, when nodes are not present anymore (e.g. filtering)
                pass

    def remove_edge(self,
                    start_layer: int,
                    start_node: int,
                    end_layer: int,
                    end_node: int) -> None:
        """
        Remove the specified edge from the graph (handle with caution - only for the offline part).

        :param start_layer:         layer number of the node the edge originates from
        :param start_node:          number of the node in the layer the edge originates from
        :param end_layer:           layer number of the node the edge destinates in
        :param end_node:            number of the node in the layer the edge destinates in

        """

        sn = str((start_layer, start_node))
        en = str((end_layer, end_node))

        edge_id = self.__g.get_eid(sn, en)
        self.__g.delete_edges(edge_id)

    def get_intersec_edges_in_range(self,
                                    start_layer: int,
                                    end_layer: int,
                                    obstacle_pos: np.ndarray,
                                    obstacle_radius: float,
                                    remove_filters: bool = True,
                                    consider_discretiz: bool = True) -> list:
        """
        Determine all edges between start_layer and end_layer, that intersect with the obstacle specified by a pos and a
        radius.

        ..warning:: Edge ids are just valid in the same graph --> when filtering make sure to apply filter before
            intersecting edge detection and directly process edge ids afterwards.

        ..note:: Since the edges in the graph are sampled in a discretized manner (e.g. a coordinate every 1m), the
            collision range of the obstacles is inflated by the sampling step-width, if 'consider_discretiz' is set
            to 'True' (default). That way it is ensured to detect all intersections by overestimation.

        :param start_layer:         layer number specifying the start of the range to be checked for edge intersections
        :param end_layer:           layer number specifying the end of the range to be checked for edge intersections
        :param obstacle_pos:        position of the obstacle to be checked for intersection with any edge in range
        :param obstacle_radius:     radius of the obstacle to be considered [in m]
        :param remove_filters:      if set to 'True' collisions are checked for all splines (not just the filtered ones)
        :param consider_discretiz:  if set to 'True' the obstacles are inflated in order to cope with the discretization
        :returns:
            * **intersec_edges** -  list of intersecting edges, each specified by a tuple of node IDs [(sn, en), (),...]

        """

        # check if start and end layer are within available layers
        if start_layer < 0:
            start_layer += self.num_layers
        if end_layer > self.num_layers:
            end_layer -= self.num_layers

        if remove_filters:
            g = self.__g_orig
        else:
            g = self.__g

        # # select subset of edges to check for collision
        # if start_layer < end_layer:
        #     selected_edges = g.es.select(virtual_eq=0, start_layer_ge=start_layer, start_layer_le=end_layer-1)
        # else:
        #     # assuming start line overlaps (next lap)
        #     layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer+1))
        #     selected_edges = g.es.select(virtual_eq=0, start_layer_in=layer_set)
        if start_layer < end_layer:
            selected_nodes = g.vs.select(layer_id_ge=start_layer, layer_id_le=end_layer)
        else:
            # assuming start line overlaps (next lap)
            layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer + 1))
            selected_nodes = g.vs.select(layer_id_in=layer_set)

        sub_g = g.induced_subgraph(selected_nodes)
        selected_edges = sub_g.es

        # obstacle reference is sum of radius of both objects plus an offset to cope with discretization
        # NOTE: formula given by isosceles triangle -> a = sqrt(h^2 + c^2/4)
        obstacle_ref = np.power(obstacle_radius + self.veh_width / 2, 2)

        if consider_discretiz:
            obstacle_ref += np.power(self.sampled_resolution, 2) / 4

        intersec_edges = []
        # for all edges (with applied filter)
        for edge in selected_edges:
            # check edge for collision
            param = edge["spline_param"]

            # check if edge is not a virtual one
            if param is not None:
                param = param
                x = param[:, 0] - obstacle_pos[0]
                y = param[:, 1] - obstacle_pos[1]
                distances2 = x * x + y * y
                if any(distances2 <= obstacle_ref):
                    intersec_edges.append([sub_g.vs[edge.source]["name"], sub_g.vs[edge.target]["name"]])

        return intersec_edges

    def get_edges(self) -> list:
        """
        Returns a list of all stored edges in the graph.

        :returns:
            * **edges** -   list of edges, each specified by a tuple holding number of start layer, number of start node
              number of end layer, number of end node, e.g. [(ls, ns, le, ne), (...), ...]

        """

        return [(self.__g.vs[edge.source]["layer_id"], self.__g.vs[edge.source]["node_id"],
                 self.__g.vs[edge.target]["layer_id"], self.__g.vs[edge.target]["node_id"]) for edge in self.__g.es
                if self.__g.vs[edge.source]["node_id"] is not None and self.__g.vs[edge.target]["node_id"] is not None]

    # ------------------------------------------------------------------------------------------------------------------
    # FILTERING --------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def init_filtering(self) -> None:
        """
        Stores a version of the graph as an original copy.

        """

        self.__g_orig = self.__g.copy()

    def set_node_filter_layers(self,
                               start_layer: int,
                               end_layer: int,
                               applied_filter="default",
                               base: str = None) -> None:
        """
        Initializes a filter with the nodes belonging to layers in range "start_layer" to "end_layer" remaining in the
        graph. If no "applied_filter" is provided, the internal active node filter is used.

        :param start_layer:      index of the first layer to be included in the filtered graph
        :param end_layer:        index of the last layer to be included in the filtered graph
        :param applied_filter:   name of the filtered graph
        :param base:             provide a string of a filtered set here, to build the new filtered graph upon an
                                 existing one (if not set, build from scratch / unfiltered graph)

        """

        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            if base in self.__g_filter.keys():
                g = self.__g_filter[base]
            else:
                g = self.__g_orig

        if start_layer < end_layer:
            selected_nodes = g.vs.select(layer_id_ge=start_layer, layer_id_le=end_layer)
        else:
            # assuming start line overlaps (next lap)
            layer_set = list(range(start_layer, self.num_layers)) + list(range(0, end_layer + 1))
            selected_nodes = g.vs.select(layer_id_in=layer_set)

        self.__g_filter[applied_filter] = g.induced_subgraph(selected_nodes)

    def remove_nodes_filter(self,
                            layer_ids: list,
                            node_ids: list,
                            applied_filter="default",
                            base: str = None) -> None:
        """
        Remove the nodes specified by the lists "layer_ids" and "node_ids" in the filtered graph "applied_filter".

        :param layer_ids:       list of layer numbers (pairwise with 'node_ids')
        :param node_ids:        list of node numbers (pairwise with 'layer_ids')
        :param applied_filter:  name of the filtered graph
        :param base:            provide a string of a filtered set here, to build the new filtered graph upon an
                                existing one (if not set, build from scratch / unfiltered graph)

        """

        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            if base in self.__g_filter.keys():
                g = self.__g_filter[base]
            else:
                g = self.__g_orig

        name_ref = [str((x, y)) for x, y in zip(layer_ids, node_ids)]

        selected_nodes = g.vs.select(name_notin=name_ref)
        self.__g_filter[applied_filter] = g.induced_subgraph(selected_nodes)

    def init_edge_filter(self,
                         disabled_edges,
                         applied_filter="default",
                         base: str = None) -> None:
        """
        Initializes a filter with the edges in the list "disabled_edges" being removed

        :param disabled_edges:   list of edges (start and end node) to be not present in the resulting filtered graph
        :param applied_filter:   name of the filtered graph
        :param base:             provide a string of a filtered set here, to build the new filtered graph upon an
                                 existing one (if not set, build from scratch / unfiltered graph)

        """

        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        if base is None:
            g = self.__g_orig
        elif base == "current":
            g = self.__g
        else:
            g = self.__g_filter[base]

        # Determine edge ids to be deleted
        edge_ids = g.get_eids(pairs=disabled_edges)

        self.__g_filter[applied_filter] = g.copy()
        self.__g_filter[applied_filter].delete_edges(edge_ids)

    def deactivate_filter(self) -> None:
        """
        Deactivate any active graph filter (restore original one).

        """

        self.__g = self.__g_orig

    def activate_filter(self,
                        applied_filter="default") -> None:
        """
        Apply a previously defined filter.

        :param applied_filter:  name of the filtered graph

        """
        if self.__g_orig is None:
            self.__g_orig = self.__g.copy()

        self.__g = self.__g_filter[applied_filter]

    # ------------------------------------------------------------------------------------------------------------------
    # GRAPH SEARCH -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def search_graph(self,
                     start_node: int,
                     end_node: int,
                     max_solutions: int = 1) -> tuple:
        """
        Search a cost minimal path in the graph (returns "None"s if graph search problem is infeasible).

        :param start_node:         node id (internal graph node type identifier)
        :param end_node:           node id (internal graph node type identifier)
        :param max_solutions:      maximum number of path solutions to be returned (NOTE: 1 is the fastest!)
        :returns:
            * **positions_list** - list of positions (nodes) along the cost minimal path
            * **node_ids_list** -  list of node ids (each a tuple of layer and node number) along the cost minimal path

        """

        v_list_iterator = self.__g.get_shortest_paths(start_node,
                                                      to=end_node,
                                                      weights="offline_cost",
                                                      output="vpath")
        # NOTE: self.__g.get_eid(path=v_list_iterator) -> returns edge ids

        # Check if the graph search problem is feasible
        if not v_list_iterator or not any(v_list_iterator[0]):
            positions_list = None
            node_ids_list = None
        else:
            positions_list = []
            node_ids_list = []
            for v_list in itertools.islice(v_list_iterator, max_solutions):
                # we want to extract the nodes IDs and their positions
                node_ids = []
                pos = []
                # loop through all edges in sublist
                for node_id in v_list:
                    # if node is not a virtual one
                    node = self.__g.vs[node_id]
                    if node["node_id"] is not None:
                        node_ids.append([node["layer_id"], node["node_id"]])
                        pos.append(node["position"])

                # append this path to list
                positions_list.append(pos)
                node_ids_list.append(node_ids)

            # if no valid nodes for iterator found
            if not any(positions_list):
                positions_list = None
                node_ids_list = None

        return positions_list, node_ids_list

    def search_graph_layer(self,
                           start_layer: int,
                           start_node: int,
                           end_layer: int,
                           max_solutions: int = 1) -> tuple:
        """
        Interfaces the graph search function and handles the problem of finding the goal node in the specified end
        layer. Either sequentially testing the nodes in the goal layer or using the virtual goal layer (if active).

        ..note:: If virtual nodes are deactivated (in the config), the search tries first to end in the race line node
            on the end_layer and then iteratively tests all nodes, iteratively increasing the distance to the race line
            node. If virtual nodes are activated, the search results in the cost optimal node in the goal layer based on
            the connections with the virtual goal node.

        :param start_layer:        layer number of the start node the search should start from
        :param start_node:         number of the start node the search should start from
        :param end_layer:          layer number the graph search should end in (try to end on race line if not blocked)
        :param max_solutions:      maximum number of path solutions to be returned (NOTE: 1 is the fastest!)
        :returns:
            * **positions_list** - list of positions (nodes) along the cost minimal path
            * **node_ids_list** -  list of node ids (each a tuple of layer and node number) along the cost minimal path

        """

        positions_list = None
        node_ids_list = None

        # Try to extract start node, if blocked return as unsolvable
        try:
            start_node_id = self.__g.vs.find(str((start_layer, start_node)))
        except ValueError:
            return None, None

        if self.virt_goal_node:
            # Determine virtual goal layer node (Note: the virtual node is automatically removed after graph search)
            virt_node = self.__virtual_layer_node[end_layer]

            # trigger graph search to virtual goal node
            positions_list, node_ids_list = self.search_graph(start_node=start_node_id,
                                                              end_node=virt_node,
                                                              max_solutions=max_solutions)

        else:
            # Search in defined goal layer, if search does not result in a solution, check pts nxt to raceline
            # Search through nodes in end_layer until a solution is found or no node is left (computationally expensive)
            end_node = None
            while True:
                if end_node is None:
                    end_node = self.raceline_index[end_layer]
                elif end_node <= self.raceline_index[end_layer]:
                    # Race line point seems to be blocked -> look through points with smaller index
                    if end_node <= 0:
                        end_node = self.raceline_index[end_layer] + 1
                    else:
                        end_node -= 1
                else:
                    # Race line point & points with smaller index are blocked -> look through points with larger index
                    end_node += 1

                    # Check if node exists
                    if self.get_node_info(layer=end_layer, node_number=end_node)[0] is None:
                        break

                end_node_id = self.__g.vs.find(str((end_layer, end_node)))

                # Trigger graph search (if goal node exists)
                if self.get_node_info(layer=end_layer, node_number=end_node)[0] is not None:
                    positions_list, node_ids_list = self.search_graph(start_node=start_node_id,
                                                                      end_node=end_node_id,
                                                                      max_solutions=max_solutions)

                # If solution is found, return (otherwise search for goal state in next layer)
                if node_ids_list is not None:
                    break

        return positions_list, node_ids_list


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
