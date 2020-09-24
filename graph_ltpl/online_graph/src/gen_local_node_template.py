import numpy as np
import bisect
import logging

# custom modules
import graph_ltpl


UNBLOCK_N_LAYERS_WHEN_IN_ZONE = 4
BLOCK_N_LAYERS_WHEN_REMOVING_ZONE = 0


def gen_local_node_template(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                            start_node: tuple,
                            obj_veh: list,
                            obj_zone: list,
                            last_solution_nodes: list = None,
                            w_last_edges: list = ()) -> tuple:
    """
    Generates a template for the offline Graph, which specifies the nodes available in the local online
    graph segment. The output of this template is well suited as input for the adjacent graph solver.

    :param graph_base:              reference to GraphBase object instance
    :param start_node:              id of the start node (list consisting of layer and node id)
    :param obj_veh:                 list of objects holding info like "pos", "ids", "vel", "radius"
    :param obj_zone:                list of objects holding information about the blocked zones
    :param last_solution_nodes:     nodes resulted from the previous search
    :param w_last_edges:            online cost reduction per previously planned path segment
    :returns:
        * **end_layer** -           index of the end layer (determined by start layer and a provided min. path length)
        * **closest_obj_index** -   index of object in object list which is the closest upfront the ego-vehicle
        * **closest_obj_node** -    closest node in the graph framework to the closest object

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        30.10.2018

    """

    # -- OVERTAKING ZONES ----------------------------------------------------------------------------------------------
    if not obj_zone or not all([zone.processed for zone in obj_zone]) or any([zone.disabled for zone in obj_zone]):
        layer_ids_total = []
        node_ids_total = []
        for zone in obj_zone:
            layer_ids, node_ids = zone.get_blocked_nodes(graph_base=graph_base)

            # if new zone
            if not zone.processed or zone.disabled:
                # check if new zone contains overlaps in layer range with ego pose
                if not zone.processed:
                    n = UNBLOCK_N_LAYERS_WHEN_IN_ZONE
                else:  # zone.disabled
                    n = BLOCK_N_LAYERS_WHEN_REMOVING_ZONE

                # determine relevant layer range
                if (start_node[0] + n) <= graph_base.num_layers:
                    u_l = np.logical_and(np.array(layer_ids) >= start_node[0], np.array(layer_ids)
                                         < (start_node[0] + n))
                else:
                    u_l = np.logical_or(
                        np.logical_and(np.array(layer_ids) >= start_node[0], np.array(layer_ids)
                                       < graph_base.num_layers),
                        np.logical_and(np.array(layer_ids) >= 0,
                                       np.array(layer_ids) < ((start_node[0] + n) % (graph_base.num_layers - 1) - 1)))

                # process nodes according to type
                if not zone.processed:
                    if any(u_l) and not zone.fixed:
                        # throw warning
                        logging.getLogger("local_trajectory_logger").\
                            critical("Vehicle within provided zone, unblock active!")

                        # remove upcoming n layers from zone
                        layer_ids = list(np.array(layer_ids)[~np.array(u_l)])
                        node_ids = list(np.array(node_ids)[~np.array(u_l)])
                    zone.set_processed()
                if zone.disabled:
                    if any(u_l):
                        # remain some layers from removed zone (avoid abrupt changes)
                        layer_ids = list(np.array(layer_ids)[np.array(u_l)])
                        node_ids = list(np.array(node_ids)[np.array(u_l)])
                    else:
                        layer_ids = []
                        node_ids = []

                    zone.update_blocked_nodes(layer_ids=layer_ids,
                                              node_ids=node_ids)
                    zone.update_bound_coords(bound_l_coord=[0.0, 0.0],
                                             bound_r_coord=[0.0, 0.0])

            layer_ids_total.extend(layer_ids)
            node_ids_total.extend(node_ids)

        graph_base.remove_nodes_filter(layer_ids=layer_ids_total,
                                       node_ids=node_ids_total,
                                       applied_filter="overtaking_zones",
                                       base=None)

    # -- PLANNING RANGE ------------------------------------------------------------------------------------------------
    start_layer = start_node[0]

    if not hasattr(graph_base, 'plan_horizon_mode') or graph_base.plan_horizon_mode == 'distance':
        # ensure backward compatibility
        if not hasattr(graph_base, 'plan_horizon_mode'):
            min_plan_horizon = 200.0
        else:
            min_plan_horizon = graph_base.min_plan_horizon

        # Determine end layer (based on distance "min_path_length")
        des_dist = graph_base.s_raceline[start_layer] + min_plan_horizon

        # if targeted end is further away than reference line
        if des_dist > graph_base.s_raceline[-1]:
            if graph_base.closed:
                # if track is closed, continue measure from start line
                des_dist -= graph_base.s_raceline[-1]

            else:
                # if unclosed, limit to end of reference line
                des_dist = graph_base.s_raceline[-1]

        end_layer = bisect.bisect_left(graph_base.s_raceline, des_dist)

    elif graph_base.plan_horizon_mode == 'layers':
        if graph_base.closed:
            # if track is closed, continue count from start line
            end_layer = (start_layer + int(graph_base.min_plan_horizon)) % graph_base.num_layers

        else:
            # if unclosed, select layer (maximum is last layer)
            end_layer = max((start_layer + int(graph_base.min_plan_horizon)), graph_base.num_layers - 1)

    else:
        raise ValueError('Unsupported planning horizon mode "' + graph_base.plan_horizon_mode + '"!')

    # Get planning horizon in number of layers
    planning_dist = end_layer - start_layer
    if planning_dist < 0:
        # when overlapping start line
        planning_dist = graph_base.num_layers - start_layer + end_layer

    graph_base.set_node_filter_layers(start_layer=start_layer,
                                      end_layer=end_layer,
                                      applied_filter="planning_range",
                                      base="overtaking_zones")

    # Activate node filter
    # NOTE: In order to enable a fast collision checking it is crucial to cancel out irrelevant nodes before evaluating
    #       edges for collision. -> Apply node filter first
    graph_base.activate_filter(applied_filter="planning_range")

    # --  REDUCE COST ALONG PREVIOUS PATH (if provided) ----------------------------------------------------------------
    if last_solution_nodes is not None:
        for i in range(min(len(last_solution_nodes) - 1, len(w_last_edges))):
            graph_base.factor_edge_cost(start_layer=last_solution_nodes[i][0],
                                        start_node=last_solution_nodes[i][1],
                                        end_layer=last_solution_nodes[i + 1][0],
                                        end_node=last_solution_nodes[i + 1][1],
                                        cost_factor=w_last_edges[i],
                                        active_filter="planning_range")

    # -- OBJECT INTERSECTION -------------------------------------------------------------------------------------------
    edges = []
    closest_obj_layer_dist = None
    closest_obj_index = None
    closest_obj_node = None
    for idx, vehicle in enumerate(obj_veh):
        # vehicle
        intersecting_edges, obj_layer = graph_ltpl.online_graph.src.get_intersec_edges. \
            get_intersec_edges(graph_base=graph_base,
                               object_pos=vehicle.get_pos(),
                               object_radius=vehicle.get_radius(),
                               planning_start_layer=start_layer,
                               planning_end_layer=end_layer,
                               remove_filters=False)
        edges.extend(intersecting_edges)

        # prediction
        for pos_pred in vehicle.get_prediction():
            intersecting_edges, obj_layer = graph_ltpl.online_graph.src.get_intersec_edges. \
                get_intersec_edges(graph_base=graph_base,
                                   object_pos=pos_pred,
                                   object_radius=vehicle.get_radius(),
                                   planning_start_layer=start_layer,
                                   planning_end_layer=end_layer,
                                   remove_filters=False)
            edges.extend(intersecting_edges)

        if obj_layer is not None:
            # calculate object layer distance from start layer
            layer_dist = obj_layer - start_layer
            if layer_dist < 0:
                # when overlapping start line or object behind vehicle
                layer_dist = graph_base.num_layers - start_layer + obj_layer

            # check for closest layer upfront the ego vehicle (start layer)
            if layer_dist <= planning_dist and \
                    ((closest_obj_layer_dist is None) or (layer_dist < closest_obj_layer_dist)):
                closest_obj_layer_dist = layer_dist
                closest_obj_index = idx
                closest_obj_node = [obj_layer, None]  # Node to be determined for the closest object only

    # if an object was in the planning range -> find the closest node to this object (for action type generation)
    if closest_obj_layer_dist is not None:
        # calculate distances to all nodes in determined layer
        pos = graph_base.get_layer_info(layer=closest_obj_node[0])[0]

        distances2 = np.power(np.array(pos)[:, 0] - obj_veh[closest_obj_index].get_pos()[0], 2) + np.power(
            np.array(pos)[:, 1] - obj_veh[closest_obj_index].get_pos()[1], 2)

        closest_obj_node[1] = np.argmin(distances2)

    # Init and activate edge filter with disabled edges (objects) (Note: This list may contain duplicates!)
    graph_base.init_edge_filter(disabled_edges=edges,
                                applied_filter="default",
                                base="planning_range")

    graph_base.activate_filter()

    return end_layer, closest_obj_index, closest_obj_node


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
