import numpy as np
import graph_ltpl


def get_intersec_edges(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                       object_pos: np.ndarray,
                       object_radius: float,
                       planning_start_layer: int = 0,
                       planning_end_layer: int = 999999,
                       remove_filters: bool = True,
                       consider_discretiz: bool = True) -> tuple:
    """
    Determine all intersecting edges in the provided 'graph_base' object instance with a specified circular obstacle.

    :param graph_base:           reference to the GraphBase object instance holding all graph relevant information
    :param object_pos:           position of the obstacle of interest (circular abstraction)
    :param object_radius:        radius of the obstacle of interest
    :param planning_start_layer: start layer of the planning horizon (if obstacle not in range, do not consider)
    :param planning_end_layer:   end layer of the planning horizon (if obstacle not in range, do not consider)
    :param remove_filters:       if 'True', all active filters are disabled, when searching for intersecting edges
                                 if 'False', the current filter is used (this can increase speed drastically)
    :param consider_discretiz:   if 'True', the discretization of the grid is taken into account (obstacles inflated to
                                 stay on safe side)
    :returns:
        * **edges** -            list of edges intersecting with the provided obstacle
        * **obj_layer** -        layer id in graph, which is closest to object

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        30.10.2018

    """

    # layer overlap
    lo = 1

    # determine affected layers (based on closest point of refline)
    refline = graph_base.refline
    distances2 = np.power(refline[:, 0] - object_pos[0], 2) + np.power(refline[:, 1] - object_pos[1], 2)
    val, obj_layer = min((val, idx) for (idx, val) in enumerate(distances2))

    # relevant layers (at the moment, just consider previous and next layer)
    start_layer = obj_layer - lo
    end_layer = obj_layer + lo

    # if obstacle in range of planning horizon
    if (planning_start_layer - lo <= obj_layer <= planning_end_layer + lo
            or (planning_start_layer > planning_end_layer
                and (planning_start_layer - lo <= obj_layer or obj_layer <= planning_end_layer + lo))):
        # check all splines between the determined layers for intersection (distance smaller than provided radius)
        edges = graph_base.get_intersec_edges_in_range(start_layer=start_layer,
                                                       end_layer=end_layer,
                                                       obstacle_pos=object_pos,
                                                       obstacle_radius=object_radius,
                                                       remove_filters=remove_filters,
                                                       consider_discretiz=consider_discretiz)
    else:
        edges = []
        obj_layer = None

    return edges, obj_layer
