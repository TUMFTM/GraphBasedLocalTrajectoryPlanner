import numpy as np

# custom modules
import graph_ltpl

# Threshold triggering detailed normal vector matching (squared distance in meters²)
DIST2_THRESHOLD = 0.1


def get_zone_nodes(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                   ref_pos: np.ndarray,
                   norm_vec: np.ndarray,
                   bound_l: np.ndarray,
                   bound_r: np.ndarray,
                   obstacle_width: float = 0.0) -> tuple:
    """
    Determine enclosed nodes for given normal vectors and corresponding lateral bounds for each of them.

    :param graph_base:       reference to GraphBase object instance holding relevant parameters
    :param ref_pos:          reference pose of normal vectors holding the zone constraints
    :param norm_vec:         normal vectors on the corresponding reference poses
    :param bound_l:          left bound in meters on the normal vector
    :param bound_r:          right bound in meters on the normal vector
    :param obstacle_width:   (optional) set obstacle width in order to enlarge zones according to possible obstacle size
    :returns:
        * **layer_ids** -    list of lists of layer indexes of affected nodes (pairwise with node list)
        * **node_ids** -     list of lists of node indexes of affected nodes (pairwise with layer list)
        * **succ_match** -   boolean stating whether the provided ref_pos and norm_vecs match to the skeleton of the
          GraphBase object instance

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        08.02.2019

    """

    # Init containers
    common_norm_vec_obj_idx = []
    layer_ids = []
    node_ids = []

    # Add vehicle width (including safety margin) and half lateral resolution (cope with discrete) to overtaking zones
    if bound_l[0] > bound_r[0]:
        bound_s_l = bound_l + max(graph_base.veh_width / 2, obstacle_width / 2) + graph_base.lat_resolution / 2
        bound_s_r = bound_r - max(graph_base.veh_width / 2, obstacle_width / 2) - graph_base.lat_resolution / 2
    else:
        bound_s_l = bound_l - max(graph_base.veh_width / 2, obstacle_width / 2) - graph_base.lat_resolution / 2
        bound_s_r = bound_r + max(graph_base.veh_width / 2, obstacle_width / 2) + graph_base.lat_resolution / 2

    # Try to match normal vectors
    for i in range(np.size(ref_pos, axis=0)):
        layer_idx, dist2 = \
            graph_ltpl.helper_funcs.src.closest_path_index.closest_path_index(path=graph_base.refline,
                                                                              pos=ref_pos[i, :])

        # if distance smaller than threshold, check if normal vectors are similar
        if dist2[layer_idx] < DIST2_THRESHOLD:
            if np.allclose(norm_vec[i, :], graph_base.normvec_normalized[layer_idx], atol=0.01):
                common_norm_vec_obj_idx.append(i)

                # Determine affected nodes per layer

                steps_l = (bound_s_l[i] - graph_base.alpha[layer_idx[0]]) / graph_base.lat_resolution
                steps_r = (bound_s_r[i] - graph_base.alpha[layer_idx[0]]) / graph_base.lat_resolution
                rl_idx = graph_base.raceline_index[layer_idx[0]]
                l_idx = min(max(rl_idx + int(np.ceil(steps_l)), 0), graph_base.nodes_in_layer[layer_idx[0]])
                r_idx = min(max(rl_idx + int(np.ceil(steps_r)), 0), graph_base.nodes_in_layer[layer_idx[0]])

                local_nodes = list(range(min([l_idx, r_idx]), max([l_idx, r_idx])))

                # Extend layer and node index list
                layer_ids.extend(layer_idx * len(local_nodes))
                node_ids.extend(local_nodes)

    # If no common normal vector, raise warning
    succ_match = bool(common_norm_vec_obj_idx)

    return layer_ids, node_ids, succ_match
