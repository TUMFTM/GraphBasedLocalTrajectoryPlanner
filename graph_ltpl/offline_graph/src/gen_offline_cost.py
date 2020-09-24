import configparser
import time
import numpy as np

# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph


def gen_offline_cost(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                     cost_config_path: str):
    """
    Generate offline cost for given edges (in a GraphBase object instance).

    :param graph_base:           reference to the GraphBase object instance holding all graph relevant information
    :param cost_config_path:     path pointing to the configuration file, parameterizing the cost generation

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        10.10.2018

    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE DATA -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Read cost configuration from provided datafile
    cost_config = configparser.ConfigParser()
    if not cost_config.read(cost_config_path):
        raise ValueError('Specified graph config file does not exist or is empty!')
    if 'COST' not in cost_config:
        raise ValueError('Specified graph config file does not hold the expected data!')

    # ------------------------------------------------------------------------------------------------------------------
    # GENERATE OFFLINE COST --------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # sample a path for each edge in the graph base
    tic = time.time()
    edges = graph_base.get_edges()
    i = 0
    for edge in edges:
        tph.progressbar.progressbar(i, len(edges) - 1, prefix="Generating cost  ")

        # retrieve stored data for given edge
        spline_coeff, spline_param, offline_cost, spline_len = graph_base.get_edge(edge[0], edge[1], edge[2], edge[3])

        # Calculate offline cost / init
        offline_cost = 0.0

        # average curvature (div. by #elements and multiplied by length (to be independent of coord.-resolution)
        offline_cost += cost_config.getfloat('COST', 'w_curv_avg') * np.power(
            sum(abs(spline_param[:, 3])) / float(len(spline_param[:, 3])), 2) * spline_len

        # peak curvature
        offline_cost += cost_config.getfloat('COST', 'w_curv_peak') * np.power(
            abs(max(spline_param[:, 3]) - min(spline_param[:, 3])), 2) * spline_len

        # Path length
        offline_cost += cost_config.getfloat('COST', 'w_length') * spline_len

        # Raceline cost (normed by distance)
        raceline_dist = abs(graph_base.raceline_index[edge[2]] - edge[3]) * graph_base.lat_resolution
        offline_cost += min(cost_config.getfloat('COST', 'w_raceline') * spline_len * raceline_dist,
                            cost_config.getfloat('COST', 'w_raceline_sat') * spline_len)

        # store determined value along with graph edge
        graph_base.update_edge(start_layer=edge[0],
                               start_node=edge[1],
                               end_layer=edge[2],
                               end_node=edge[3],
                               offline_cost=offline_cost)

        i += 1

    toc = time.time()
    print("Cost generation took " + '%.3f' % (toc - tic) + "s")


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
