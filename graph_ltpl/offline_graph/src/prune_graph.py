# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph


def prune_graph(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                closed: bool = True) -> None:
    """
    Prune graph - remove nodes and edges that are not reachable within the cyclic graph.

    :param graph_base:      reference to the GraphBase object instance holding all graph relevant information
    :param closed:          if false, an un-closed track is assumed, i.e. last layer nodes will not be pruned

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        28.09.2018

    """

    j = 0
    rmv_cnt_tot = 0
    nodes = graph_base.get_nodes()

    while True:
        rmv_cnt = 0
        for i, node in enumerate(nodes):
            tph.progressbar.progressbar(min(j * len(nodes) + i, len(nodes) * 10 - 2),
                                        len(nodes) * 10 - 1, prefix="Pruning graph    ")

            # if not closed, keep all nodes in start and end-layer
            if not closed and (node[0] == graph_base.num_layers - 1 or node[0] == 0):
                continue

            # get children and parents of node
            _, _, _, children, parents = graph_base.get_node_info(layer=node[0],
                                                                  node_number=node[1],
                                                                  return_child=True,
                                                                  return_parent=True)

            # remove edges (removing nodes may destroy indexing conventions)
            if not children or not parents:
                # if no children or no parents, remove all connecting edges
                if not children:
                    for parent in parents:
                        rmv_cnt += 1
                        graph_base.remove_edge(start_layer=parent[0],
                                               start_node=parent[1],
                                               end_layer=node[0],
                                               end_node=node[1])
                else:
                    for child in children:
                        rmv_cnt += 1
                        graph_base.remove_edge(start_layer=node[0],
                                               start_node=node[1],
                                               end_layer=child[0],
                                               end_node=child[1])

        if rmv_cnt == 0:
            break
        else:
            rmv_cnt_tot += rmv_cnt

        j += 1

    tph.progressbar.progressbar(100, 100, prefix="Pruning graph    ")

    if rmv_cnt_tot > 0:
        print("Removed %d edges, identified as dead ends!" % rmv_cnt_tot)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
