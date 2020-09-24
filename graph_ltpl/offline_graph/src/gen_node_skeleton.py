import numpy as np

# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph


def gen_node_skeleton(graph_base: graph_ltpl.data_objects.GraphBase.GraphBase,
                      length_raceline: list,
                      var_heading=True,
                      closed: bool = True) -> list:
    """
    Generate a node skeleton for a graph structure based on center line, track width and race line.

    :param graph_base:      reference to the GraphBase object instance holding all graph relevant information:

                            * track_width:        track width at given center line coordinates in meters
                            * normvec_normalized: x and y components of normized normal vector at given center line pos
                            * alpha:      alpha parameter (percentage of track width) for min global curvature
                            * veh_width:          width of the vehicle (i.e. distance of CG to wall)

    :param length_raceline: list holding element lengths of the raceline
    :param var_heading:     flag, defining whether the heading of each node is linearly sampled between the bounds
                            orientation and the race line heading (else: all points with same heading as raceline)
    :param closed:          if true, a closed circuit is assumed
    :returns:
        * **state_pos** -   stacked list of x and y coordinates of all nodes generated

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        26.09.2018

    """

    # closed index - include last length element for closed tracks
    if closed:
        closed_idx = None
    else:
        closed_idx = -1

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE / EXTRACT DATA -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # extract data stored in the graph_base object
    normvec_normalized = graph_base.normvec_normalized
    alpha = graph_base.alpha
    track_width_right = graph_base.track_width_right
    track_width_left = graph_base.track_width_left

    # calculate raceline points
    raceline_points = graph_base.refline + normvec_normalized * alpha[:, np.newaxis]

    # ------------------------------------------------------------------------------------------------------------------
    # EXTRACT ORIENTATION ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # heading of race line points in global coord (subtract pi/2, since zero is north)
    psi = tph.calc_head_curv_num.calc_head_curv_num(path=raceline_points,
                                                    el_lengths=np.array(length_raceline[:closed_idx]),
                                                    is_closed=closed)[0]

    # if variable heading, generate heading of outer bounds
    psi_bound_l = None
    psi_bound_r = None
    if var_heading:
        # get bounds
        # bound_l = graph_base.refline - normvec_normalized * [1, 1]
        # bound_r = graph_base.refline + normvec_normalized * [1, 1]
        bound_r = graph_base.refline + normvec_normalized * np.expand_dims(track_width_right, 1)
        bound_l = graph_base.refline - normvec_normalized * np.expand_dims(track_width_left, 1)

        # get distance between bound elements
        d = np.diff(np.vstack((bound_l, bound_l[0])), axis=0)
        len_bl = np.hypot(d[:, 0], d[:, 1])

        d = np.diff(np.vstack((bound_r, bound_r[0])), axis=0)
        len_br = np.hypot(d[:, 0], d[:, 1])

        psi_bound_l = tph.calc_head_curv_num.calc_head_curv_num(path=bound_l,
                                                                el_lengths=np.array(len_bl[:closed_idx]),
                                                                is_closed=closed)[0]

        psi_bound_r = tph.calc_head_curv_num.calc_head_curv_num(path=bound_r,
                                                                el_lengths=np.array(len_br[:closed_idx]),
                                                                is_closed=closed)[0]

    # ------------------------------------------------------------------------------------------------------------------
    # GENERATE SAMPLES ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    state_pos = []
    raceline_index_array = []

    # Check if safety margin is valid (paired with raceline)
    margin_left = min(track_width_left - graph_base.veh_width / 2 + alpha)
    margin_right = min(track_width_right - graph_base.veh_width / 2 - alpha)
    if (margin_left < 0.0) or (margin_right < 0.0):
        # calculate maximum vehicle width
        max_veh_width = graph_base.veh_width + min(margin_left, margin_right) * 2

        print("#############################\n")
        print("Specified vehicle width is too wide in order to follow the provided raceline! The maximum possible "
              "vehicle width for this raceline is %.3fm!" % max_veh_width)
        print("-> In order to specify the vehicle width of the local planner, change 'veh_width' in "
              "'/params/graph_config_offline.ini'")
        print("\n#############################")
        raise ValueError("Provided raceline holds points outside the safety margin! "
                         "Reduce the vehicle width or adapt the race line (## SEE DETAILS ABOVE ##).")

    # for each normvect
    for i in range(len(normvec_normalized)):
        # determine raceline index in array to be generated
        raceline_index = int(np.floor((track_width_left[i] - graph_base.veh_width / 2 + alpha[i])
                                      / graph_base.lat_resolution))
        raceline_index_array.append(raceline_index)

        # determine start point on normal relative to ideal raceline point
        s = alpha[i] - raceline_index * graph_base.lat_resolution

        # spread points on normal with defined lateral offset and calculated starting point
        temp_alphas = np.arange(s, track_width_right[i] - graph_base.veh_width / 2, graph_base.lat_resolution)

        # transfer alphas to cartesian space
        temp_pos = np.repeat(graph_base.refline[i][None, :], len(temp_alphas), axis=0) + np.repeat(
            normvec_normalized[i][None, :], len(temp_alphas), axis=0) * temp_alphas[:, np.newaxis]

        # calculate psi for each coordinate
        if var_heading:
            # since psi is defined in [-pi <-> pi], we need to check, which heading path is the shortest
            if abs(psi_bound_l[i] - psi[i]) < np.pi:
                psi1 = np.linspace(psi_bound_l[i], psi[i], num=raceline_index + 1)[:-1]
            else:
                temp_bl = psi_bound_l[i] + 2 * np.pi * (psi_bound_l[i] < 0)
                temp_psi = psi[i] + 2 * np.pi * (psi[i] < 0)
                psi1 = tph.normalize_psi.\
                    normalize_psi(np.linspace(temp_bl, temp_psi, num=raceline_index + 1)[:-1])
            if abs(psi_bound_r[i] - psi[i]) < np.pi:
                psi2 = np.linspace(psi[i], psi_bound_r[i], num=len(temp_alphas) - raceline_index)
            else:
                temp_br = psi_bound_r[i] + 2 * np.pi * (psi_bound_r[i] < 0)
                temp_psi = psi[i] + 2 * np.pi * (psi[i] < 0)
                psi2 = tph.normalize_psi. \
                    normalize_psi(np.linspace(temp_psi, temp_br, num=len(temp_alphas) - raceline_index))

            temp_psi = np.append(psi1, psi2)
        else:
            temp_psi = np.repeat(psi[i], len(temp_alphas), axis=0)

        # store info in local variable
        list.append(state_pos, [temp_pos, temp_psi])

        # store node information in data object
        graph_base.add_layer(layer=i,
                             pos_multi=temp_pos,
                             psi=temp_psi,
                             raceline_index=raceline_index)

    # store race line index array in graph base
    graph_base.raceline_index = raceline_index_array

    return state_pos


def vec_angle(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    """

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
