import numpy as np

# custom modules
import graph_ltpl


def check_inside_bounds(bound1: np.ndarray,
                        bound2: np.ndarray,
                        pos: list) -> bool:
    """
    Check if the provided pos is within the bounds.

    :param bound1:            bound coordinates (numpy array with columns x and y)
    :param bound2:            bound coordinates (numpy array with columns x and y)
    :param pos:               position to be checked
    :returns:
        * **within_bounds** - boolean flag - 'True', when the position 'pos' is within the bounds

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        19.03.2019

    """

    # calculate center line (between bounds)
    centerline = (bound1 + bound2) / 2

    # get bound index before and after
    b_idx = graph_ltpl.helper_funcs.src.get_s_coord.get_s_coord(ref_line=centerline,
                                                                pos=tuple(pos),
                                                                only_index=True,
                                                                closed=True)[1]

    # interpolate bounds and center line between closest points
    bound1 = np.column_stack((np.linspace(bound1[b_idx[0], 0], bound1[b_idx[1], 0]),
                              np.linspace(bound1[b_idx[0], 1], bound1[b_idx[1], 1])))
    bound2 = np.column_stack((np.linspace(bound2[b_idx[0], 0], bound2[b_idx[1], 0]),
                              np.linspace(bound2[b_idx[0], 1], bound2[b_idx[1], 1])))
    centerline = np.column_stack((np.linspace(centerline[b_idx[0], 0], centerline[b_idx[1], 0]),
                                  np.linspace(centerline[b_idx[0], 1], centerline[b_idx[1], 1])))

    # get closest interpolated bound pair
    b_idx = graph_ltpl.helper_funcs.src.closest_path_index.closest_path_index(path=centerline,
                                                                              pos=tuple(pos))[0][0]

    # get track width at selected index
    d_track_2 = (np.power(bound1[b_idx, :][0] - bound2[b_idx, :][0], 2)
                 + np.power(bound1[b_idx, :][1] - bound2[b_idx, :][1], 2))

    # get distances from ego to track bounds
    d_b1_2 = np.power(bound1[b_idx, :][0] - pos[0], 2) + np.power(bound1[b_idx, :][1] - pos[1], 2)
    d_b2_2 = np.power(bound2[b_idx, :][0] - pos[0], 2) + np.power(bound2[b_idx, :][1] - pos[1], 2)

    # Validate position
    within_bounds = not (d_b1_2 > d_track_2 or d_b2_2 > d_track_2)

    return within_bounds
