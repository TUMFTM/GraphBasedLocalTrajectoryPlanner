import numpy as np
import math

# custom modules
import graph_ltpl


def get_s_coord(ref_line: np.ndarray,
                pos: tuple,
                s_array: np.ndarray = None,
                only_index=False,
                closed=False) -> tuple:
    """
    Get the s coordinate of a provided coordinate on a provided coordinate and (optional s) array

    :param ref_line:            numpy array of reference line holding columns with x- and y-coordinates
    :param pos:                 reference position
    :param s_array:             matching s values for the provided 'ref_line', if not set: euclidean distances between
                                the points will be calculated
    :param only_index:          only return indexes and do not calculate coordinates
    :param closed:              boolean flag, set to tro if the reference line should be interpreted as a closed line
    :returns:
        * **s** -               interpolated s value of the provided pos
        * **closest_index** -   index in the 'ref_line' array which is closest to the provided pos

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        13.02.2019

    """

    idx_nb = graph_ltpl.helper_funcs.src.closest_path_index.closest_path_index(path=ref_line,
                                                                               pos=pos,
                                                                               n_closest=1)[0][0]

    if closed:
        idx1 = idx_nb - 1
        idx2 = idx_nb + 1
        if idx2 > (ref_line.shape[0] - 1):
            idx2 = 0
    else:
        idx1 = max(idx_nb - 1, 0)
        idx2 = min(idx_nb + 1, np.size(ref_line, axis=0) - 1)

    # get angle between selected point and neighbours
    ang1 = abs(angle3pt(ref_line[idx_nb, :], pos, ref_line[idx1, :]))
    ang2 = abs(angle3pt(ref_line[idx_nb, :], pos, ref_line[idx2, :]))

    if not only_index:
        # Extract neighboring points (A and B)
        # "pos" is between the closest point and the point resulting in the larger angle
        if ang1 > ang2:
            a_pos = ref_line[idx1, :]
            b_pos = ref_line[idx_nb, :]
        else:
            a_pos = ref_line[idx_nb, :]
            b_pos = ref_line[idx2, :]

        # if s_array not provided
        if s_array is None:
            # calculate squared distances between path array and reference poses
            s_array = np.cumsum(np.sqrt(np.sum(np.power(np.diff(ref_line, axis=0), 2), axis=1)))

        # check if s_array starts at zero, else insert at beginning
        if s_array[0] > 0.05:
            s_array = np.insert(s_array, 0, 0.0)

        # get point perpendicular on the line between the two closest points
        # https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
        t = ((pos[0] - a_pos[0]) * (b_pos[0] - a_pos[0]) + (pos[1] - a_pos[1]) * (b_pos[1] - a_pos[1])) / \
            (np.power(b_pos[0] - a_pos[0], 2) + np.power(b_pos[1] - a_pos[1], 2))
        s_pos = [a_pos[0] + t * (b_pos[0] - a_pos[0]), a_pos[1] + t * (b_pos[1] - a_pos[1])]
        ds = np.sqrt(np.power(a_pos[0] - s_pos[0], 2) + np.power(a_pos[1] - s_pos[1], 2))

        # x_inter = np.linspace(a_pos[0], b_pos[0], 100)
        # y_inter = np.linspace(a_pos[1], b_pos[1], 100)
        # inter_line = np.column_stack((x_inter, y_inter))
        #
        # ds_idx = helper_funcs.src.closest_path_index.closest_path_index(path=inter_line,
        #                                                                     pos=pos,
        #                                                                     n_closest=1)[0][0]
        # get total s
        if ang1 > ang2:
            # ds = (s_array[idx_nb] - s_array[idx_nb - 1]) / 100 * ds_idx
            s = s_array[idx1] + ds
        else:
            # ds = (s_array[idx_nb + 1] - s_array[idx_nb]) / 100 * ds_idx
            s = s_array[idx_nb] + ds
    else:
        s = None

    if ang1 >= ang2:
        closest_indexes = [idx1, idx_nb]
    else:
        closest_indexes = [idx_nb, idx2]

    return s, closest_indexes


def angle3pt(a: tuple, b: tuple, c: tuple) -> float:
    """
    Calculate the angle by turning from coordinate a to c around b.

    :param a:             coordinate a (x, y)
    :param b:             coordinate b (x, y)
    :param c:             coordinate c (x, y)
    :returns:
        * **ang** -       angle between a and c

    """

    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang <= -math.pi:
        ang += 2 * math.pi

    return ang


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
