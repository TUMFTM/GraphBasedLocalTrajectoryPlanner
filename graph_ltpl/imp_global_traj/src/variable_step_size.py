import numpy as np


def variable_step_size(kappa: np.ndarray,
                       dist: np.ndarray,
                       d_curve: float,
                       d_straight: float,
                       curve_th: float,
                       force_last: bool = False) -> list:
    """
    Section race track into straight and curve segments (via 'kappa' and 'dist'). Then a subset of the elements on the
    straight and curve segments is chosen (e.g. larger spacing on the straights, tighter spacing in the curves). The
    index of the elements to be kept are returned.

    :param kappa:        curvature array used for selection of normal vectors and corresponding raceline parameters
    :param dist:         euclidean or spline length distance between the given kappa values
    :param d_curve:      min. separation of norm vectors (hosting the nodes) along the ref/race line in curves in m
    :param d_straight:   min. separation of norm vectors (hosting the nodes) along the ref/race line on straights in m
    :param curve_th:     curve thresh. (curvature, i.e. 1/[curve radius]) used to toggle between straight and curve seg.
    :param force_last:   if true, the last point of the series is forced to be in the list of returned indices
    :returns:
        * **idx_array** - array holding the indexes of the suggested values to be used

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        18.01.2019

    """

    next_dist = 0
    next_dist_min = 0
    cur_dist = 0
    idx_array = []
    for idx, (kappa_val, dist_val) in enumerate(zip(kappa, dist)):
        # when travelled the minimal required distance, check if curvature threshold is exceeded
        if (cur_dist + dist_val) > next_dist_min:
            if abs(kappa[idx]) > curve_th:
                next_dist = cur_dist

        # if reached the next intended distance, evaluate if in straight or curve segment
        if (cur_dist + dist_val) > next_dist:
            idx_array.append(idx)

            if abs(kappa[idx]) < curve_th:
                next_dist += d_straight
            else:
                next_dist += d_curve
            next_dist_min = cur_dist + d_curve

        cur_dist += dist_val

    if force_last and len(kappa) - 1 not in idx_array:
        idx_array.append(len(kappa) - 1)

    return idx_array


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
