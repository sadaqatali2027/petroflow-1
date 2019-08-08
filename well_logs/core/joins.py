"""Various types of DataFrame joins."""

import numpy as np
import pandas as pd


def cross_join(left, right):
    """Return the cartesian product of rows from both tables.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to join.

    Returns
    -------
    df : pandas.DataFrame
        Cross-joined `DataFrame`.
    """
    left["_TMP_KEY"] = 1
    right["_TMP_KEY"] = 1
    return pd.merge(left, right, on="_TMP_KEY").drop("_TMP_KEY", axis=1)


def between_join(left, right, left_on="DEPTH", right_on=("DEPTH_FROM", "DEPTH_TO")):
    """Return a `DataFrame`, consisting of all combinations of rows from both
    `left` and `right`, so that `left.left_on` is between `right.right_on[0]`
    (inclusive) and `right.right_on[1]` (exclusive).

    Parameters
    ----------
    left : pandas.DataFrame
        Table with depth values.
    right : pandas.DataFrame
        Table with depth ranges.
    left_on : str
        Depth column in `left`. Defaults to `"DEPTH"`.
    right_on : tuple of two str
        Columns, specifying depth ranges in `right`. Defaults to
        `("DEPTH_FROM", "DEPTH_TO")`.

    Returns
    -------
    df : pandas.DataFrame
        Joined `DataFrame`.
    """
    # TODO: avoid cross join
    right_from, right_to = right_on
    cross = cross_join(left, right)
    mask = (cross[left_on] >= cross[right_from]) & (cross[left_on] < cross[right_to])
    return cross[mask]


def fdtd_join(left, right, left_on=("DEPTH_FROM", "DEPTH_TO"), right_on=("DEPTH_FROM", "DEPTH_TO")):
    """Return full outer join of two tables on depth ranges, specified in
    `left_on` and `right_on` (from depth - to depth join).

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to join.
    left_on, right_on : tuple of two str
        Columns, specifying depth ranges in `left` and `right` respectively.
        Defaults to `("DEPTH_FROM", "DEPTH_TO")`.

    Returns
    -------
    df : pandas.DataFrame
        Joined `DataFrame`.
    """
    # TODO: avoid cross joins
    left_from, left_to = left_on
    right_from, right_to = right_on

    depths = pd.concat([left["DEPTH_FROM"], left["DEPTH_TO"], right["DEPTH_FROM"], right["DEPTH_TO"]])
    depths = np.sort(pd.unique(depths))
    res = pd.DataFrame({"_TMP_DEPTH_FROM": depths[:-1],
                        "_TMP_DEPTH_TO": depths[1:]})

    res = cross_join(res, left)
    mask = (res["_TMP_DEPTH_FROM"] >= res[left_from]) & (res["_TMP_DEPTH_TO"] <= res[left_to])
    res = res[mask].drop([left_from, left_to], axis=1)

    res = cross_join(res, right)
    mask = (res["_TMP_DEPTH_FROM"] >= res[right_from]) & (res["_TMP_DEPTH_TO"] <= res[right_to])
    res = res[mask].drop([right_from, right_to], axis=1)

    res = res.rename(columns={"_TMP_DEPTH_FROM": "DEPTH_FROM", "_TMP_DEPTH_TO": "DEPTH_TO"})
    return res
