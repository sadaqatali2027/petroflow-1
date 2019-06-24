import numpy as np
import pandas as pd


def cross_join(left, right):
    left["key"] = 1
    right["key"] = 1
    return pd.merge(left, right, on="key").drop("key", axis=1)


def between_join(left, right, left_on="DEPTH", right_on=("DEPTH_FROM", "DEPTH_TO")):
    right_from, right_to = right_on
    cross = cross_join(left, right)
    mask = (cross[left_on] >= cross[right_from]) & (cross[left_on] < cross[right_to])
    return cross[mask]


def fdtd_join(left, right, left_on=("DEPTH_FROM", "DEPTH_TO"), right_on=("DEPTH_FROM", "DEPTH_TO")):
    left_from, left_to = left_on
    right_from, right_to = right_on

    depths = pd.concat([left["DEPTH_FROM"], left["DEPTH_TO"], right["DEPTH_FROM"], right["DEPTH_TO"]])
    depths = np.sort(pd.unique(depths))
    res = pd.DataFrame({"TMP_DEPTH_FROM": depths[:-1],
                        "TMP_DEPTH_TO": depths[1:]})

    res = cross_join(res, left)
    mask = (res["TMP_DEPTH_FROM"] >= res[left_from]) & (res["TMP_DEPTH_TO"] <= res[left_to])
    res = res[mask].drop([left_from, left_to], axis=1)

    res = cross_join(res, right)
    mask = (res["TMP_DEPTH_FROM"] >= res[right_from]) & (res["TMP_DEPTH_TO"] <= res[right_to])
    res = res[mask].drop([right_from, right_to], axis=1)

    res = res.rename(columns={"TMP_DEPTH_FROM": "DEPTH_FROM", "TMP_DEPTH_TO": "DEPTH_TO"})
    return res
