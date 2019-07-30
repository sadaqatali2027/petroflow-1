from itertools import product
from collections import namedtuple

import multiprocess as mp
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def trunc(values, decimals=0):
    return np.trunc(values * 10**decimals) / (10**decimals)


def select_contigious_intervals(df, max_gap=0):
    split_indices = np.where((df["DEPTH_FROM"] - df["DEPTH_TO"].shift()) > max_gap)[0]
    return np.split(df, split_indices)


def generate_init_deltas(bi_n_lith_ints, bi_gap_lengths, segment_delta_from, segment_delta_to, segment_delta_step):
    interval_deltas = []
    for n, gap_length in zip(bi_n_lith_ints, bi_gap_lengths):
        interval_deltas.append([np.zeros(n),  # Unrecovered core in the end
                                np.full(n, gap_length / (n + 1)),  # Unrecovered core split equally
                                np.array([gap_length] + [0] * (n - 1))])  # Unrecovered core in the beginning
    interval_deltas = [np.concatenate(delta) for delta in zip(*interval_deltas)]
    segment_delta = np.arange(segment_delta_from, segment_delta_to, segment_delta_step)
    return [np.concatenate([[d1], d2]) for d1, d2 in product(segment_delta, interval_deltas)]


def loss(deltas, bi_n_lith_ints, core_depths, log_interpolator, core_log):
    segment_delta = deltas[0]
    interval_deltas = np.concatenate([np.cumsum(d) for d in np.split(deltas[1:], np.cumsum(bi_n_lith_ints)[:-1])])
    interval_deltas += segment_delta
    shifted_depths = []
    for depths, deltas in zip(core_depths, interval_deltas):
        shifted_depths.append(depths + deltas)
    shifted_depths = np.concatenate(shifted_depths)
    well_log = np.nan_to_num(log_interpolator(shifted_depths))
    # TODO: find out why NaNs appear
    return -np.corrcoef(well_log, core_log)[0, 1]


def match_segment(segment, lithology_intervals, well_log, core_log, max_shift, delta_from, delta_to, delta_step,
                  max_iter, timeout):
    well_depth_from = well_log.index.min()
    well_depth_to = well_log.index.max()
    well_log = well_log.dropna()
    log_interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")

    bi_n_lith_ints = []
    bi_gap_lengths = []

    core_depths = []
    core_logs = []

    segment = segment[["DEPTH_FROM", "DEPTH_TO", "CORE_RECOVERY"]]
    segment_depth_from = segment["DEPTH_FROM"].min()
    segment_depth_to = segment["DEPTH_TO"].max()
    core_len = len(core_log[segment_depth_from:segment_depth_to])

    for _, (bi_depth_from, bi_depth_to, recovery) in segment.iterrows():
        mask = ((lithology_intervals["DEPTH_FROM"] >= bi_depth_from) &
                (lithology_intervals["DEPTH_TO"] <= bi_depth_to))
        segment_lithology_intervals = lithology_intervals[mask]
        bi_n_lith_ints.append(len(segment_lithology_intervals))
        bi_gap_lengths.append(max(0, bi_depth_to - bi_depth_from - recovery))

        for _, (li_depth_from, li_depth_to) in segment_lithology_intervals.iterrows():
            log_slice = core_log[li_depth_from:li_depth_to]
            core_depths.append(log_slice.index.values)
            core_logs.append(log_slice.values)

    core_logs = np.concatenate(core_logs)

    # Optimization constraints
    constraints = []

    starts = np.cumsum([0] + bi_n_lith_ints) + 1
    for start, end, gap_length in zip(starts[:-1], starts[1:], bi_gap_lengths):
        def con_gap_length(x, start=start, end=end, gap_length=gap_length):
            return gap_length - x[start:end].sum()
        constraints.append({"type": "ineq", "fun": con_gap_length})

    for i in range(sum(bi_n_lith_ints)):
        def con_non_negative_gap(x, i=i):
            return x[i + 1]
        constraints.append({"type": "ineq", "fun": con_non_negative_gap})

    max_shift_up = min(max_shift, max(0, segment_depth_from - well_depth_from))
    def con_max_shift_up(x):
        return x[0] + max_shift_up
    constraints.append({"type": "ineq", "fun": con_max_shift_up})

    max_shift_down = min(max_shift, max(0, well_depth_to - segment_depth_to))
    def con_max_shift_down(x):
        return max_shift_down - x[0]
    constraints.append({"type": "ineq", "fun": con_max_shift_down})

    # Optimization
    Shift = namedtuple("Shift", ["depth_from", "depth_to", "segment_delta", "interval_deltas", "loss"])

    zero_deltas = np.zeros(np.sum(bi_n_lith_ints) + 1)
    zero_shift_loss = loss(zero_deltas, bi_n_lith_ints, core_depths, log_interpolator, core_logs)
    zero_shift = Shift(segment_depth_from, segment_depth_to, 0, zero_deltas[1:], zero_shift_loss)
    shifts = [zero_shift]

    if core_len <= 1 or core_len < segment["CORE_RECOVERY"].sum():
        return shifts

    futures = []
    init_deltas = generate_init_deltas(bi_n_lith_ints, bi_gap_lengths, delta_from, delta_to, delta_step)
    with mp.Pool() as pool:
        for init_delta in init_deltas:
            args = (loss, init_delta)
            kwargs = {
                "args": (bi_n_lith_ints, core_depths, log_interpolator, core_logs),
                "method": "SLSQP",
                "options": {"maxiter": max_iter, "ftol": 1e-3},
                "constraints": constraints,
            }
            futures.append(pool.apply_async(minimize, args=args, kwds=kwargs))

        for future, init_delta in zip(futures, init_deltas):
            try:
                res = future.get(timeout=timeout)
                future_deltas = res.x
                future_loss = res.fun
            except mp.TimeoutError:
                future_deltas = init_delta
                future_loss = loss(future_deltas, bi_n_lith_ints, core_depths, log_interpolator, core_logs)

            segment_delta = trunc(future_deltas[0], 2)
            interval_deltas = np.clip(trunc(future_deltas[1:], 2), 0, None)
            interval_deltas = [np.cumsum(d) for d in np.split(interval_deltas, np.cumsum(bi_n_lith_ints)[:-1])]
            interval_deltas = np.concatenate(interval_deltas) + segment_delta

            shift = Shift(segment_depth_from + segment_delta, segment_depth_to + segment_delta,
                          segment_delta, interval_deltas, future_loss)
            shifts.append(shift)
    return shifts
