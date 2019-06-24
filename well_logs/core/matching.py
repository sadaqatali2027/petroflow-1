from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def select_contigious_intervals(df, max_gap=0):
    split_indices = np.where((df["DEPTH_FROM"] - df["DEPTH_TO"].shift()) > max_gap)[0]
    return np.split(df, split_indices)


def generate_init_deltas(n_lith_ints, gap_lengths, segment_delta_from, segment_delta_to, segment_delta_step):
    interval_deltas = []
    for n, gap_length in zip(n_lith_ints, gap_lengths):
        interval_deltas.append([np.zeros(n),  # Unrecovered core in the end
                                np.full(n, gap_length / (n + 1)),  # Unrecovered core split equally
                                np.array([gap_length] + [0] * (n - 1))])  # Unrecovered core in the beginning
    interval_deltas = [np.concatenate(delta) for delta in zip(*interval_deltas)]
    segment_delta = np.arange(segment_delta_from, segment_delta_to, segment_delta_step)
    return [np.concatenate([[d1], d2]) for d1, d2 in product(segment_delta, interval_deltas)]


def loss(deltas, n_lith_ints, core_depths, log_interpolator, core_log):
    segment_delta = deltas[0]
    interval_deltas = np.concatenate([np.cumsum(d) for d in np.split(deltas[1:], np.cumsum(n_lith_ints)[:-1])])
    interval_deltas += segment_delta
    shifted_depths = []
    for depths, deltas in zip(core_depths, interval_deltas):
        shifted_depths.append(depths + deltas)
    shifted_depths = np.concatenate(shifted_depths)
    well_log = np.nan_to_num(log_interpolator(shifted_depths)).reshape(-1, 1)

    reg = LinearRegression().fit(well_log, core_log)
    r2 = r2_score(core_log, reg.predict(well_log))
    return -r2


def trunc(values, decimals=0):
    return np.trunc(values * 10**decimals) / (10**decimals)


def match_segment(segment, lithology_intervals, well_log, core_log, max_shift, delta_from, delta_to, delta_step):
    well_depth_from = well_log.index.min()
    well_depth_to = well_log.index.max()
    well_log = well_log.dropna()
    log_interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")

    n_lith_ints = []
    gap_lengths = []

    core_depths = []
    core_logs = []

    segment = segment[["DEPTH_FROM", "DEPTH_TO", "CORE_RECOVERY"]]
    for _, (segment_depth_from, segment_depth_to, recovery) in segment.iterrows():
        mask = ((lithology_intervals["DEPTH_FROM"] >= segment_depth_from) &
                (lithology_intervals["DEPTH_TO"] <= segment_depth_to))
        segment_lithology_intervals = lithology_intervals[mask]
        n_lith_ints.append(len(segment_lithology_intervals))
        gap_lengths.append(max(0, segment_depth_to - segment_depth_from - recovery))

        for _, (depth_from, depth_to) in segment_lithology_intervals.iterrows():
            log_slice = core_log[depth_from:depth_to]
            core_depths.append(log_slice.index.values)
            core_logs.append(log_slice.values)

    core_logs = np.concatenate(core_logs)

    # Optimization constraints
    constraints = []

    starts = np.cumsum([0] + n_lith_ints) + 1
    for start, end, gap_length in zip(starts[:-1], starts[1:], gap_lengths):
        def con_gap_length(x, start=start, end=end, gap_length=gap_length):
            return gap_length - x[start:end].sum()
        constraints.append({"type": "ineq", "fun": con_gap_length})

    for i in range(sum(n_lith_ints)):
        def con_non_negative_gap(x, i=i):
            return x[i + 1]
        constraints.append({"type": "ineq", "fun": con_non_negative_gap})

    max_shift_up = min(max_shift, max(0, segment["DEPTH_FROM"].min() - well_depth_from))
    def con_max_shift_up(x):
        return x[0] + max_shift_up
    constraints.append({"type": "ineq", "fun": con_max_shift_up})

    max_shift_down = min(max_shift, max(0, well_depth_to - segment["DEPTH_TO"].max()))
    def con_max_shift_down(x):
        return max_shift_down - x[0]
    constraints.append({"type": "ineq", "fun": con_max_shift_down})

    # Optimization
    init_deltas = generate_init_deltas(n_lith_ints, gap_lengths, delta_from, delta_to, delta_step)

    best_loss = None
    best_deltas = None
    for init_delta in init_deltas:
        res = minimize(loss, init_delta, args=(n_lith_ints, core_depths, log_interpolator, core_logs),
                       constraints=constraints, method="COBYLA")
        if best_loss is None or res.fun < best_loss:
            best_loss = res.fun
            best_deltas = res.x

    # Computing of final deltas
    segment_delta = trunc(best_deltas[0], 2)
    interval_deltas = np.clip(trunc(best_deltas[1:], 2), 0, None)
    interval_deltas = np.concatenate([np.cumsum(d) for d in np.split(interval_deltas, np.cumsum(n_lith_ints)[:-1])])
    interval_deltas += segment_delta

    mask = ((lithology_intervals["DEPTH_FROM"] >= segment["DEPTH_FROM"].min()) &
            (lithology_intervals["DEPTH_TO"] <= segment["DEPTH_TO"].max()))
    segment_lithology_intervals = lithology_intervals[mask]
    segment_lithology_intervals["DELTA"] = interval_deltas
    segment["DELTA"] = segment_delta
    return segment, segment_lithology_intervals
