import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..batchflow import parallel

def select_contigious_intervals(samples_df, max_shift):
    split_indices = np.where((samples_df["DEPTH_FROM"] - samples_df["DEPTH_TO"].shift()) >= max_shift * 2)[0]
    return np.split(samples_df, split_indices)

def sample_gen(samples_df):
    for row in samples_df.iterrows():
        sample_id = row[0]
        depth_from, depth_to = row[1][["DEPTH_FROM", "DEPTH_TO"]]
        yield sample_id, depth_from, depth_to

def core_gen(core_series):
    for depth, log in core_series.iteritems():
        yield depth, log

def join_samples(samples_df, core_series):
    samples_iter = sample_gen(samples_df)
    core_iter = core_gen(core_series)

    sample_id, sample_depth_from, sample_depth_to = next(samples_iter)
    core_depth, core_log = next(core_iter)

    rows = []
    while True:
        try:
            if core_depth < sample_depth_from:
                core_depth, core_log = next(core_iter)
            elif core_depth >= sample_depth_to:
                sample_id, sample_depth_from, sample_depth_to = next(samples_iter)
            else:
                rows.append((sample_id, sample_depth_from, sample_depth_to, core_depth, core_log))
                core_depth, core_log = next(core_iter)
        except StopIteration:
            break
    columns = ["SAMPLE", "DEPTH_FROM", "DEPTH_TO", "CORE_DEPTH", "LOG"]
    joined_df = pd.DataFrame(rows, columns=columns)
    return joined_df

def loss(deltas, depths, log_interpolator, target):
    pred = np.concatenate([log_interpolator(sample_depths + sample_delta)
                           for sample_depths, sample_delta in zip(depths, deltas)])
    pred = (pred - pred.mean()) / (pred.std() + 1e-10)
    loss = np.mean(np.abs(target - pred))
    return loss

def init_fn(*args, bounds, delta_from, delta_to, delta_step, **kwargs):
    return [np.full(len(bounds), delta) for delta in np.arange(delta_from, delta_to, delta_step)]

def post_fn(results, *args, **kwargs):
    losses, deltas = zip(*results)
    ix = np.argmin(losses)
    return losses[ix], deltas[ix]

@parallel(init=init_fn, post=post_fn, target="for")
def minimize_loss(delta, loss, depths, log_interpolator, target, bounds, constraints,
                  delta_from, delta_to, delta_step):
    res = minimize(loss, delta, args=(depths, log_interpolator, target),
                   bounds=bounds, constraints=constraints)
    return res.fun, res.x

def optimize_shift(samples_df, joined_df, log_interpolator, max_shift,
                   delta_from=-0.5, delta_to=0.5, delta_step=0.5):
    grouped_df = joined_df.groupby("SAMPLE")
    target = []
    depths = []
    for _, group in grouped_df:
        target.append(group["LOG"].to_numpy())
        depths.append(group["CORE_DEPTH"].to_numpy())
    target = np.concatenate(target)
    target = (target - target.mean()) / (target.std() + 1e-10)

    samples = samples_df.index
    constraints = []
    for i, (s1, s2) in enumerate(zip(samples[:-1], samples[1:])):
        def con(x, i=i, s1=s1, s2=s2):
            return samples_df.loc[s2]["DEPTH_FROM"] - samples_df.loc[s1]["DEPTH_TO"] + x[i + 1] - x[i]
        constraints.append({"type": "ineq", "fun": con})
    bounds = [(-max_shift, max_shift)] * len(samples_df)
    
    beta_loss, beta_deltas = minimize_loss(loss, depths, log_interpolator, target, bounds=bounds,
                                           constraints=constraints, delta_from=delta_from,
                                           delta_to=delta_to, delta_step=delta_step)
    return beta_loss, beta_deltas
