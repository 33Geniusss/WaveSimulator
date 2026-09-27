"""Ray initialization followed by finite-difference first-pick matching.

Only c1/c2 change. Source, grid, domain, depth, damping and acquisition times
are assumed known. Candidate records are clean; observed noise is not replayed.
"""
from dataclasses import replace
import math
import threading
import time

import numpy as np

from .inversion import InversionResult, invert_first_arrivals
from .model import SimulationCancelled, pick_arrival, simulate


def refine_first_arrivals(config, positions, observed, sample_times, threshold,
                          *, speed_min=.1, speed_max=100., max_evaluations=48,
                          target_rmse=.0001, cancel=None, progress=None):
    started = time.perf_counter()
    cancel = cancel if cancel is not None else threading.Event()
    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0) or not np.isclose(times[-1], config.duration):
        raise ValueError("反演时间轴必须从 0 开始、严格递增，并覆盖模拟时长。")
    if type(max_evaluations) is not int or not 8 <= max_evaluations <= 200:
        raise ValueError("有限差分搜索预算须为 8–200 次。")
    if not math.isfinite(target_rmse) or target_rmse <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("目标到时 RMSE 和拾取门槛须为正的有限数值。")
    if cancel.is_set():
        raise SimulationCancelled()
    dt = float(np.min(np.diff(times)))
    ray = invert_first_arrivals(positions, observed, config.interface_depth,
                               speed_min=speed_min, speed_max=speed_max, sample_interval=dt)
    if not ray.identifiable or ray.c2 is None:
        return replace(ray, ray_stage=ray.to_dict(), termination="ray_unidentifiable",
                       elapsed_seconds=time.perf_counter()-started,
                       warnings=ray.warnings+["几何阶段无法确定两个波速，未启动有限差分精修；请先改善测点布置或首波记录。"])
    observations = np.asarray(observed, dtype=float)
    # Original generating velocities are overwritten BEFORE any candidate PDE
    # config is validated or solved; they never initialize or bound the search.
    template = replace(config, c1=ray.c1, c2=ray.c2, max_frames=2, auto_right=False)
    lower, upper = np.log([speed_min, speed_max])
    gap = min(1e-6, (upper-lower)/10)
    history, cache = [], {}
    fd_calls = 0
    best = None

    def project(x):
        x = np.clip(np.array(x, dtype=float), lower, upper)
        x[0] = min(x[0], upper-gap)
        x[1] = max(x[1], x[0]+gap)
        return x

    def evaluate(x):
        nonlocal fd_calls, best
        if cancel.is_set():
            raise SimulationCancelled()
        x = project(x)
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        if len(history) >= max_evaluations:
            return None
        c1, c2 = np.exp(x)
        candidate = replace(template, c1=float(c1), c2=float(c2))
        record = {"c1": float(c1), "c2": float(c2), "rmse_s": None, "valid": False}
        residual, picks = None, None
        try:
            candidate.validate()
            fd_calls += 1
            result = simulate(candidate, cancel=cancel, minimum_steps=len(times)-1)
            # Use the measured acquisition time grid even when CFL stability
            # requires smaller internal timesteps for a faster candidate.
            picks = [pick_arrival(times, np.interp(times, result.times, result.signal(xr)), threshold) for xr in positions]
            missing = sum(t is None for t in picks)
            record["missing_picks"] = missing
            if missing == 0:
                residual = np.asarray(picks)-observations
                record["rmse_s"] = float(np.sqrt(np.mean(residual**2)))
                record["valid"] = True
            else:
                record["reason"] = "候选记录有未检出的测点"
        except (ValueError, FloatingPointError) as error:
            record["reason"] = str(error)
        answer = {"x": x, "residual": residual, "picks": picks, "score": record["rmse_s"] if residual is not None else math.inf}
        history.append(record)
        cache[key] = answer
        if residual is not None and (best is None or answer["score"] < best["score"]):
            best = answer
        if progress:
            progress(len(history), max_evaluations, None if best is None else best["score"], float(c1), float(c2))
        return answer

    x = project(np.log([ray.c1, ray.c2]))
    initial = evaluate(x)
    initial_rmse = initial["score"] if initial["residual"] is not None else None
    radius = .12
    termination = "evaluation_budget"
    search_budget = max_evaluations-2  # reserve two probes for final sensitivity

    def jacobian_at(base):
        columns = []
        for axis in range(2):
            trial = base["x"].copy()
            h = .003
            trial[axis] += h
            if trial[axis] > upper or (axis == 0 and trial[axis] >= trial[1]-gap):
                trial[axis] = base["x"][axis]-h
            trial = project(trial)
            step = trial[axis]-base["x"][axis]
            item = evaluate(trial)
            if item is None or item["residual"] is None or abs(step) < 1e-9:
                return None
            columns.append((item["residual"]-base["residual"])/step)
        return np.column_stack(columns)

    while len(history) < search_budget-1:
        if cancel.is_set():
            raise SimulationCancelled()
        base = best
        if base is not None and base["score"] <= target_rmse:
            termination = "target_rmse"
            break
        improved = False
        if base is not None:
            x = base["x"].copy()
            jac = jacobian_at(base)
            if jac is not None:
                delta = np.linalg.lstsq(jac, -base["residual"], rcond=1e-6)[0]
                delta *= min(1., .35/max(float(np.linalg.norm(delta)), 1e-12))
                for fraction in (1., .5, .25):
                    if len(history) >= search_budget:
                        break
                    item = evaluate(x+fraction*delta)
                    if item is not None and item["score"] < base["score"]-1e-10:
                        improved = True
                        break
        if not improved:
            # Derivative-free fallback for threshold jumps or missing picks.
            before = math.inf if base is None else base["score"]
            for axis in range(2):
                for sign in (1, -1):
                    if len(history) >= search_budget:
                        break
                    trial = x.copy()
                    trial[axis] += sign*radius
                    evaluate(trial)
            improved = best is not None and best["score"] < before-1e-10
            if not improved:
                if best is None:
                    radius *= 2  # expand rather than hard-lock to a narrow box
                    if radius > upper-lower:
                        termination = "no_valid_candidate"
                        break
                else:
                    radius *= .5
                    if radius < .0005:
                        termination = "stalled"
                        break
        else:
            radius = max(.01, radius*.7)
    if best is None:
        raise ValueError("有限差分候选均未得到完整首波记录；请检查记录时长、门槛和搜索范围。")
    # Probe one fixed best point; probes are diagnostics, not a second optimizer.
    final = best
    jac = jacobian_at(final)
    c1, c2 = map(float, np.exp(final["x"]))
    singular = np.linalg.svd(jac, compute_uv=False) if jac is not None else None
    identifiable = bool(singular is not None and singular[1] > max(1e-8, 1e-6*singular[0]))
    condition = float(singular[0]/singular[1]) if identifiable else None
    if final["score"] <= target_rmse:
        termination = "target_rmse"
    warnings = ["几何阶段："+message for message in ray.warnings]
    warnings.append("有限差分与观测使用相同源模型和网格；同模型合成数据的高精度不代表独立验证。")
    if not identifiable:
        warnings.append("有限差分局部灵敏度不足或检查未完成，无法可靠确定 c2。")
    elif condition > 100:
        warnings.append("有限差分到时对两个波速的灵敏度差异较大，结果可能不稳定。")
    if termination != "target_rmse":
        warnings.append("尚未达到目标 RMSE；显示当前最佳结果，可增加预算或检查首波拾取。")
    if any(abs(v-speed_min) < 1e-5*speed_max or abs(v-speed_max) < 1e-5*speed_max for v in (c1, c2)):
        warnings.append("估计触及速度搜索边界，结果受搜索范围限制。")
    residual = final["residual"]
    dc = 2*config.interface_depth*math.sqrt((c2+c1)/(c2-c1))
    return InversionResult(c1, c2 if identifiable else None, config.interface_depth, speed_min, speed_max,
                           np.abs(positions).tolist(), observations.tolist(), list(final["picks"]),
                           residual.tolist(), ["头波（射线参考）" if abs(d) > dc else "直达波（射线参考）" for d in positions],
                           final["score"], float(np.max(abs(residual))), identifiable, condition, dc, warnings,
                           method="ray_initialized_fd_first_arrival_least_squares", ray_stage=ray.to_dict(),
                           fd_evaluations=fd_calls, fd_initial_rmse=initial_rmse,
                           elapsed_seconds=time.perf_counter()-started, termination=termination,
                           evaluation_history=history)
