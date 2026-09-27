"""Known-depth, two-layer ray inversion of measured first-arrival times.

No SimulationConfig or generating velocities enter the inverse problem.
We minimize unweighted squared time residuals, with common speed bounds.
For phi=acos(c1/c2), t=a*g(phi), a=1/c1. The optimal a at each phi
is an exactly solved, bounded scalar linear least-squares problem.
"""
from dataclasses import asdict, dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class InversionResult:
    c1: float
    c2: float | None
    depth: float
    speed_min: float
    speed_max: float
    offsets: list[float]
    observed_times: list[float]
    fitted_times: list[float]
    residuals: list[float]
    branches: list[str]
    rmse: float
    max_residual: float
    identifiable: bool
    condition_number: float | None
    crossover_distance: float | None
    warnings: list[str]
    method: str = "known_depth_two_layer_ray_least_squares"
    ray_stage: dict | None = None
    fd_evaluations: int = 0
    fd_initial_rmse: float | None = None
    elapsed_seconds: float = 0.
    termination: str = ""
    evaluation_history: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def travel_times(offsets, depth: float, c1: float, c2: float) -> np.ndarray:
    distances = np.abs(np.asarray(offsets, dtype=float))
    direct = distances / c1
    if c2 <= c1:
        return direct
    head = distances / c2 + 2 * depth * math.sqrt(1/c1**2 - 1/c2**2)
    # Below the critical distance this line lies above the direct line, so it
    # cannot win the minimum even though its physical head ray is unavailable.
    return np.minimum(direct, head)


def _golden_minimum(function, left, right):
    ratio = (math.sqrt(5)-1)/2
    a = right-ratio*(right-left)
    b = left+ratio*(right-left)
    fa, fb = function(a), function(b)
    for _ in range(72):
        if fa <= fb:
            right, b, fb = b, a, fa
            a = right-ratio*(right-left)
            fa = function(a)
        else:
            left, a, fa = a, b, fb
            b = left+ratio*(right-left)
            fb = function(b)
    return (a, fa) if fa < fb else (b, fb)


def invert_first_arrivals(offsets, arrivals, depth: float, *, speed_min=.1,
                          speed_max=100., sample_interval=0.) -> InversionResult:
    """Fit a variable number of picks, with known origin time and layer depth.

    Search is independent of simulation truth. Equal-speed/all-direct solutions
    are retained as a diagnostic boundary case, never reported as recovered c2.
    The dense one-dimensional profile is refined around every sampled local
    minimum; branch-transition points are explicitly included in the search.
    """
    d = np.abs(np.asarray(offsets, dtype=float))
    if d.ndim != 1 or d.size == 0 or np.ndim(arrivals) != 1 or len(arrivals) != d.size:
        raise ValueError("源距和首波到时必须是非空、等长的一维数组。")
    if any(t is None for t in arrivals):
        missing = ", ".join(str(i+1) for i, t in enumerate(arrivals) if t is None)
        raise ValueError(f"第 {missing} 道未检出首波。请检查记录时长、门槛及测点位置后重试。")
    t = np.asarray(arrivals, dtype=float)
    if not np.isfinite(d).all() or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("位置和首波到时必须为有限数值，到时不能为负。")
    if not all(math.isfinite(v) for v in (depth, speed_min, speed_max, sample_interval)) or depth <= 0 or not 0 < speed_min < speed_max or sample_interval < 0:
        raise ValueError("界面深度须为正，速度下限须大于零且小于上限，采样间隔不能为负。")
    distinct_offsets = len(np.unique(np.round(d[d > 0], 10)))
    if distinct_offsets < 2:
        raise ValueError("同时反推两个波速至少需要 2 个不同的非零源距；一个接收器或同源距重复测点的信息不足。")
    if np.any((d > 0) & (t <= 0)):
        raise ValueError("非原点接收器在 t=0 触发，疑似噪声误检；请调整门槛或噪声后重试。")

    def profile(phi):
        cosine = np.cos(phi)
        g = np.minimum(d, np.asarray(cosine)[..., None]*d + 2*depth*np.asarray(np.sin(phi))[..., None])
        lower = 1/(speed_max*cosine)
        upper = 1/speed_min
        a = np.clip(np.sum(g*t, axis=-1)/np.sum(g*g, axis=-1), lower, upper)
        residual = np.asarray(a)[..., None]*g-t
        return np.sum(residual**2, axis=-1), a

    phi_max = math.acos(speed_min/speed_max)
    switches = 2*np.arctan2(2*depth, d[d > 0])
    grid = np.unique(np.r_[np.linspace(0, phi_max, 2049), switches[switches < phi_max]])
    scores, _ = profile(grid)
    candidates = [(float(grid[0]), float(scores[0])), (float(grid[-1]), float(scores[-1]))]
    # Flat all-direct intervals need no refinement. Include all sampled minima
    # and the exact switches, then refine nonflat local basins.
    best = int(np.argmin(scores))
    candidates.append((float(grid[best]), float(scores[best])))
    for i in range(1, len(grid)-1):
        if scores[i] <= scores[i-1] and scores[i] <= scores[i+1] and (scores[i] < scores[i-1] or scores[i] < scores[i+1]):
            candidates.append(_golden_minimum(lambda x: float(profile(x)[0]), float(grid[i-1]), float(grid[i+1])))
    phi, score = min(candidates, key=lambda candidate: candidate[1])
    # Prefer the all-direct diagnostic when an apparent head fit only improves
    # floating-point roundoff; never select a c2 arbitrarily on a flat profile.
    if float(scores[0])-score <= 1e-12 * max(float(np.dot(t, t)), 1e-24):
        phi, score = 0., float(scores[0])
    _, a = profile(phi)
    c1, c2 = 1/float(a), 1/(float(a)*math.cos(phi))
    fitted = travel_times(d, depth, c1, c2)
    residuals = fitted-t
    direct = d/c1
    is_head = fitted < direct-1e-9*max(float(np.max(t)), 1e-12)
    warnings = []
    if distinct_offsets == 2:
        warnings.append("只有 2 个不同的非零源距，缺少额外几何测点交叉检查；即使残差很小，噪声和拾取偏差仍可能显著影响波速。")
    jacobian = np.zeros((len(d), 2))  # derivatives with respect to log speeds
    jacobian[~is_head, 0] = -direct[~is_head]
    if np.any(is_head):
        root = math.sqrt(1/c1**2-1/c2**2)
        jacobian[is_head, 0] = -2*depth/(c1**2*root)
        jacobian[is_head, 1] = -d[is_head]/c2+2*depth/(c2**2*root)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    identifiable = bool(singular[1] > 1e-8*singular[0])
    condition = float(singular[0]/singular[1]) if identifiable else None
    crossover = 2*depth*math.sqrt((c2+c1)/(c2-c1)) if c2 > c1 else None
    if not identifiable:
        warnings.append("在当前搜索范围内，最优拟合只有直达波信息或灵敏度不足，无法确定 c2；请核对搜索范围并扩大测点源距，使记录包含头波首达。")
    elif condition > 100:
        warnings.append("两个波速的灵敏度差异较大，反演不稳定；微小拾取误差可能明显改变结果。")
    if identifiable and np.all(is_head):
        warnings.append("所有测点在拟合模型中均为头波首达；c1 依赖已知深度和截距，建议增加近源测点交叉检查。")
    if min(abs(c1-speed_min), abs(c1-speed_max), abs(c2-speed_min), abs(c2-speed_max)) < 1e-5*speed_max:
        warnings.append("估计触及速度搜索边界；请核对数据并调整上下限，当前结果受搜索范围限制。")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if rmse > max(2*sample_interval, .05*float(np.median(t))):
        warnings.append("走时拟合误差较大，可能存在噪声误触发、漏检弱头波、拾取偏差或模型不适用；不要直接采信估计。")
    if np.any((d > 0) & (t < d/speed_max)):
        warnings.append("部分到时早于搜索速度上限允许的传播时间；请检查噪声、门槛和搜索范围。")
    if np.any(d == 0):
        warnings.append("原点接收器不提供几何走时速度信息，其到时仅参与残差显示。")
    if crossover is not None and np.any(abs(d-crossover) < .02*max(crossover, 1e-12)):
        warnings.append("有测点接近拟合交叉距离，直达/头波分类对拾取误差敏感。")
    return InversionResult(c1, c2 if identifiable else None, depth, speed_min, speed_max,
                           d.tolist(), t.tolist(), fitted.tolist(), residuals.tolist(),
                           ["头波" if value else "直达波" for value in is_head], rmse,
                           float(np.max(abs(residuals))), identifiable, condition,
                           crossover if identifiable else None, warnings)
