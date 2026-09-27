from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import threading
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    """Independent discrete Gaussian measurement noise; never part of the PDE."""
    enabled: bool = False
    std_au: float = 0.00002
    seed: int = 6730

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("噪声开关必须是布尔值。")
        if isinstance(self.std_au, bool) or not isinstance(self.std_au, (int, float)) or not math.isfinite(self.std_au) or not 0 <= self.std_au <= 1:
            raise ValueError("噪声标准差须为 0–1 a.u. 内的有限数值。")
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647:
            raise ValueError("噪声种子须为 0–2147483647 内的整数。")


def add_measurement_noise(signal: np.ndarray, receiver_id: int, noise: NoiseConfig) -> np.ndarray:
    """Stable per-channel realization, independent of receiver order or position.

    Recreating the explicit PCG64 stream makes redraws, threshold changes and
    exports identical. Moving the same channel retains its noise realization.
    """
    noise.validate()
    clean = np.asarray(signal, dtype=float)
    if not noise.enabled or noise.std_au == 0:
        return clean.copy()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([noise.seed, receiver_id])))
    return clean + noise.std_au * rng.standard_normal(clean.shape)


@dataclass(frozen=True)
class SimulationConfig:
    width: float = 8.0
    depth: float = 4.0
    x_min: float = -2.0
    y_min: float = -2.0
    spacing: float = 0.025
    c1: float = 1.0
    c2: float = 2.0
    interface_depth: float = 1.0
    source_x: float = 0.0
    source_depth: float = 0.0
    frequency: float = 3.0
    duration: float = 7.0
    cfl: float = 0.85
    boundary: str = "sponge"
    sponge_width: float = 0.8
    sponge_strength: float = 4.0
    max_frames: int = 240
    cache_mb: int = 128
    auto_right: bool = False
    crossover_factor: float = 1.5

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name != "boundary" and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name} 必须是有限数值。")
        for name in ("width", "depth", "spacing", "c1", "c2", "frequency", "duration"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于零。")
        if not 0 < self.cfl <= 0.95:
            raise ValueError("CFL 安全系数必须在 (0, 0.95] 内。")
        if not isinstance(self.auto_right, bool) or not 1.1 <= self.crossover_factor <= 3:
            raise ValueError("自动右边界开关必须是布尔值，覆盖倍数须在 1.1–3 内。")
        if self.source_x != 0 or self.source_depth != 0:
            raise ValueError("此版本波源固定为 (0, 0)，接收器固定在 y=0。")
        if not self.y_min < 0 < self.interface_depth < self.depth or not self.x_min < 0 < self.width:
            raise ValueError("须满足左/上边界 < 0，且 0 < 界面深度 < 下边界。")
        if self.boundary not in {"sponge", "reflecting", "dirichlet"}:
            raise ValueError("未知边界类型。")
        if self.spacing > min(self.x_span, self.y_span) / 12:
            raise ValueError("网格过粗：每个方向至少需要 12 个单元。")
        if not 0 < self.sponge_width < min(self.x_span, self.y_span) / 2:
            raise ValueError("吸收层宽度必须小于区域宽度和高度的一半。")
        if self.sponge_strength < 0:
            raise ValueError("阻尼系数不能为负。")
        if self.boundary == "sponge":
            # Keep the finite source footprint (4 sigma) out of every sponge.
            source_margin = self.sponge_width + 4.8 * self.spacing
            if min(-self.x_min, self.width, -self.y_min, self.depth) <= source_margin:
                raise ValueError("四周边界离原点太近；需为吸收层和波源留足缓冲。")
        if not isinstance(self.max_frames, int) or self.max_frames < 2:
            raise ValueError("缓存帧数须为至少 2 的整数。")
        if not isinstance(self.cache_mb, int) or not 8 <= self.cache_mb <= 1024:
            raise ValueError("缓存预算须为 8–1024 MiB 的整数。")
        nx, nz = self.shape
        if nx * nz > 1_500_000:
            raise ValueError("网格超过 150 万节点；请增大网格间距。")
        if self.steps > 100_000:
            raise ValueError("时间步数超过 10 万；请缩短时长或增大网格间距。")
        if self.estimated_cache_bytes > self.cache_mb * 1024**2:
            raise ValueError("全采样测量面及至少两帧波场超过缓存预算。")

    @property
    def crossover_distance(self) -> float | None:
        if self.c2 <= self.c1:
            return None
        return 2 * self.interface_depth * math.sqrt((self.c2+self.c1)/(self.c2-self.c1))

    @property
    def x_span(self) -> float:
        return self.width - self.x_min

    @property
    def y_span(self) -> float:
        return self.depth - self.y_min

    @property
    def shape(self) -> tuple[int, int]:
        return math.ceil(self.x_span / self.spacing) + 1, math.ceil(self.y_span / self.spacing) + 1

    @property
    def dt_limit(self) -> float:
        nx, nz = self.shape
        dx, dz = self.x_span / (nx - 1), self.y_span / (nz - 1)
        return self.cfl / (max(self.c1, self.c2) * math.sqrt(dx**-2 + dz**-2))

    @property
    def steps(self) -> int:
        return math.ceil(self.duration / self.dt_limit)

    @property
    def receiver_bounds(self) -> tuple[float, float]:
        margin = self.sponge_width + self.spacing if self.boundary == "sponge" else self.spacing
        return self.x_min + margin, self.width - margin

    @property
    def frame_count(self) -> int:
        nx, nz = self.shape
        row_bytes = (self.steps + 1) * (nx * 4 + 8)
        available = self.cache_mb * 1024**2 - row_bytes
        return max(2, min(self.max_frames, self.steps + 1, int(available // (nx * nz * 4 + 8))))

    @property
    def estimated_cache_bytes(self) -> int:
        nx, nz = self.shape
        return (self.steps + 1) * (nx * 4 + 8) + self.frame_count * (nx * nz * 4 + 8)


@dataclass
class Receiver:
    id: int
    x: float

    @property
    def y(self) -> float:
        return 0.0


def recommended_right_boundary(config: SimulationConfig, receivers: list[Receiver]) -> float | None:
    """Engineering coverage rule, not a reflection-error bound or optimum array.

    Leave one upper-layer central wavelength (at least four cells) between the
    target/outermost existing receiver and the sponge. Round outwards for GUI.
    """
    dc = config.crossover_distance
    if dc is None:
        return None
    farthest = max([0.0] + [r.x for r in receivers])
    target = max(config.crossover_factor * dc, farthest)
    guard = max(config.c1/config.frequency, 4*config.spacing)
    sponge = config.sponge_width if config.boundary == "sponge" else 0.0
    raw = target + guard + sponge
    return math.ceil(round(math.ceil(raw/config.spacing)*config.spacing, 10)*100)/100


@dataclass
class SimulationResult:
    config: SimulationConfig
    x: np.ndarray
    z: np.ndarray
    times: np.ndarray
    frames: np.ndarray
    frame_times: np.ndarray
    plane: np.ndarray
    elapsed_seconds: float = 0.0

    @property
    def y(self) -> np.ndarray:
        """Depth-positive-down coordinate; z remains an internal compatibility alias."""
        return self.z

    def signal(self, position: float) -> np.ndarray:
        if not self.x[0] <= position <= self.x[-1]:
            raise ValueError("接收器位置越界。")
        index = np.clip(np.searchsorted(self.x, position) - 1, 0, len(self.x) - 2)
        weight = (position - self.x[index]) / (self.x[index + 1] - self.x[index])
        return (1 - weight) * self.plane[:, index] + weight * self.plane[:, index + 1]


class SimulationCancelled(Exception):
    pass


def pulse(t: float, frequency: float) -> float:
    """Compact, zero-mean two-cycle pulse, supported on [0, 2/f]."""
    if t <= 0 or t >= 2 / frequency:
        return 0.0
    return math.sin(2 * math.pi * frequency * t) * math.sin(math.pi * frequency * t / 2)**2


def laplacian(u: np.ndarray, dx: float, dz: float, boundary: str) -> np.ndarray:
    # Neumann ghost points mirror interior values. Sponge edges are overwritten
    # by the outgoing update; these ghost values only provide a provisional edge.
    out = np.zeros_like(u)
    out[:, 1:-1] += (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
    out[1:-1, :] += (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dz**2
    if boundary != "dirichlet":
        out[:, 0] += 2 * (u[:, 1] - u[:, 0]) / dx**2
        out[:, -1] += 2 * (u[:, -2] - u[:, -1]) / dx**2
        out[0, :] += 2 * (u[1, :] - u[0, :]) / dz**2
        out[-1, :] += 2 * (u[-2, :] - u[-1, :]) / dz**2
    return out


def damping(config: SimulationConfig, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    if config.boundary != "sponge":
        return np.zeros((len(z), len(x)))
    w = config.sponge_width
    left = np.maximum(0, (config.x_min + w - x) / w)**3
    right = np.maximum(0, (x - (config.width - w)) / w)**3
    bottom = np.maximum(0, (z - (config.depth - w)) / w)**3
    top = np.maximum(0, (config.y_min + w - z) / w)**3
    return config.sponge_strength * (left[None, :] + right[None, :] + top[:, None] + bottom[:, None])


def outgoing_edges(nxt: np.ndarray, current: np.ndarray, speeds: np.ndarray, dt: float, dx: float, dy: float) -> None:
    """First-order Mur outgoing condition, backed by the interior sponge.

    Discretize u_t +/- c u_n = 0 across the edge/interior pair at half time.
    This is an approximate local absorber, not a perfectly matched layer.
    Corners average the two one-way estimates, avoiding directional priority.
    """
    rx = (speeds * dt - dx) / (speeds * dt + dx)
    nxt[:, 0] = current[:, 1] + rx * (nxt[:, 1] - current[:, 0])
    nxt[:, -1] = current[:, -2] + rx * (nxt[:, -2] - current[:, -1])
    x_corners = nxt[np.ix_([0, -1], [0, -1])].copy()
    for edge, neighbor in ((0, 1), (-1, -2)):
        ry = (speeds[edge] * dt - dy) / (speeds[edge] * dt + dy)
        nxt[edge, :] = current[neighbor, :] + ry * (nxt[neighbor, :] - current[edge, :])
    nxt[np.ix_([0, -1], [0, -1])] = .5 * (x_corners + nxt[np.ix_([0, -1], [0, -1])])


def simulate(
    config: SimulationConfig,
    cancel: threading.Event | None = None,
    progress: Callable[[int, float, np.ndarray], None] | None = None,
    initial: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    source_enabled: bool = True,
    minimum_steps: int = 0,
) -> SimulationResult:
    import time
    config.validate()
    started = time.perf_counter()
    nx, nz = config.shape
    x, z = np.linspace(config.x_min, config.width, nx), np.linspace(config.y_min, config.depth, nz)
    dx, dz = x[1] - x[0], z[1] - z[0]
    if type(minimum_steps) is not int or not 0 <= minimum_steps <= 100_000:
        raise ValueError("最少时间步数须为 0–100000 的整数。")
    steps = max(config.steps, minimum_steps)
    available = config.cache_mb * 1024**2 - (steps+1)*(nx*4+8)
    affordable_frames = int(available // (nx*nz*4+8))
    if affordable_frames < 2:
        raise ValueError("指定时间采样超过缓存预算。")
    times = np.linspace(0, config.duration, steps + 1)
    dt = times[1]
    c2 = np.where(z[:, None] < config.interface_depth, config.c1**2, config.c2**2)
    speeds = np.sqrt(c2[:, 0])
    gamma_dt = damping(config, x, z) * dt
    xx, zz = np.meshgrid(x, z)
    sigma = 1.2 * max(dx, dz)
    radius2 = xx**2 + zz**2
    source = np.exp(-radius2 / (2 * sigma**2))
    source[radius2 > (4 * sigma)**2] = 0
    # Normalize spatial forcing integral: source strength does not change with h.
    source /= np.sum(source) * dx * dz
    u = np.zeros((nz, nx)) if initial is None else np.asarray(initial(xx, zz), dtype=float).copy()
    if u.shape != (nz, nx) or not np.isfinite(u).all():
        raise ValueError("初始波场形状或数值无效。")
    if config.boundary == "dirichlet":
        u[[0, -1], :] = 0
        u[:, [0, -1]] = 0
    # Zero initial velocity and Taylor-consistent startup.
    prev = u + 0.5 * dt**2 * c2 * laplacian(u, dx, dz, config.boundary)
    if config.boundary == "dirichlet":
        prev[[0, -1], :] = 0
        prev[:, [0, -1]] = 0
    sample_indices = np.unique(np.linspace(0, steps, min(config.frame_count, affordable_frames), dtype=int))
    frames = np.empty((len(sample_indices), nz, nx), dtype=np.float32)
    plane = np.empty((steps + 1, nx), dtype=np.float32)
    zi = int(np.clip(np.searchsorted(z, 0.0) - 1, 0, nz - 2))
    zw = -z[zi] / dz
    frame_i = 0
    for n in range(steps + 1):
        if cancel is not None and cancel.is_set():
            raise SimulationCancelled()
        plane[n] = (1 - zw) * u[zi] + zw * u[zi + 1]
        if frame_i < len(sample_indices) and n == sample_indices[frame_i]:
            frames[frame_i] = u
            frame_i += 1
        if progress is not None and (n % max(1, steps // 80) == 0 or n == steps):
            progress(round(100 * n / steps), times[n], u.astype(np.float32))
        if n == steps:
            break
        forcing = source * pulse(times[n], config.frequency) if source_enabled else 0.0
        nxt = (2 * u - (1 - gamma_dt) * prev + dt**2 * (c2 * laplacian(u, dx, dz, config.boundary) + forcing)) / (1 + gamma_dt)
        if config.boundary == "dirichlet":
            nxt[[0, -1], :] = 0
            nxt[:, [0, -1]] = 0
        elif config.boundary == "sponge":
            outgoing_edges(nxt, u, speeds, dt, dx, dz)
        if n % 100 == 0 and not np.isfinite(nxt).all():
            raise FloatingPointError("波场出现非有限数值；请检查参数。")
        prev, u = u, nxt
    return SimulationResult(config, x, z, times, frames, times[sample_indices], plane, time.perf_counter() - started)


def pick_arrival(times: np.ndarray, signal: np.ndarray, threshold: float = 1e-4) -> float | None:
    """First fixed absolute-amplitude crossing. No future peaks or ray oracle."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("拾取阈值必须是大于零的有限幅值。")
    if len(times) != len(signal) or len(times) == 0:
        raise ValueError("时间与信号数组必须非空且等长。")
    amplitude = np.abs(signal)
    hits = np.flatnonzero(amplitude >= threshold)
    if not len(hits):
        return None
    i = int(hits[0])
    if i == 0:
        return float(times[0])
    weight = (threshold - amplitude[i - 1]) / (amplitude[i] - amplitude[i - 1])
    return float(times[i - 1] + weight * (times[i] - times[i - 1]))


def ray_front(config: SimulationConfig, receiver_x: float) -> float:
    d = abs(receiver_x - config.source_x)
    direct = d / config.c1
    if config.c2 <= config.c1:
        return direct
    h = config.interface_depth - config.source_depth
    critical = 2 * h * config.c1 / math.sqrt(config.c2**2 - config.c1**2)
    head = d / config.c2 + 2 * h / config.c1 * math.sqrt(1 - (config.c1 / config.c2)**2)
    return min(direct, head) if d >= critical else direct


def save_configuration(path: str | Path, config: SimulationConfig, receivers: list[Receiver], threshold: float = 1e-4, noise: NoiseConfig = NoiseConfig()) -> None:
    config.validate()
    noise.validate()
    payload = {"schema_version": 2, "coordinate_system": "x_y_depth_positive_down", "simulation": asdict(config),
               "receivers": [{"id": r.id, "x": r.x, "y": r.y} for r in receivers], "arrival_threshold_au": threshold,
               "noise": asdict(noise)}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_configuration(path: str | Path, *, include_noise: bool = False) -> tuple:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2:
        raise ValueError("此版本需要 v2 配置（原点波源、y=0 测点、四侧吸收、绝对阈值）。旧 v1 配置请保留作历史记录，使用 scenes 中的新示例。")
    config = SimulationConfig(**raw["simulation"])
    config.validate()
    if any(r.get("y") != 0 for r in raw["receivers"]):
        raise ValueError("所有接收器的 y 坐标必须为 0。")
    receivers = [Receiver(r["id"], r["x"]) for r in raw["receivers"]]
    lo, hi = config.receiver_bounds
    if not receivers or len({r.id for r in receivers}) != len(receivers):
        raise ValueError("至少需要 1 个接收器，且编号不可重复。")
    if any(not isinstance(r.id, int) or r.id <= 0 or not math.isfinite(r.x) or not lo <= r.x <= hi for r in receivers):
        raise ValueError("接收器编号或位置无效。")
    threshold = float(raw.get("arrival_threshold_au", 1e-4))
    if not math.isfinite(threshold) or not 1e-8 <= threshold <= 1:
        raise ValueError("绝对阈值须在 1e-8–1 a.u. 内。")
    noise = NoiseConfig(**raw.get("noise", {}))
    noise.validate()
    if include_noise:
        return config, receivers, threshold, noise
    return config, receivers, threshold


def export_results(directory: str | Path, result: SimulationResult, receivers: list[Receiver], threshold: float, noise: NoiseConfig = NoiseConfig()) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_configuration(directory / "configuration.json", result.config, receivers, threshold, noise)
    clean_signals = [result.signal(r.x) for r in receivers]
    signals = [add_measurement_noise(signal, r.id, noise) for r, signal in zip(receivers, clean_signals)]
    for filename, traces in (("signals.csv", signals), ("signals_clean.csv", clean_signals)):
        with (directory / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + [f"R{r.id}_pressure_au" for r in receivers])
            writer.writerows(zip(result.times, *traces))
    with (directory / "receivers.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["receiver_id", "x_m", "y_m", "offset_m", "picked_arrival_s", "ideal_ray_front_s", "arrival_threshold_au"])
        for r, signal in zip(receivers, signals):
            arrival = pick_arrival(result.times, signal, threshold)
            writer.writerow([r.id, r.x, result.config.source_depth, abs(r.x - result.config.source_x), "" if arrival is None else arrival, ray_front(result.config, r.x), threshold])
    np.savez_compressed(directory / "wavefield.npz", x=result.x, y=result.y, frame_times=result.frame_times, frames=result.frames, times=result.times, observation_plane=result.plane)
    (directory / "README.txt").write_text(
        "Wave Lab 2D export\nCoordinates: m; time: s; speeds: m/s; pressure: arbitrary units.\n"
        "Source fixed at (0,0). All receivers stationary at y=0. Positive y points down.\n"
        "signals.csv uses every solver timestep; wavefield frames are decimated.\n"
        "signals.csv contains the displayed observations, with noise when enabled.\n"
        "signals_clean.csv and wavefield.npz always contain clean simulation data.\n"
        "Noise: independent zero-mean Gaussian samples, absolute std_au per sample.\n"
        "Generator: NumPy PCG64, SeedSequence([seed, receiver_id]), standard_normal.\n"
        "configuration.json saves enabled/std_au/seed; moving a channel retains its noise.\n"
        "Noise is receiver measurement noise, not a force in the wave equation.\n"
        "Picks use signals.csv; noise alone may cross the threshold before any wave.\n"
        "Picker: first |p| >= arrival_threshold_au, linear time interpolation.\n"
        "No future peak or predicted arrival is used. No pick means threshold not reached.\n"
        "Ideal ray front is a geometric reference, without finite-pulse/threshold/grid delay.\n"
        "Four-sided sponge plus first-order Mur outgoing edges; approximate, not PML.\n"
        "No reflecting surface at y=0.\n"
        "The y<0 buffer extends the upper medium; it does not represent air.\n"
        "configuration.json records the actual selected boundary.\n", encoding="utf-8")
