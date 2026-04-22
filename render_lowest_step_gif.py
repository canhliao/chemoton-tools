from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle
import numpy as np
from tqdm.auto import tqdm

from chemoton_accessibility_core import (
    DatabaseConfig,
    DatabaseManager,
    Model,
    ModelConfig,
    ProgressReporter,
    _database_config_from_manager,
    _model_config_from_model,
    _model_from_config,
)
from render_reaction_common import (
    RequestedReaction,
    SkippedReaction,
    TrajectoryFrame,
    bond_pairs,
    collect_requested_reactions,
    color_for_element,
    element_symbol,
    sample_step_frames,
    select_lowest_barrier_step_for_direction,
    write_vmd_script,
    write_xyz_trajectory,
)
from user_input_config import (
    ACCESSIBILITY_DEFAULTS,
    DATABASE_DEFAULTS,
    GIF_RENDER_DEFAULTS,
    MODEL_DEFAULTS,
)


ATOM_RADII = {
    "H": 0.19,
    "C": 0.32,
    "N": 0.31,
    "O": 0.31,
    "S": 0.39,
}


@dataclass(frozen=True)
class GifRenderResult:
    requested: RequestedReaction
    skipped_reason: str | None
    gif_path: str | None
    xyz_path: str | None
    vmd_path: str | None
    step_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all listed directional reactions to GIFs using the lowest-barrier elementary step for each direction."
        )
    )
    parser.add_argument(
        "reactions_file",
        nargs="?",
        help="Optional text file containing one directional reaction id per line, e.g. 69c50ae865f50a833301e4d5;1;",
    )
    parser.add_argument(
        "--reaction-id",
        action="append",
        dest="reaction_ids",
        default=None,
        help="Reaction id to render. Accepts either reaction_id;0;/reaction_id;1; or bare reaction_id (renders both directions). Repeatable.",
    )
    parser.add_argument("--db-name", default=DATABASE_DEFAULTS["db_name"])
    parser.add_argument("--ip", default=DATABASE_DEFAULTS["ip"])
    parser.add_argument("--port", type=int, default=DATABASE_DEFAULTS["port"])
    parser.add_argument("--energy-type", default=ACCESSIBILITY_DEFAULTS["energy_type"])
    parser.add_argument("--method-family", default=MODEL_DEFAULTS["method_family"])
    parser.add_argument("--method", default=MODEL_DEFAULTS["method"])
    parser.add_argument("--basisset", default=MODEL_DEFAULTS["basisset"])
    parser.add_argument("--spin-mode", default=MODEL_DEFAULTS["spin_mode"])
    parser.add_argument("--program", default=MODEL_DEFAULTS["program"])
    parser.add_argument("--frames", type=int, default=GIF_RENDER_DEFAULTS["frames"], help="Number of frames to sample from a spline.")
    parser.add_argument("--fps", type=int, default=GIF_RENDER_DEFAULTS["fps"])
    parser.add_argument(
        "--render-dim",
        choices=["2d", "3d"],
        default=GIF_RENDER_DEFAULTS["render_dim"],
        help="Render as a 2D projection or a 3D view. Default: 3d.",
    )
    parser.add_argument("--view-elev", type=float, default=GIF_RENDER_DEFAULTS["view_elev"], help="Initial elevation angle for 3D rendering.")
    parser.add_argument("--view-azim", type=float, default=GIF_RENDER_DEFAULTS["view_azim"], help="Initial azimuth angle for 3D rendering.")
    parser.add_argument(
        "--rotate-azim-deg",
        type=float,
        default=GIF_RENDER_DEFAULTS["rotate_azim_deg"],
        help="Total azimuth rotation to apply across the GIF frames.",
    )
    parser.add_argument(
        "--rotate-elev-deg",
        type=float,
        default=GIF_RENDER_DEFAULTS["rotate_elev_deg"],
        help="Total elevation rotation to apply across the GIF frames.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars while rendering the requested reactions.",
    )
    parser.add_argument(
        "--render-quality",
        choices=["high", "low"],
        default=GIF_RENDER_DEFAULTS["render_quality"],
        help="Rendering style quality. 'low' restores the earlier simple marker/line look.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes to use across requested reactions. Default: 1.",
    )
    parser.add_argument("--output-dir", default=GIF_RENDER_DEFAULTS["output_dir"])
    return parser.parse_args()


def atom_radius(symbol: str) -> float:
    return ATOM_RADII.get(symbol, 0.25)


def glossy_facecolors(base_color: str, normals: np.ndarray) -> np.ndarray:
    base_rgb = np.array(mcolors.to_rgb(base_color))
    light_dir = np.array([0.35, -0.45, 0.82])
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(np.tensordot(normals, light_dir, axes=([2], [0])), 0.0, 1.0)
    specular = np.power(np.clip(diffuse, 0.0, 1.0), 14)
    shading = 0.52 + 0.38 * diffuse + 0.34 * specular
    colors = np.clip(base_rgb[None, None, :] * shading[..., None], 0.0, 1.0)
    alpha = np.ones((*colors.shape[:2], 1))
    return np.concatenate([colors, alpha], axis=2)


def draw_sphere(ax, center: np.ndarray, radius: float, color: str) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 20)
    cos_u, sin_u = np.cos(u), np.sin(u)
    sin_v, cos_v = np.sin(v), np.cos(v)
    x = center[0] + radius * np.outer(cos_u, sin_v)
    y = center[1] + radius * np.outer(sin_u, sin_v)
    z = center[2] + radius * np.outer(np.ones_like(u), cos_v)
    normals = np.stack(
        [
            np.outer(cos_u, sin_v),
            np.outer(sin_u, sin_v),
            np.outer(np.ones_like(u), cos_v),
        ],
        axis=2,
    )
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=glossy_facecolors(color, normals),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=3,
    )


def cylinder_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    normal_a = np.cross(direction, trial)
    normal_a = normal_a / np.linalg.norm(normal_a)
    normal_b = np.cross(direction, normal_a)
    return normal_a, normal_b


def draw_cylinder(ax, start: np.ndarray, end: np.ndarray, radius: float, color: str) -> None:
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return
    direction = axis / length
    normal_a, normal_b = cylinder_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    t = np.linspace(0.0, length, 2)
    theta_grid, t_grid = np.meshgrid(theta, t, indexing="ij")
    circle = (
        radius * np.cos(theta_grid)[..., None] * normal_a
        + radius * np.sin(theta_grid)[..., None] * normal_b
    )
    points = start + t_grid[..., None] * direction + circle
    normals = (
        np.cos(theta_grid)[..., None] * normal_a
        + np.sin(theta_grid)[..., None] * normal_b
    )
    ax.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        facecolors=glossy_facecolors(color, normals),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=2,
    )

def render_gif(
    frames: list[TrajectoryFrame],
    output_path: Path,
    fps: int,
    render_dim: str,
    render_quality: str,
    view_elev: float,
    view_azim: float,
    rotate_azim_deg: float,
    rotate_elev_deg: float,
    show_progress: bool = False,
) -> None:
    all_positions = np.concatenate([frame.atoms.positions for frame in frames], axis=0)
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.6 * span if span > 0 else 1.0

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d" if render_dim == "3d" else None)
    progress_bar_x = 0.18
    progress_bar_y = 0.06
    progress_bar_width = 0.64
    progress_bar_height = 0.024
    progress_background = Rectangle(
        (progress_bar_x, progress_bar_y),
        progress_bar_width,
        progress_bar_height,
        transform=fig.transFigure,
        facecolor=(1.0, 1.0, 1.0, 0.7),
        edgecolor="#5a5a5a",
        linewidth=0.9,
    )
    progress_fill = Rectangle(
        (progress_bar_x, progress_bar_y),
        0.0,
        progress_bar_height,
        transform=fig.transFigure,
        facecolor="#2f7ed8",
        edgecolor="none",
    )
    fig.add_artist(progress_background)
    fig.add_artist(progress_fill)
    progress_label = fig.text(
        0.5,
        progress_bar_y + progress_bar_height + 0.01,
        "",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#202020",
    )
    adjusted_fps = max(1, round(fps * (8.0 / 9.0)))
    writer = PillowWriter(fps=adjusted_fps)
    writer.setup(fig, str(output_path), dpi=120)
    try:
        denom = max(1, len(frames) - 1)
        frame_iterable = enumerate(frames)
        if show_progress:
            frame_iterable = tqdm(
                frame_iterable,
                total=len(frames),
                desc=f"GIF frames {output_path.name}",
                unit="frame",
                leave=False,
            )
        for index, frame in frame_iterable:
            ax.cla()
            atoms = frame.atoms
            positions = atoms.positions
            symbols = [element_symbol(element) for element in atoms.elements]
            colors = [color_for_element(symbol) for symbol in symbols]
            sizes = [180 if symbol != "H" else 90 for symbol in symbols]

            for i, j in bond_pairs(atoms):
                xyz = positions[[i, j], :]
                if render_dim == "3d" and render_quality == "high":
                    start = xyz[0]
                    end = xyz[1]
                    axis = end - start
                    axis_length = np.linalg.norm(axis)
                    if axis_length > 1e-8:
                        direction = axis / axis_length
                        start = start + direction * atom_radius(symbols[i]) * 0.92
                        end = end - direction * atom_radius(symbols[j]) * 0.92
                    draw_cylinder(ax, start, end, radius=0.065, color="#a6a6a6")
                else:
                    if render_dim == "3d":
                        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#808080", linewidth=2.0, zorder=1)
                    else:
                        ax.plot(xyz[:, 0], xyz[:, 1], color="#808080", linewidth=2.6, zorder=1)

            if render_dim == "3d" and render_quality == "high":
                for position, symbol, color in zip(positions, symbols, colors):
                    draw_sphere(ax, position, atom_radius(symbol), color)
                ax.set_xlim(center[0] - half, center[0] + half)
                ax.set_ylim(center[1] - half, center[1] + half)
                ax.set_zlim(center[2] - half, center[2] + half)
                ax.set_box_aspect((1, 1, 1))
                elev = view_elev + rotate_elev_deg * (index / denom)
                azim = view_azim + rotate_azim_deg * (index / denom)
                ax.view_init(elev=elev, azim=azim)
            else:
                if render_dim == "3d":
                    ax.scatter(
                        positions[:, 0],
                        positions[:, 1],
                        positions[:, 2],
                        c=colors,
                        s=sizes,
                        edgecolors="black",
                        linewidths=0.5,
                        depthshade=False,
                        zorder=2,
                    )
                    ax.set_xlim(center[0] - half, center[0] + half)
                    ax.set_ylim(center[1] - half, center[1] + half)
                    ax.set_zlim(center[2] - half, center[2] + half)
                    ax.set_box_aspect((1, 1, 1))
                    elev = view_elev + rotate_elev_deg * (index / denom)
                    azim = view_azim + rotate_azim_deg * (index / denom)
                    ax.view_init(elev=elev, azim=azim)
                else:
                    ax.scatter(
                        positions[:, 0],
                        positions[:, 1],
                        c=colors,
                        s=sizes,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=2,
                    )
                    ax.set_xlim(center[0] - half, center[0] + half)
                    ax.set_ylim(center[1] - half, center[1] + half)
                    ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            if frame.delta_energy_kj_per_mol is not None:
                ax.set_title(f"Delta E = {frame.delta_energy_kj_per_mol:.1f} kJ/mol", pad=12)
            fraction = (index + 1) / max(1, len(frames))
            progress_fill.set_width(progress_bar_width * fraction)
            progress_label.set_text(f"Reaction progress: {fraction * 100:.0f}%")
            writer.grab_frame()
    finally:
        writer.finish()
        plt.close(fig)


def render_requested_reaction(
    requested: RequestedReaction,
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
    frame_count: int,
    fps: int,
    render_dim: str,
    render_quality: str,
    view_elev: float,
    view_azim: float,
    rotate_azim_deg: float,
    rotate_elev_deg: float,
    output_dir: Path,
    show_progress: bool = False,
) -> tuple[Path, Path, Path, str]:
    step = select_lowest_barrier_step_for_direction(requested, manager, model, energy_type)
    frames = sample_step_frames(
        step,
        manager,
        frame_count,
        model,
        energy_type,
        requested.reaction_id,
        requested.direction,
    )

    stem = f"{requested.reaction_id}_{requested.direction}"
    xyz_path = output_dir / f"{stem}.xyz"
    vmd_path = output_dir / f"{stem}.vmd.tcl"
    gif_path = output_dir / f"{stem}_{render_dim}_{render_quality}.gif"

    write_xyz_trajectory(frames, xyz_path)
    write_vmd_script(xyz_path, vmd_path)
    render_gif(
        frames,
        gif_path,
        fps,
        render_dim,
        render_quality,
        view_elev,
        view_azim,
        rotate_azim_deg,
        rotate_elev_deg,
        show_progress,
    )
    return gif_path, xyz_path, vmd_path, step.get_id().string()


def _render_requested_reaction_worker(
    requested: RequestedReaction,
    database_config: DatabaseConfig,
    model_config: ModelConfig,
    energy_type: str,
    frame_count: int,
    fps: int,
    render_dim: str,
    render_quality: str,
    view_elev: float,
    view_azim: float,
    rotate_azim_deg: float,
    rotate_elev_deg: float,
    output_dir: str,
) -> GifRenderResult:
    try:
        manager = DatabaseManager(database_config.db_name, database_config.ip, database_config.port)
        manager.loadCollections()
        model = _model_from_config(model_config)
        gif_path, xyz_path, vmd_path, step_id = render_requested_reaction(
            requested=requested,
            manager=manager,
            model=model,
            energy_type=energy_type,
            frame_count=frame_count,
            fps=fps,
            render_dim=render_dim,
            render_quality=render_quality,
            view_elev=view_elev,
            view_azim=view_azim,
            rotate_azim_deg=rotate_azim_deg,
            rotate_elev_deg=rotate_elev_deg,
            output_dir=Path(output_dir),
            show_progress=False,
        )
        return GifRenderResult(
            requested=requested,
            skipped_reason=None,
            gif_path=str(gif_path),
            xyz_path=str(xyz_path),
            vmd_path=str(vmd_path),
            step_id=step_id,
        )
    except SkippedReaction as exc:
        return GifRenderResult(
            requested=requested,
            skipped_reason=str(exc),
            gif_path=None,
            xyz_path=None,
            vmd_path=None,
            step_id=None,
        )


def main() -> None:
    args = parse_args()
    jobs = max(1, args.jobs)
    requested_reactions = collect_requested_reactions(args.reactions_file, args.reaction_ids)
    if not requested_reactions:
        raise RuntimeError("No reaction IDs provided. Pass a reactions file and/or one or more --reaction-id flags.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = DatabaseManager(args.db_name, args.ip, args.port)
    manager.loadCollections()
    model = Model(args.method_family, args.method, args.basisset, args.spin_mode, args.program)
    progress = ProgressReporter(args.progress)
    database_config = _database_config_from_manager(manager)
    model_config = _model_config_from_model(model)

    if jobs == 1:
        results: list[GifRenderResult] = []
        for requested in progress.wrap(
            requested_reactions,
            total=len(requested_reactions),
            desc="Rendering GIF reactions",
        ):
            try:
                gif_path, xyz_path, vmd_path, step_id = render_requested_reaction(
                    requested=requested,
                    manager=manager,
                    model=model,
                    energy_type=args.energy_type,
                    frame_count=args.frames,
                    fps=args.fps,
                    render_dim=args.render_dim,
                    render_quality=args.render_quality,
                    view_elev=args.view_elev,
                    view_azim=args.view_azim,
                    rotate_azim_deg=args.rotate_azim_deg,
                    rotate_elev_deg=args.rotate_elev_deg,
                    output_dir=output_dir,
                    show_progress=args.progress,
                )
                results.append(
                    GifRenderResult(
                        requested=requested,
                        skipped_reason=None,
                        gif_path=str(gif_path),
                        xyz_path=str(xyz_path),
                        vmd_path=str(vmd_path),
                        step_id=step_id,
                    )
                )
            except SkippedReaction as exc:
                results.append(
                    GifRenderResult(
                        requested=requested,
                        skipped_reason=str(exc),
                        gif_path=None,
                        xyz_path=None,
                        vmd_path=None,
                        step_id=None,
                    )
                )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(
                progress.wrap(
                    executor.map(
                        _render_requested_reaction_worker,
                        requested_reactions,
                        [database_config] * len(requested_reactions),
                        [model_config] * len(requested_reactions),
                        [args.energy_type] * len(requested_reactions),
                        [args.frames] * len(requested_reactions),
                        [args.fps] * len(requested_reactions),
                        [args.render_dim] * len(requested_reactions),
                        [args.render_quality] * len(requested_reactions),
                        [args.view_elev] * len(requested_reactions),
                        [args.view_azim] * len(requested_reactions),
                        [args.rotate_azim_deg] * len(requested_reactions),
                        [args.rotate_elev_deg] * len(requested_reactions),
                        [str(output_dir)] * len(requested_reactions),
                    ),
                    total=len(requested_reactions),
                    desc="Rendering GIF reactions",
                )
            )

    for result in results:
        if result.skipped_reason is not None:
            print(result.skipped_reason)
            continue
        print(f"Rendered {result.requested.reaction_id};{result.requested.direction};")
        print(f"  elementary step: {result.step_id}")
        print(f"  gif: {result.gif_path}")
        print(f"  xyz: {result.xyz_path}")
        print(f"  vmd script: {result.vmd_path}")


if __name__ == "__main__":
    main()
