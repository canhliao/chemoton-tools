from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import colors as mcolors
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
    SkippedReaction,
    bond_pairs,
    collect_requested_reactions,
    color_for_element,
    element_symbol,
    sample_step_frames,
    select_lowest_barrier_step_for_direction,
)
from user_input_config import (
    ACCESSIBILITY_DEFAULTS,
    DATABASE_DEFAULTS,
    INTERACTIVE_RENDER_DEFAULTS,
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
class InteractiveRenderResult:
    reaction_token: str
    skipped_reason: str | None
    html_path: str | None
    step_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render requested directional reactions as standalone interactive 3D HTML trajectory viewers."
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
    parser.add_argument("--frames", type=int, default=INTERACTIVE_RENDER_DEFAULTS["frames"], help="Number of frames to sample from a spline.")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars while rendering the requested reactions.",
    )
    parser.add_argument(
        "--render-quality",
        choices=["high", "low"],
        default=INTERACTIVE_RENDER_DEFAULTS["render_quality"],
        help="Rendering style quality. 'low' uses simple markers and lines.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes to use across requested reactions. Default: 1.",
    )
    parser.add_argument("--output-dir", default=INTERACTIVE_RENDER_DEFAULTS["output_dir"])
    return parser.parse_args()


def atom_radius(symbol: str) -> float:
    return ATOM_RADII.get(symbol, 0.25)


def atom_trace_low(atoms) -> go.Scatter3d:
    positions = atoms.positions
    symbols = [element_symbol(element) for element in atoms.elements]
    colors = [color_for_element(symbol) for symbol in symbols]
    sizes = [22 if symbol != "H" else 12 for symbol in symbols]
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker={"size": sizes, "color": colors, "line": {"color": "black", "width": 1}},
        text=symbols,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def bond_trace_low(atoms) -> go.Scatter3d:
    positions = atoms.positions
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for i, j in bond_pairs(atoms):
        x.extend([positions[i, 0], positions[j, 0], None])
        y.extend([positions[i, 1], positions[j, 1], None])
        z.extend([positions[i, 2], positions[j, 2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": "#808080", "width": 6},
        hoverinfo="skip",
        showlegend=False,
    )


def sphere_mesh(center: np.ndarray, radius: float, color: str) -> go.Mesh3d:
    u_count = 18
    v_count = 12
    theta = np.linspace(0.0, 2.0 * np.pi, u_count, endpoint=False)
    phi = np.linspace(0.0, np.pi, v_count)
    vertices: list[np.ndarray] = []
    for phi_value in phi:
        sin_phi = np.sin(phi_value)
        cos_phi = np.cos(phi_value)
        for theta_value in theta:
            vertices.append(
                center
                + radius
                * np.array(
                    [
                        np.cos(theta_value) * sin_phi,
                        np.sin(theta_value) * sin_phi,
                        cos_phi,
                    ]
                )
            )
    vertices_array = np.asarray(vertices)
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    for v_index in range(v_count - 1):
        ring_start = v_index * u_count
        next_ring_start = (v_index + 1) * u_count
        for u_index in range(u_count):
            current = ring_start + u_index
            nxt = ring_start + (u_index + 1) % u_count
            below = next_ring_start + u_index
            below_next = next_ring_start + (u_index + 1) % u_count
            i_idx.extend([current, nxt])
            j_idx.extend([below, below])
            k_idx.extend([nxt, below_next])
    return go.Mesh3d(
        x=vertices_array[:, 0],
        y=vertices_array[:, 1],
        z=vertices_array[:, 2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        flatshading=False,
        hoverinfo="skip",
        showscale=False,
        lighting={"ambient": 0.35, "diffuse": 0.7, "specular": 1.0, "roughness": 0.15, "fresnel": 0.1},
        lightposition={"x": 120, "y": -160, "z": 240},
        opacity=1.0,
    )


def cylinder_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    normal_a = np.cross(direction, trial)
    normal_a = normal_a / np.linalg.norm(normal_a)
    normal_b = np.cross(direction, normal_a)
    return normal_a, normal_b


def cylinder_mesh(start: np.ndarray, end: np.ndarray, radius: float, color: str) -> go.Mesh3d | None:
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return None
    direction = axis / length
    normal_a, normal_b = cylinder_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    vertices: list[np.ndarray] = []
    for point in [start, end]:
        for theta_value in theta:
            vertices.append(
                point
                + radius * np.cos(theta_value) * normal_a
                + radius * np.sin(theta_value) * normal_b
            )
    vertices_array = np.asarray(vertices)
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    ring_size = len(theta)
    for idx in range(ring_size):
        nxt = (idx + 1) % ring_size
        top_a = idx
        top_b = nxt
        bottom_a = ring_size + idx
        bottom_b = ring_size + nxt
        i_idx.extend([top_a, top_b])
        j_idx.extend([bottom_a, bottom_a])
        k_idx.extend([top_b, bottom_b])
    return go.Mesh3d(
        x=vertices_array[:, 0],
        y=vertices_array[:, 1],
        z=vertices_array[:, 2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        flatshading=False,
        hoverinfo="skip",
        showscale=False,
        lighting={"ambient": 0.28, "diffuse": 0.75, "specular": 0.9, "roughness": 0.2, "fresnel": 0.05},
        lightposition={"x": 120, "y": -160, "z": 240},
        opacity=1.0,
    )


def quality_traces(atoms, render_quality: str) -> list[go.BaseTraceType]:
    if render_quality == "low":
        return [bond_trace_low(atoms), atom_trace_low(atoms)]

    positions = atoms.positions
    symbols = [element_symbol(element) for element in atoms.elements]
    colors = [color_for_element(symbol) for symbol in symbols]
    traces: list[go.BaseTraceType] = []
    bond_color = mcolors.to_hex((0.65, 0.65, 0.65))
    for i, j in bond_pairs(atoms):
        start = positions[i]
        end = positions[j]
        axis = end - start
        axis_length = np.linalg.norm(axis)
        if axis_length > 1e-8:
            direction = axis / axis_length
            start = start + direction * atom_radius(symbols[i]) * 0.92
            end = end - direction * atom_radius(symbols[j]) * 0.92
        trace = cylinder_mesh(start, end, radius=0.065, color=bond_color)
        if trace is not None:
            traces.append(trace)
    for position, symbol, color in zip(positions, symbols, colors):
        traces.append(sphere_mesh(position, atom_radius(symbol), color))
    return traces


def scene_bounds(frames) -> tuple[list[float], list[float], list[float]]:
    all_positions = np.concatenate([frame.atoms.positions for frame in frames], axis=0)
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.6 * span if span > 0 else 1.0
    return (
        [center[0] - half, center[0] + half],
        [center[1] - half, center[1] + half],
        [center[2] - half, center[2] + half],
    )


def frame_title(reaction_token: str, index: int, delta_energy_kj_per_mol: float | None) -> str:
    title = f"{reaction_token} | frame {index + 1}"
    if delta_energy_kj_per_mol is not None:
        title += f" | Delta E = {delta_energy_kj_per_mol:.1f} kJ/mol"
    return title


def sample_requested_frames(
    requested,
    step,
    manager: DatabaseManager,
    frame_count: int,
    model: Model,
    energy_type: str,
):
    return sample_step_frames(
        step,
        manager,
        frame_count,
        model,
        energy_type,
        requested.reaction_id,
        requested.direction,
    )


def render_interactive_html(
    frames,
    reaction_token: str,
    output_path: Path,
    render_quality: str,
    show_progress: bool = False,
) -> None:
    x_range, y_range, z_range = scene_bounds(frames)
    plotly_frames = []
    slider_steps = []

    frame_iterable = enumerate(frames)
    if show_progress:
        frame_iterable = tqdm(frame_iterable, total=len(frames), desc=f"HTML frames {output_path.name}", unit="frame", leave=False)
    for index, frame in frame_iterable:
        traces = quality_traces(frame.atoms, render_quality)
        plotly_frames.append(
            go.Frame(
                name=str(index),
                data=traces,
                layout={"title": {"text": frame_title(reaction_token, index, frame.delta_energy_kj_per_mol)}},
            )
        )
        slider_steps.append(
            {
                "label": str(index + 1),
                "method": "animate",
                "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            }
        )

    fig = go.Figure(
        data=quality_traces(frames[0].atoms, render_quality),
        frames=plotly_frames,
    )
    fig.update_layout(
        title={"text": frame_title(reaction_token, 0, frames[0].delta_energy_kj_per_mol)},
        showlegend=False,
        scene={
            "xaxis": {"visible": False, "range": x_range},
            "yaxis": {"visible": False, "range": y_range},
            "zaxis": {"visible": False, "range": z_range},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 120, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "pad": {"r": 10, "t": 10},
                "x": 0.1,
                "y": 0.0,
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 40},
                "steps": slider_steps,
                "x": 0.1,
                "y": 0.0,
                "len": 0.8,
            }
        ],
    )
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def _render_interactive_worker(
    requested,
    database_config: DatabaseConfig,
    model_config: ModelConfig,
    energy_type: str,
    frames: int,
    render_quality: str,
    output_dir: str,
) -> InteractiveRenderResult:
    reaction_token = f"{requested.reaction_id};{requested.direction};"
    try:
        manager = DatabaseManager(database_config.db_name, database_config.ip, database_config.port)
        manager.loadCollections()
        model = _model_from_config(model_config)
        step = select_lowest_barrier_step_for_direction(requested, manager, model, energy_type)
        sampled_frames = sample_requested_frames(
            requested, step, manager, frames, model, energy_type
        )
        output_path = Path(output_dir) / f"{requested.reaction_id}_{requested.direction}_{render_quality}.html"
        render_interactive_html(sampled_frames, reaction_token, output_path, render_quality, False)
        return InteractiveRenderResult(
            reaction_token=reaction_token,
            skipped_reason=None,
            html_path=str(output_path),
            step_id=step.get_id().string(),
        )
    except SkippedReaction as exc:
        return InteractiveRenderResult(
            reaction_token=reaction_token,
            skipped_reason=str(exc),
            html_path=None,
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
        results: list[InteractiveRenderResult] = []
        for requested in progress.wrap(
            requested_reactions,
            total=len(requested_reactions),
            desc="Rendering interactive reactions",
        ):
            reaction_token = f"{requested.reaction_id};{requested.direction};"
            try:
                step = select_lowest_barrier_step_for_direction(requested, manager, model, args.energy_type)
                frames = sample_requested_frames(
                    requested, step, manager, args.frames, model, args.energy_type
                )
                output_path = output_dir / f"{requested.reaction_id}_{requested.direction}_{args.render_quality}.html"
                render_interactive_html(frames, reaction_token, output_path, args.render_quality, args.progress)
                results.append(
                    InteractiveRenderResult(
                        reaction_token=reaction_token,
                        skipped_reason=None,
                        html_path=str(output_path),
                        step_id=step.get_id().string(),
                    )
                )
            except SkippedReaction as exc:
                results.append(
                    InteractiveRenderResult(
                        reaction_token=reaction_token,
                        skipped_reason=str(exc),
                        html_path=None,
                        step_id=None,
                    )
                )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(
                progress.wrap(
                    executor.map(
                        _render_interactive_worker,
                        requested_reactions,
                        [database_config] * len(requested_reactions),
                        [model_config] * len(requested_reactions),
                        [args.energy_type] * len(requested_reactions),
                        [args.frames] * len(requested_reactions),
                        [args.render_quality] * len(requested_reactions),
                        [str(output_dir)] * len(requested_reactions),
                    ),
                    total=len(requested_reactions),
                    desc="Rendering interactive reactions",
                )
            )

    for result in results:
        if result.skipped_reason is not None:
            print(result.skipped_reason)
            continue
        print(f"Rendered {result.reaction_token}")
        print(f"  elementary step: {result.step_id}")
        print(f"  html: {result.html_path}")


if __name__ == "__main__":
    main()
