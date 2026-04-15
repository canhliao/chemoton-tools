from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from chemoton_accessibility_core import DEFAULT_CONFIG, DatabaseManager, Model
from render_reaction_common import (
    bond_pairs,
    collect_requested_reactions,
    color_for_element,
    element_symbol,
    sample_step_frames,
    select_lowest_barrier_step_for_direction,
)


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
    parser.add_argument("--db-name", default=DEFAULT_CONFIG["db_name"])
    parser.add_argument("--ip", default=DEFAULT_CONFIG["ip"])
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"])
    parser.add_argument("--energy-type", default=DEFAULT_CONFIG["energy_type"])
    parser.add_argument("--method-family", default="dft")
    parser.add_argument("--method", default="m062x")
    parser.add_argument("--basisset", default="6-311+G**")
    parser.add_argument("--spin-mode", default="unrestricted")
    parser.add_argument("--program", default="orca")
    parser.add_argument("--frames", type=int, default=48, help="Number of frames to sample from a spline.")
    parser.add_argument("--output-dir", default="interactive_reactions")
    return parser.parse_args()


def atom_trace(atoms) -> go.Scatter3d:
    positions = atoms.positions
    symbols = [element_symbol(element) for element in atoms.elements]
    colors = [color_for_element(symbol) for symbol in symbols]
    sizes = [18 if symbol != "H" else 10 for symbol in symbols]
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


def bond_trace(atoms) -> go.Scatter3d:
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


def frame_title(reaction_token: str, index: int, energy_hartree: float | None) -> str:
    title = f"{reaction_token} | frame {index + 1}"
    if energy_hartree is not None:
        title += f" | E = {energy_hartree:.6f} Eh"
    return title


def render_interactive_html(frames, reaction_token: str, output_path: Path) -> None:
    x_range, y_range, z_range = scene_bounds(frames)
    plotly_frames = []
    slider_steps = []

    for index, frame in enumerate(frames):
        traces = [bond_trace(frame.atoms), atom_trace(frame.atoms)]
        plotly_frames.append(
            go.Frame(
                name=str(index),
                data=traces,
                layout={"title": {"text": frame_title(reaction_token, index, frame.energy_hartree)}},
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
        data=[bond_trace(frames[0].atoms), atom_trace(frames[0].atoms)],
        frames=plotly_frames,
    )
    fig.update_layout(
        title={"text": frame_title(reaction_token, 0, frames[0].energy_hartree)},
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


def main() -> None:
    args = parse_args()
    requested_reactions = collect_requested_reactions(args.reactions_file, args.reaction_ids)
    if not requested_reactions:
        raise RuntimeError("No reaction IDs provided. Pass a reactions file and/or one or more --reaction-id flags.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = DatabaseManager(args.db_name, args.ip, args.port)
    manager.loadCollections()
    model = Model(args.method_family, args.method, args.basisset, args.spin_mode, args.program)

    for requested in requested_reactions:
        step = select_lowest_barrier_step_for_direction(requested, manager, model, args.energy_type)
        frames = sample_step_frames(step, manager, args.frames)
        reaction_token = f"{requested.reaction_id};{requested.direction};"
        output_path = output_dir / f"{requested.reaction_id}_{requested.direction}.html"
        render_interactive_html(frames, reaction_token, output_path)
        print(f"Rendered {reaction_token}")
        print(f"  elementary step: {step.get_id().string()}")
        print(f"  html: {output_path}")


if __name__ == "__main__":
    main()
