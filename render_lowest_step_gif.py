from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from chemoton_accessibility_core import DEFAULT_CONFIG, DatabaseManager, Model
from render_reaction_common import (
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
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir", default="rendered_reactions")
    return parser.parse_args()

def render_gif(frames: list[TrajectoryFrame], output_path: Path, fps: int) -> None:
    all_positions = np.concatenate([frame.atoms.positions for frame in frames], axis=0)
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.6 * span if span > 0 else 1.0

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    adjusted_fps = max(1, round((fps / 3) * 2))
    writer = PillowWriter(fps=adjusted_fps)
    writer.setup(fig, str(output_path), dpi=120)
    try:
        for frame in frames:
            ax.cla()
            atoms = frame.atoms
            positions = atoms.positions
            symbols = [element_symbol(element) for element in atoms.elements]
            colors = [color_for_element(symbol) for symbol in symbols]
            sizes = [140 if symbol != "H" else 70 for symbol in symbols]

            for i, j in bond_pairs(atoms):
                xyz = positions[[i, j], :]
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#808080", linewidth=2.0, zorder=1)

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
            ax.set_axis_off()
            ax.view_init(elev=18, azim=38)
            if frame.energy_hartree is not None:
                ax.set_title(f"E = {frame.energy_hartree:.6f} Eh", pad=12)
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
    output_dir: Path,
) -> tuple[Path, Path, Path, str]:
    step = select_lowest_barrier_step_for_direction(requested, manager, model, energy_type)
    frames = sample_step_frames(step, manager, frame_count)

    stem = f"{requested.reaction_id}_{requested.direction}"
    xyz_path = output_dir / f"{stem}.xyz"
    vmd_path = output_dir / f"{stem}.vmd.tcl"
    gif_path = output_dir / f"{stem}.gif"

    write_xyz_trajectory(frames, xyz_path)
    write_vmd_script(xyz_path, vmd_path)
    render_gif(frames, gif_path, fps)
    return gif_path, xyz_path, vmd_path, step.get_id().string()


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
        gif_path, xyz_path, vmd_path, step_id = render_requested_reaction(
            requested=requested,
            manager=manager,
            model=model,
            energy_type=args.energy_type,
            frame_count=args.frames,
            fps=args.fps,
            output_dir=output_dir,
        )
        print(f"Rendered {requested.reaction_id};{requested.direction};")
        print(f"  elementary step: {step_id}")
        print(f"  gif: {gif_path}")
        print(f"  xyz: {xyz_path}")
        print(f"  vmd script: {vmd_path}")


if __name__ == "__main__":
    main()
