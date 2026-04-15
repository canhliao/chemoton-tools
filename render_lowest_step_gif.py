from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import scine_database as db
import scine_database.energy_query_functions as dbfxn
import scine_utilities as utils

from chemoton_accessibility_core import DEFAULT_CONFIG, DatabaseManager, Model


@dataclass(frozen=True)
class RequestedReaction:
    reaction_id: str
    direction: str


@dataclass(frozen=True)
class TrajectoryFrame:
    atoms: utils.AtomCollection
    energy_hartree: float | None


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


def read_requested_reactions(path: str) -> list[RequestedReaction]:
    requested: list[RequestedReaction] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(",", 1)[0].strip()
            parts = [part for part in token.split(";") if part != ""]
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid reaction line '{raw_line.rstrip()}'. Expected format like '69c50ae865f50a833301e4d5;1;'."
                )
            reaction_id, direction = parts[0], parts[1]
            if direction not in {"0", "1"}:
                raise ValueError(f"Invalid direction '{direction}' in line '{raw_line.rstrip()}'.")
            requested.append(RequestedReaction(reaction_id=reaction_id, direction=direction))
    return requested


def parse_requested_token(token: str) -> list[RequestedReaction]:
    cleaned = token.strip().split(",", 1)[0].strip()
    if not cleaned:
        return []
    parts = [part for part in cleaned.split(";") if part != ""]
    if len(parts) == 1:
        return [
            RequestedReaction(reaction_id=parts[0], direction="0"),
            RequestedReaction(reaction_id=parts[0], direction="1"),
        ]
    if len(parts) >= 2:
        reaction_id, direction = parts[0], parts[1]
        if direction not in {"0", "1"}:
            raise ValueError(f"Invalid direction '{direction}' in token '{token}'.")
        return [RequestedReaction(reaction_id=reaction_id, direction=direction)]
    raise ValueError(f"Invalid reaction token '{token}'.")


def collect_requested_reactions(
    reactions_file: str | None,
    reaction_ids: list[str] | None,
) -> list[RequestedReaction]:
    requested: list[RequestedReaction] = []
    if reactions_file:
        requested.extend(read_requested_reactions(reactions_file))
    if reaction_ids:
        for token in reaction_ids:
            requested.extend(parse_requested_token(token))
    deduplicated = list(dict.fromkeys(requested))
    return deduplicated


def select_lowest_barrier_step_for_direction(
    requested: RequestedReaction,
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
) -> db.ElementaryStep:
    reaction = db.Reaction(db.ID(requested.reaction_id), manager.reaction_collection_)
    reaction.link(manager.reaction_collection_)

    best_step: db.ElementaryStep | None = None
    best_barrier: float | None = None

    for step_id in reaction.get_elementary_steps():
        step = db.ElementaryStep(step_id, manager.elementary_step_collection_)
        step.link(manager.elementary_step_collection_)
        barriers = dbfxn.get_barriers_for_elementary_step_by_type(
            step,
            energy_type,
            model,
            manager.structure_collection_,
            manager.properties_collection_,
        )
        if barriers[0] is None or barriers[1] is None:
            continue
        barrier = barriers[0] if requested.direction == "0" else barriers[1]
        if barrier is None or barrier < 0.0:
            continue
        if not (step.has_spline() or step.has_path()):
            continue
        if best_barrier is None or barrier < best_barrier:
            best_barrier = barrier
            best_step = step

    if best_step is None:
        raise RuntimeError(
            f"No renderable elementary step with barrier data found for {requested.reaction_id};{requested.direction};"
        )
    return best_step


def sample_step_frames(
    step: db.ElementaryStep,
    manager: DatabaseManager,
    frame_count: int,
) -> list[TrajectoryFrame]:
    if step.has_spline():
        spline = step.get_spline()
        sample_points = np.linspace(0.0, 1.0, frame_count)
        frames: list[TrajectoryFrame] = []
        for x in sample_points:
            energy, atoms = spline.evaluate(float(x))
            frames.append(TrajectoryFrame(atoms=atoms, energy_hartree=float(energy)))
        return frames
    if step.has_path():
        frames: list[TrajectoryFrame] = []
        for structure_id in step.get_path():
            structure = db.Structure(structure_id, manager.structure_collection_)
            frames.append(TrajectoryFrame(atoms=structure.get_atoms(), energy_hartree=None))
        if frames:
            return frames
    raise RuntimeError(f"Elementary step {step.get_id().string()} has neither spline nor path data.")


def element_symbol(element) -> str:
    return element.name.capitalize()


def write_xyz_trajectory(frames: Iterable[TrajectoryFrame], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            atoms = frame.atoms
            handle.write(f"{atoms.size()}\n")
            comment = f"frame={index}"
            if frame.energy_hartree is not None:
                comment += f" energy_hartree={frame.energy_hartree}"
            handle.write(comment + "\n")
            for element, position in zip(atoms.elements, atoms.positions):
                handle.write(
                    f"{element_symbol(element)} {position[0]: .10f} {position[1]: .10f} {position[2]: .10f}\n"
                )


def write_vmd_script(xyz_path: Path, script_path: Path) -> None:
    content = f"""mol new {{{xyz_path}}} type xyz waitfor all
display projection Orthographic
axes location Off
color Display Background white
mol delrep 0 top
mol representation CPK 0.9 0.2 18.0 18.0
mol color Name
mol selection all
mol material Opaque
mol addrep top
animate style Loop
"""
    script_path.write_text(content, encoding="utf-8")


def bond_pairs(atoms: utils.AtomCollection) -> list[tuple[int, int]]:
    bond_orders = utils.BondDetector.detect_bonds(atoms)
    matrix = bond_orders.matrix.toarray() if hasattr(bond_orders.matrix, "toarray") else np.asarray(bond_orders.matrix)
    pairs: list[tuple[int, int]] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] > 0.1:
                pairs.append((i, j))
    return pairs


def color_for_element(symbol: str) -> str:
    return {
        "H": "#d9d9d9",
        "C": "#303030",
        "N": "#2f5eff",
        "O": "#e53935",
        "S": "#d4aa00",
    }.get(symbol, "#7f7f7f")


def render_gif(frames: list[TrajectoryFrame], output_path: Path, fps: int) -> None:
    all_positions = np.concatenate([frame.atoms.positions for frame in frames], axis=0)
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.6 * span if span > 0 else 1.0

    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    slowed_fps = max(1, round(fps / 3))
    writer = PillowWriter(fps=slowed_fps)
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
