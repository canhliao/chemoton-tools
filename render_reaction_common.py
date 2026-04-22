from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Iterable
import warnings

import numpy as np
import scine_database as db
import scine_database.energy_query_functions as dbfxn
import scine_utilities as utils

from chemoton_accessibility_core import (
    AggregateCache,
    HARTREE_TO_KJ_PER_MOL,
    DatabaseManager,
    Model,
)
import scine_molassembler as masm


@dataclass(frozen=True)
class RequestedReaction:
    reaction_id: str
    direction: str


@dataclass(frozen=True)
class TrajectoryFrame:
    atoms: utils.AtomCollection
    delta_energy_kj_per_mol: float | None


class SkippedReaction(RuntimeError):
    pass


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
    return list(dict.fromkeys(requested))


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
    total_steps = 0
    missing_barrier_data = 0
    missing_requested_barrier = 0
    negative_barrier = 0
    missing_trajectory = 0

    for step_id in reaction.get_elementary_steps():
        total_steps += 1
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
            missing_barrier_data += 1
            continue
        barrier = barriers[0] if requested.direction == "0" else barriers[1]
        if barrier is None:
            missing_requested_barrier += 1
            continue
        if barrier < 0.0:
            negative_barrier += 1
            continue
        if not (step.has_spline() or step.has_path()):
            missing_trajectory += 1
            continue
        if best_barrier is None or barrier < best_barrier:
            best_barrier = barrier
            best_step = step

    if best_step is None:
        reasons: list[str] = []
        if total_steps == 0:
            reasons.append("reaction has no elementary steps")
        if missing_barrier_data:
            reasons.append(f"{missing_barrier_data} step(s) missing forward/backward barrier data")
        if missing_requested_barrier:
            reasons.append(f"{missing_requested_barrier} step(s) missing the requested directional barrier")
        if negative_barrier:
            reasons.append(f"{negative_barrier} step(s) have negative requested-direction barrier")
        if missing_trajectory:
            reasons.append(f"{missing_trajectory} step(s) have no spline/path trajectory data")
        if not reasons:
            reasons.append("no step matched the renderer selection criteria")
        raise SkippedReaction(
            f"Skipped {requested.reaction_id};{requested.direction};: " + "; ".join(reasons)
        )
    return best_step


def sample_step_frames(
    step: db.ElementaryStep,
    manager: DatabaseManager,
    frame_count: int,
    model: Model,
    energy_type: str,
    reaction_id: str | None = None,
    direction: str = "0",
) -> list[TrajectoryFrame]:
    if step.has_spline():
        spline = step.get_spline()
        sample_points = np.linspace(0.0, 1.0, frame_count)
        frames: list[TrajectoryFrame] = []
        for x in sample_points:
            _, atoms = spline.evaluate(float(x))
            frames.append(TrajectoryFrame(atoms=atoms, delta_energy_kj_per_mol=None))
        return _orient_frames_for_requested_direction(
            frames, manager, reaction_id, direction
        )
    if step.has_path():
        frames: list[TrajectoryFrame] = []
        raw_energies: list[float | None] = []
        structure_ids = list(step.get_path())
        for structure_id in structure_ids:
            structure = db.Structure(structure_id, manager.structure_collection_)
            raw_energies.append(
                _get_energy_for_structure(
                    structure,
                    manager,
                    model,
                    energy_type,
                )
            )
            frames.append(TrajectoryFrame(atoms=structure.get_atoms(), delta_energy_kj_per_mol=None))
        if frames:
            if _should_reverse_frames(frames, manager, reaction_id, direction):
                frames.reverse()
                raw_energies.reverse()
            if raw_energies and raw_energies[0] is not None and all(energy is not None for energy in raw_energies):
                reference_energy = float(raw_energies[0])
                return [
                    TrajectoryFrame(
                        atoms=frame.atoms,
                        delta_energy_kj_per_mol=(float(energy) - reference_energy) * HARTREE_TO_KJ_PER_MOL,
                    )
                    for frame, energy in zip(frames, raw_energies)
                ]
            return frames
    raise RuntimeError(f"Elementary step {step.get_id().string()} has neither spline nor path data.")


def _orient_frames_for_requested_direction(
    frames: list[TrajectoryFrame],
    manager: DatabaseManager,
    reaction_id: str | None,
    direction: str,
) -> list[TrajectoryFrame]:
    if _should_reverse_frames(frames, manager, reaction_id, direction):
        return list(reversed(frames))
    return frames


def _should_reverse_frames(
    frames: list[TrajectoryFrame],
    manager: DatabaseManager,
    reaction_id: str | None,
    direction: str,
) -> bool:
    if reaction_id is None or len(frames) < 2:
        return direction == "1"

    try:
        expected_reactants, expected_products = _expected_side_signatures(
            reaction_id, direction, manager, signature_kind="graph"
        )
        first_signatures = _frame_component_graph_signatures(frames[0].atoms)
        last_signatures = _frame_component_graph_signatures(frames[-1].atoms)
    except Exception:
        expected_reactants, expected_products = _expected_side_signatures(
            reaction_id, direction, manager, signature_kind="formula"
        )
        first_signatures = _frame_component_formula_signatures(frames[0].atoms)
        last_signatures = _frame_component_formula_signatures(frames[-1].atoms)

    current_score = _signature_distance(first_signatures, expected_reactants) + _signature_distance(
        last_signatures, expected_products
    )
    reversed_score = _signature_distance(first_signatures, expected_products) + _signature_distance(
        last_signatures, expected_reactants
    )
    if reversed_score != current_score:
        return reversed_score < current_score
    warnings.warn(
        (
            f"Ambiguous trajectory orientation for reaction {reaction_id};{direction}; "
            "using requested direction as fallback."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return direction == "1"


def _expected_side_signatures(
    reaction_id: str,
    direction: str,
    manager: DatabaseManager,
    signature_kind: str,
) -> tuple[Counter[str], Counter[str]]:
    reaction = db.Reaction(db.ID(reaction_id), manager.reaction_collection_)
    reaction.link(manager.reaction_collection_)
    reactants, products = reaction.get_reactants(db.Side.BOTH)
    reactant_types, product_types = reaction.get_reactant_types(db.Side.BOTH)

    lhs_ids = tuple(aggregate_id.string() for aggregate_id in reactants)
    rhs_ids = tuple(aggregate_id.string() for aggregate_id in products)
    lhs_types = tuple(aggregate_type.name for aggregate_type in reactant_types)
    rhs_types = tuple(aggregate_type.name for aggregate_type in product_types)
    if direction == "1":
        lhs_ids, rhs_ids = rhs_ids, lhs_ids
        lhs_types, rhs_types = rhs_types, lhs_types

    aggregate_cache = AggregateCache(manager)
    return (
        _side_component_signatures(aggregate_cache, lhs_ids, lhs_types, signature_kind),
        _side_component_signatures(aggregate_cache, rhs_ids, rhs_types, signature_kind),
    )


def _side_component_signatures(
    aggregate_cache: AggregateCache,
    aggregate_ids: Iterable[str],
    aggregate_types: Iterable[str],
    signature_kind: str,
) -> Counter[str]:
    signatures: Counter[str] = Counter()
    for aggregate_id, aggregate_type in zip(aggregate_ids, aggregate_types):
        record = aggregate_cache.get(aggregate_id, aggregate_type)
        if signature_kind == "graph":
            signatures.update(constituent.masm_cbor_graph for constituent in record.constituents)
        elif signature_kind == "formula":
            signatures.update(constituent.formula for constituent in record.constituents)
        else:
            raise ValueError(f"Unsupported signature kind: {signature_kind}")
    return signatures


def _frame_component_graph_signatures(atoms: utils.AtomCollection) -> Counter[str]:
    signatures: Counter[str] = Counter()
    bond_orders = utils.BondDetector.detect_bonds(atoms)
    interpreted = masm.interpret.molecules(
        atoms,
        bond_orders,
        masm.interpret.BondDiscretization.RoundToNearest,
    )
    for molecule in interpreted.molecules:
        signatures[_molecule_graph_signature(molecule)] += 1
    return signatures


def _frame_component_formula_signatures(atoms: utils.AtomCollection) -> Counter[str]:
    bond_orders = utils.BondDetector.detect_bonds(atoms)
    matrix = bond_orders.matrix.toarray() if hasattr(bond_orders.matrix, "toarray") else np.asarray(bond_orders.matrix)
    count = matrix.shape[0]
    seen: set[int] = set()
    signatures: Counter[str] = Counter()
    for start in range(count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in range(count):
                if matrix[current, neighbor] > 0.1 and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        signatures[_formula_for_component(atoms, component)] += 1
    return signatures


def _molecule_graph_signature(molecule: masm.Molecule) -> str:
    molecule.canonicalize()
    serializer = masm.JsonSerialization(molecule)
    standardized = serializer.standardize()
    return masm.JsonSerialization.base_64_encode(
        standardized.to_binary(masm.JsonSerialization.BinaryFormat.CBOR)
    )


def _formula_for_component(atoms: utils.AtomCollection, indices: Iterable[int]) -> str:
    counts: Counter[str] = Counter()
    for index in indices:
        counts[element_symbol(atoms.elements[index])] += 1
    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
        if "H" in counts:
            ordered_symbols.append("H")
    ordered_symbols.extend(symbol for symbol in sorted(counts) if symbol not in ordered_symbols)
    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}"
        for symbol in ordered_symbols
    )


def _signature_distance(actual: Counter[str], expected: Counter[str]) -> int:
    all_keys = set(actual) | set(expected)
    return sum(abs(actual.get(key, 0) - expected.get(key, 0)) for key in all_keys)


def _get_energy_for_structure(
    structure: db.Structure,
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
) -> float | None:
    try:
        return dbfxn.get_energy_for_structure(
            structure,
            energy_type,
            model,
            manager.structure_collection_,
            manager.properties_collection_,
        )
    except Exception:
        return None


def element_symbol(element) -> str:
    return element.name.capitalize()


def write_xyz_trajectory(frames: Iterable[TrajectoryFrame], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, frame in enumerate(frames):
            atoms = frame.atoms
            handle.write(f"{atoms.size()}\n")
            comment = f"frame={index}"
            if frame.delta_energy_kj_per_mol is not None:
                comment += f" delta_energy_kj_per_mol={frame.delta_energy_kj_per_mol}"
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
