from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable

import numpy as np
import scine_database as db
import scine_database.energy_query_functions as dbfxn
import scine_molassembler as masm
from scine_database.compound_and_flask_creation import get_compound_or_flask


HARTREE_TO_KJ_PER_MOL = 2625.4996394799
GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324

DEFAULT_CONFIG = {
    "db_name": "ch3sh-ch2sh",
    "ip": "172.31.55.219",
    "port": 27017,
    "energy_type": "electronic_energy",
    "max_barrier_kj_per_mol": 150.0,
    "temperature_k": 300.0,
    "starting_compound_ids": [
        "69c1abfbdf7e55117102846a",
        "69c290cd54afd82e0701e3f3",
        "69c2a6b754afd82e0701e3ff",
        "69c2e5d854afd82e0701e43d",
        "69c2f41054afd82e0701e449",
        "69c2f7da54afd82e0701e459",
    ],
    "molecule_output": "molecules.txt",
    "reaction_output": "reactions.txt",
}


class Model(db.Model):
    def __init__(
        self,
        method_family: str,
        method: str,
        basisset: str,
        spin_mode: str,
        program: str,
    ):
        super().__init__(method_family, method, basisset)
        self.spin_mode = spin_mode
        self.program = program


class DatabaseManager(db.Manager):
    def __init__(self, db_name: str, ip: str, port: int):
        super().__init__()
        credentials = db.Credentials(ip, port, db_name)
        self.set_credentials(credentials)
        self.connect()

        self.reaction_collection_ = None
        self.compound_collection_ = None
        self.flask_collection_ = None
        self.structure_collection_ = None
        self.properties_collection_ = None
        self.elementary_step_collection_ = None

    def collectReactions(self):
        self.reaction_collection_ = self.get_collection("reactions")

    def collectCompounds(self):
        self.compound_collection_ = self.get_collection("compounds")

    def collectFlasks(self):
        self.flask_collection_ = self.get_collection("flasks")

    def collectStructures(self):
        self.structure_collection_ = self.get_collection("structures")

    def collectProperties(self):
        self.properties_collection_ = self.get_collection("properties")

    def collectElementarySteps(self):
        self.elementary_step_collection_ = self.get_collection("elementary_steps")

    def loadCollections(self):
        self.collectReactions()
        self.collectCompounds()
        self.collectFlasks()
        self.collectStructures()
        self.collectProperties()
        self.collectElementarySteps()


@dataclass(frozen=True)
class AggregateRecord:
    aggregate_id: str
    aggregate_type: str
    smiles: str
    constituent_smiles: tuple[str, ...]
    multiplicity: int


@dataclass(frozen=True)
class ReactionDirection:
    direction_label: str
    reactant_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    reactant_types: tuple[str, ...]
    product_types: tuple[str, ...]
    reactant_smiles: tuple[str, ...]
    product_smiles: tuple[str, ...]
    barrier_kj_per_mol: float
    lhs_energy_hartree: float
    rhs_energy_hartree: float

    @property
    def network_direction(self) -> str:
        if self.direction_label == "forward":
            return "0"
        if self.direction_label == "backward":
            return "1"
        raise ValueError(f"Unknown direction label: {self.direction_label}")


@dataclass(frozen=True)
class EvaluatedReaction:
    reaction_id: str
    forward: ReactionDirection
    backward: ReactionDirection


class AggregateCache:
    def __init__(self, manager: DatabaseManager):
        self.manager_ = manager
        self.records_: dict[tuple[str, str], AggregateRecord] = {}

    def get(self, aggregate_id: str, aggregate_type: str) -> AggregateRecord:
        key = (aggregate_id, aggregate_type)
        if key not in self.records_:
            self.records_[key] = self._load(aggregate_id, aggregate_type)
        return self.records_[key]

    def _load(self, aggregate_id: str, aggregate_type: str) -> AggregateRecord:
        compound_or_flask = getattr(db.CompoundOrFlask, aggregate_type)
        aggregate = get_compound_or_flask(
            db.ID(aggregate_id),
            compound_or_flask,
            self.manager_.compound_collection_,
            self.manager_.flask_collection_,
        )
        centroid = db.Structure(aggregate.get_centroid(), self.manager_.structure_collection_)
        constituent_smiles = self._extract_constituent_smiles(centroid)
        smiles = " + ".join(constituent_smiles) if constituent_smiles else "None"
        return AggregateRecord(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            smiles=smiles,
            constituent_smiles=constituent_smiles,
            multiplicity=centroid.get_multiplicity(),
        )

    def _extract_constituent_smiles(self, structure: db.Structure) -> tuple[str, ...]:
        try:
            graph = structure.get_graph("masm_cbor_graph")
            molecules = [
                masm.JsonSerialization(
                    masm.JsonSerialization.base_64_decode(component),
                    masm.JsonSerialization.BinaryFormat.CBOR,
                ).to_molecule()
                for component in graph.split(";")
                if component
            ]
            smiles = tuple(masm.io.experimental.emit_smiles(molecule) for molecule in molecules)
            if smiles:
                return smiles
        except Exception:
            pass
        return tuple()


class ReactionEvaluator:
    def __init__(self, manager: DatabaseManager, model: Model, energy_type: str, temperature_k: float):
        self.manager_ = manager
        self.model_ = model
        self.energy_type_ = energy_type
        self.temperature_k_ = temperature_k

    def evaluate_all(self, aggregate_cache: AggregateCache) -> list[EvaluatedReaction]:
        reactions: list[EvaluatedReaction] = []
        for db_reaction in self.manager_.reaction_collection_.iterate_all_reactions():
            db_reaction.link(self.manager_.reaction_collection_)
            evaluated = self._evaluate_one(db_reaction, aggregate_cache)
            if evaluated is not None:
                reactions.append(evaluated)
        return reactions

    def _evaluate_one(
        self, db_reaction: db.Reaction, aggregate_cache: AggregateCache
    ) -> EvaluatedReaction | None:
        try:
            reaction_id = db_reaction.get_id().string()
            reactants, products = db_reaction.get_reactants(db.Side.BOTH)
            reactant_types, product_types = db_reaction.get_reactant_types(db.Side.BOTH)

            elementary_step = dbfxn.get_elementary_step_with_min_ts_energy(
                db_reaction,
                self.energy_type_,
                self.model_,
                self.manager_.elementary_step_collection_,
                self.manager_.structure_collection_,
                self.manager_.properties_collection_,
            )
            if elementary_step is None:
                return None

            reactant_structures, product_structures = elementary_step.get_reactants(db.Side.BOTH)
            reactant_energy = self._sum_structure_energies(reactant_structures)
            product_energy = self._sum_structure_energies(product_structures)
            fwd_barrier, bwd_barrier = dbfxn.get_barriers_for_elementary_step_by_type(
                elementary_step,
                self.energy_type_,
                self.model_,
                self.manager_.structure_collection_,
                self.manager_.properties_collection_,
            )
            if fwd_barrier < 0.0 or bwd_barrier < 0.0:
                return None

            reactant_ids = tuple(compound_id.string() for compound_id in reactants)
            product_ids = tuple(compound_id.string() for compound_id in products)
            reactant_type_names = tuple(rtype.name for rtype in reactant_types)
            product_type_names = tuple(ptype.name for ptype in product_types)
            reactant_smiles = tuple(
                aggregate_cache.get(aggregate_id, aggregate_type).smiles
                for aggregate_id, aggregate_type in zip(reactant_ids, reactant_type_names)
            )
            product_smiles = tuple(
                aggregate_cache.get(aggregate_id, aggregate_type).smiles
                for aggregate_id, aggregate_type in zip(product_ids, product_type_names)
            )
            return EvaluatedReaction(
                reaction_id=reaction_id,
                forward=ReactionDirection(
                    direction_label="forward",
                    reactant_ids=reactant_ids,
                    product_ids=product_ids,
                    reactant_types=reactant_type_names,
                    product_types=product_type_names,
                    reactant_smiles=reactant_smiles,
                    product_smiles=product_smiles,
                    barrier_kj_per_mol=fwd_barrier,
                    lhs_energy_hartree=reactant_energy,
                    rhs_energy_hartree=product_energy,
                ),
                backward=ReactionDirection(
                    direction_label="backward",
                    reactant_ids=product_ids,
                    product_ids=reactant_ids,
                    reactant_types=product_type_names,
                    product_types=reactant_type_names,
                    reactant_smiles=product_smiles,
                    product_smiles=reactant_smiles,
                    barrier_kj_per_mol=bwd_barrier,
                    lhs_energy_hartree=product_energy,
                    rhs_energy_hartree=reactant_energy,
                ),
            )
        except Exception:
            return None

    def _sum_structure_energies(self, structure_ids: Iterable[db.ID]) -> float:
        total_energy = 0.0
        for structure_id in structure_ids:
            structure = db.Structure(structure_id, self.manager_.structure_collection_)
            total_energy += dbfxn.get_energy_for_structure(
                structure,
                self.energy_type_,
                self.model_,
                self.manager_.structure_collection_,
                self.manager_.properties_collection_,
            )
        return total_energy

def build_reaction_string(direction: ReactionDirection) -> str:
    return f"{' + '.join(direction.reactant_smiles)} -> {' + '.join(direction.product_smiles)}"


def is_valid_smiles(smiles_values: Iterable[str]) -> bool:
    return all(smiles not in ("", "None") for smiles in smiles_values)


def reachable_smiles_from_starting_ids(
    aggregate_cache: AggregateCache, starting_compound_ids: Iterable[str]
) -> set[str]:
    reachable_smiles: set[str] = set()
    for aggregate_id in starting_compound_ids:
        reachable_smiles.update(aggregate_cache.get(aggregate_id, "COMPOUND").constituent_smiles)
    return reachable_smiles


def aggregate_is_reachable(
    aggregate_cache: AggregateCache,
    reachable_smiles: set[str],
    aggregate_id: str,
    aggregate_type: str,
) -> bool:
    record = aggregate_cache.get(aggregate_id, aggregate_type)
    return bool(record.constituent_smiles) and all(
        smiles in reachable_smiles for smiles in record.constituent_smiles
    )


def _side_constituent_counter(
    aggregate_cache: AggregateCache,
    aggregate_ids: Iterable[str],
    aggregate_types: Iterable[str],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for aggregate_id, aggregate_type in zip(aggregate_ids, aggregate_types):
        counter.update(aggregate_cache.get(aggregate_id, aggregate_type).constituent_smiles)
    return counter


def is_trivial_flask_relabeling(
    aggregate_cache: AggregateCache,
    direction: ReactionDirection,
) -> bool:
    if "FLASK" not in direction.reactant_types and "FLASK" not in direction.product_types:
        return False
    reactant_counter = _side_constituent_counter(
        aggregate_cache, direction.reactant_ids, direction.reactant_types
    )
    product_counter = _side_constituent_counter(
        aggregate_cache, direction.product_ids, direction.product_types
    )
    return reactant_counter == product_counter


def screen_network(
    evaluated_reactions: list[EvaluatedReaction],
    aggregate_cache: AggregateCache,
    starting_compound_ids: Iterable[str],
    max_barrier: float,
) -> tuple[set[str], list[tuple[str, ReactionDirection]]]:
    starting_set = set(starting_compound_ids)
    reachable_smiles = reachable_smiles_from_starting_ids(aggregate_cache, starting_set)
    reachable_aggregates = set(starting_set)
    feasible_directions: dict[str, tuple[str, ReactionDirection]] = {}

    changed = True
    while changed:
        changed = False
        for reaction in evaluated_reactions:
            for direction in (reaction.forward, reaction.backward):
                direction_key = f"{reaction.reaction_id};{direction.network_direction};"
                if direction_key in feasible_directions:
                    continue
                if direction.barrier_kj_per_mol > max_barrier:
                    continue
                if is_trivial_flask_relabeling(aggregate_cache, direction):
                    continue
                if not all(
                    aggregate_is_reachable(aggregate_cache, reachable_smiles, aggregate_id, aggregate_type)
                    for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
                ):
                    continue
                if not is_valid_smiles(direction.reactant_smiles) or not is_valid_smiles(direction.product_smiles):
                    continue

                feasible_directions[direction_key] = (reaction.reaction_id, direction)
                new_products = set(direction.product_ids) - reachable_aggregates
                if new_products:
                    reachable_aggregates.update(new_products)
                    for aggregate_id, aggregate_type in zip(direction.product_ids, direction.product_types):
                        reachable_smiles.update(aggregate_cache.get(aggregate_id, aggregate_type).constituent_smiles)
                    changed = True

    return reachable_aggregates, sorted(
        feasible_directions.values(),
        key=lambda item: (item[1].barrier_kj_per_mol, item[0], item[1].direction_label),
    )


def collect_accessible_subgraph_reactions(
    evaluated_reactions: list[EvaluatedReaction],
    aggregate_cache: AggregateCache,
    reachable_aggregates: Iterable[str],
    max_barrier: float,
) -> list[tuple[str, ReactionDirection]]:
    aggregate_types: dict[str, str] = {}
    for reaction in evaluated_reactions:
        for aggregate_id, aggregate_type in zip(reaction.forward.reactant_ids, reaction.forward.reactant_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)
        for aggregate_id, aggregate_type in zip(reaction.forward.product_ids, reaction.forward.product_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)

    reachable_smiles: set[str] = set()
    for aggregate_id in set(reachable_aggregates):
        aggregate_type = aggregate_types.get(aggregate_id, "COMPOUND")
        reachable_smiles.update(aggregate_cache.get(aggregate_id, aggregate_type).constituent_smiles)

    accessible_directions: list[tuple[str, ReactionDirection]] = []
    for reaction in evaluated_reactions:
        for direction in (reaction.forward, reaction.backward):
            if direction.barrier_kj_per_mol > max_barrier:
                continue
            if is_trivial_flask_relabeling(aggregate_cache, direction):
                continue
            if not all(
                aggregate_is_reachable(aggregate_cache, reachable_smiles, aggregate_id, aggregate_type)
                for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
            ):
                continue
            if not all(
                aggregate_is_reachable(aggregate_cache, reachable_smiles, aggregate_id, aggregate_type)
                for aggregate_id, aggregate_type in zip(direction.product_ids, direction.product_types)
            ):
                continue
            if not is_valid_smiles(direction.reactant_smiles) or not is_valid_smiles(direction.product_smiles):
                continue
            accessible_directions.append((reaction.reaction_id, direction))

    accessible_directions.sort(
        key=lambda item: (item[1].barrier_kj_per_mol, item[0], item[1].direction_label)
    )
    return accessible_directions


def write_molecules(
    output_path: str,
    aggregate_ids: Iterable[str],
    aggregate_cache: AggregateCache,
    evaluated_reactions: Iterable[EvaluatedReaction],
):
    aggregate_types: dict[str, str] = {}
    for reaction in evaluated_reactions:
        for aggregate_id, aggregate_type in zip(reaction.forward.reactant_ids, reaction.forward.reactant_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)
        for aggregate_id, aggregate_type in zip(reaction.forward.product_ids, reaction.forward.product_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("AggregateId,Type,SMILES,Multiplicity\n")
        for aggregate_id in sorted(set(aggregate_ids)):
            aggregate_type = aggregate_types.get(aggregate_id, "COMPOUND")
            record = aggregate_cache.get(aggregate_id, aggregate_type)
            handle.write(
                f"{record.aggregate_id},{record.aggregate_type},{record.smiles},{record.multiplicity}\n"
            )


def write_reactions(output_path: str, reaction_directions: Iterable[tuple[str, ReactionDirection]]):
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("ReactionId,Reaction,Barrier (kJ/mol),LHS Energy (Eh),RHS Energy (Eh)\n")
        for reaction_id, direction in reaction_directions:
            handle.write(
                f"{reaction_id};{direction.network_direction};,"
                f"{build_reaction_string(direction)},{direction.barrier_kj_per_mol},"
                f"{direction.lhs_energy_hartree},{direction.rhs_energy_hartree}\n"
            )
