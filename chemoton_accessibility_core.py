from __future__ import annotations

import csv
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from collections import Counter
from typing import Iterable, Iterator

import numpy as np
import scine_database as db
import scine_database.energy_query_functions as dbfxn
import scine_molassembler as masm
import scine_utilities as utils
from scine_database.compound_and_flask_creation import get_compound_or_flask
from tqdm.auto import tqdm


HARTREE_TO_KJ_PER_MOL = 2625.4996394799
GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
OPTIMIZED_STRUCTURE_LABELS = {
    db.Label.MINIMUM_OPTIMIZED,
    db.Label.USER_OPTIMIZED,
    db.Label.SURFACE_OPTIMIZED,
    db.Label.COMPLEX_OPTIMIZED,
    db.Label.USER_COMPLEX_OPTIMIZED,
    db.Label.SURFACE_COMPLEX_OPTIMIZED,
    db.Label.USER_SURFACE_OPTIMIZED,
    db.Label.USER_SURFACE_COMPLEX_OPTIMIZED,
}

DEFAULT_CONFIG = {
    "db_name": "ch3sh-ch2sh",
    "ip": "localhost",
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
    "molecule_output": "molecules.csv",
    "reaction_output": "reactions.csv",
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


@dataclass(frozen=True)
class DatabaseConfig:
    db_name: str
    ip: str
    port: int


@dataclass(frozen=True)
class ModelConfig:
    method_family: str
    method: str
    basisset: str
    spin_mode: str
    program: str


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

    @staticmethod
    def count_collection(collection: db.Collection) -> int | None:
        try:
            return collection.count("{}")
        except Exception:
            return None


class ProgressReporter:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def wrap(
        self,
        iterable: Iterable,
        *,
        total: int | None = None,
        desc: str,
    ) -> Iterable:
        if not self.enabled:
            return iterable
        return tqdm(iterable, total=total, desc=desc, unit="item")

    def iter_with_manual_update(
        self,
        iterable: Iterable,
        *,
        total: int | None = None,
        desc: str,
    ) -> Iterator[tuple[object, callable]]:
        if not self.enabled:
            for item in iterable:
                yield item, lambda _n=1: None
            return

        progress = tqdm(total=total, desc=desc, unit="item")
        try:
            for item in iterable:
                yield item, progress.update
        finally:
            progress.close()


@dataclass(frozen=True)
class AggregateRecord:
    aggregate_id: str
    aggregate_type: str
    smiles: str
    formula: str
    structured_formula: str
    constituents: tuple["ConstituentRecord", ...]
    multiplicity: int

    @property
    def constituent_smiles(self) -> tuple[str, ...]:
        return tuple(constituent.smiles for constituent in self.constituents)


@dataclass(frozen=True)
class ConstituentRecord:
    smiles: str
    formula: str
    structured_formula: str


@dataclass(frozen=True)
class CompoundRecord:
    compound_id: str
    smiles: str
    formula: str
    multiplicity: int
    energy_hartree: float | None


@dataclass(frozen=True)
class MoleculeOutputRow:
    compound_id: str
    smiles: str
    formula: str
    multiplicity: int | None
    energy_hartree: float | None


@dataclass(frozen=True)
class RawEvaluatedReaction:
    reaction_id: str
    reactant_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    reactant_types: tuple[str, ...]
    product_types: tuple[str, ...]
    fwd_barrier_kj_per_mol: float
    bwd_barrier_kj_per_mol: float
    reactant_energy_hartree: float
    product_energy_hartree: float


def extract_constituents_from_structure(structure: db.Structure) -> tuple["ConstituentRecord", ...]:
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
        constituents = tuple(
            ConstituentRecord(
                smiles=masm.io.experimental.emit_smiles(molecule),
                formula=molecular_formula_from_molecule(molecule),
                structured_formula=structured_formula_from_molecule(molecule),
            )
            for molecule in molecules
        )
        if constituents:
            return constituents
    except Exception:
        pass
    return tuple()


def _model_from_config(config: ModelConfig) -> Model:
    return Model(
        config.method_family,
        config.method,
        config.basisset,
        config.spin_mode,
        config.program,
    )


def _database_config_from_manager(manager: DatabaseManager) -> DatabaseConfig:
    credentials = manager.get_credentials()
    return DatabaseConfig(
        db_name=credentials.database_name,
        ip=credentials.hostname,
        port=credentials.port,
    )


def _model_config_from_model(model: Model) -> ModelConfig:
    return ModelConfig(
        method_family=model.method_family,
        method=model.method,
        basisset=model.basis_set,
        spin_mode=model.spin_mode,
        program=model.program,
    )


def _sum_structure_energies_worker(
    structure_ids: Iterable[db.ID],
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
) -> float:
    total_energy = 0.0
    for structure_id in structure_ids:
        structure = db.Structure(structure_id, manager.structure_collection_)
        total_energy += dbfxn.get_energy_for_structure(
            structure,
            energy_type,
            model,
            manager.structure_collection_,
            manager.properties_collection_,
        )
    return total_energy


def _evaluate_reaction_worker(
    reaction_id: str,
    database_config: DatabaseConfig,
    model_config: ModelConfig,
    energy_type: str,
) -> RawEvaluatedReaction | None:
    try:
        manager = DatabaseManager(database_config.db_name, database_config.ip, database_config.port)
        manager.loadCollections()
        model = _model_from_config(model_config)
        db_reaction = db.Reaction(db.ID(reaction_id), manager.reaction_collection_)
        db_reaction.link(manager.reaction_collection_)
        reactants, products = db_reaction.get_reactants(db.Side.BOTH)
        reactant_types, product_types = db_reaction.get_reactant_types(db.Side.BOTH)

        elementary_step = dbfxn.get_elementary_step_with_min_ts_energy(
            db_reaction,
            energy_type,
            model,
            manager.elementary_step_collection_,
            manager.structure_collection_,
            manager.properties_collection_,
        )
        if elementary_step is None:
            return None

        reactant_structures, product_structures = elementary_step.get_reactants(db.Side.BOTH)
        reactant_energy = _sum_structure_energies_worker(reactant_structures, manager, model, energy_type)
        product_energy = _sum_structure_energies_worker(product_structures, manager, model, energy_type)
        fwd_barrier, bwd_barrier = dbfxn.get_barriers_for_elementary_step_by_type(
            elementary_step,
            energy_type,
            model,
            manager.structure_collection_,
            manager.properties_collection_,
        )
        if fwd_barrier < 0.0 or bwd_barrier < 0.0:
            return None

        return RawEvaluatedReaction(
            reaction_id=reaction_id,
            reactant_ids=tuple(compound_id.string() for compound_id in reactants),
            product_ids=tuple(compound_id.string() for compound_id in products),
            reactant_types=tuple(rtype.name for rtype in reactant_types),
            product_types=tuple(ptype.name for ptype in product_types),
            fwd_barrier_kj_per_mol=fwd_barrier,
            bwd_barrier_kj_per_mol=bwd_barrier,
            reactant_energy_hartree=reactant_energy,
            product_energy_hartree=product_energy,
        )
    except Exception:
        return None


def _get_energy_for_structure_worker(
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


def _lowest_energy_optimized_structure_for_compound_worker(
    compound: db.Compound,
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
) -> db.Structure | None:
    best_structure: db.Structure | None = None
    best_energy: float | None = None
    for structure_id in compound.get_structures():
        structure = db.Structure(structure_id, manager.structure_collection_)
        if structure.get_label() not in OPTIMIZED_STRUCTURE_LABELS:
            continue
        energy = _get_energy_for_structure_worker(structure, manager, model, energy_type)
        if energy is None:
            continue
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_structure = structure
    return best_structure


def _compound_record_worker(
    compound_id: str,
    database_config: DatabaseConfig,
    model_config: ModelConfig,
    energy_type: str,
) -> CompoundRecord | None:
    try:
        manager = DatabaseManager(database_config.db_name, database_config.ip, database_config.port)
        manager.loadCollections()
        model = _model_from_config(model_config)
        db_compound = db.Compound(db.ID(compound_id), manager.compound_collection_)
        db_compound.link(manager.compound_collection_)
        optimized_structure = _lowest_energy_optimized_structure_for_compound_worker(
            db_compound, manager, model, energy_type
        )
        if optimized_structure is None:
            return None
        constituents = extract_constituents_from_structure(optimized_structure)
        if len(constituents) != 1:
            return None
        constituent = constituents[0]
        return CompoundRecord(
            compound_id=compound_id,
            smiles=constituent.smiles,
            formula=constituent.formula,
            multiplicity=optimized_structure.get_multiplicity(),
            energy_hartree=_get_energy_for_structure_worker(
                optimized_structure, manager, model, energy_type
            ),
        )
    except Exception:
        return None


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
        constituents = self._extract_constituents(centroid)
        smiles = " + ".join(constituent.smiles for constituent in constituents) if constituents else "None"
        formula = " + ".join(constituent.formula for constituent in constituents) if constituents else "None"
        structured_formula = (
            " + ".join(constituent.structured_formula for constituent in constituents)
            if constituents
            else "None"
        )
        return AggregateRecord(
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            smiles=smiles,
            formula=formula,
            structured_formula=structured_formula,
            constituents=constituents,
            multiplicity=centroid.get_multiplicity(),
        )

    def _extract_constituents(self, structure: db.Structure) -> tuple[ConstituentRecord, ...]:
        return extract_constituents_from_structure(structure)


class CompoundIndex:
    def __init__(
        self,
        manager: DatabaseManager,
        model: Model,
        energy_type: str,
        progress: ProgressReporter | None = None,
        jobs: int = 1,
    ):
        self.manager_ = manager
        self.model_ = model
        self.energy_type_ = energy_type
        self.progress_ = progress or ProgressReporter(False)
        self.jobs_ = max(1, jobs)
        self._by_smiles: dict[str, list[CompoundRecord]] | None = None
        self._by_id: dict[str, CompoundRecord] | None = None

    def get_by_id(self, compound_id: str) -> CompoundRecord | None:
        self._ensure_loaded()
        return self._by_id.get(compound_id) if self._by_id is not None else None

    def find_by_smiles(
        self,
        smiles: str,
        allowed_multiplicities: set[int] | None = None,
    ) -> list[CompoundRecord]:
        self._ensure_loaded()
        candidates = self._by_smiles.get(smiles, []) if self._by_smiles is not None else []
        if allowed_multiplicities is None:
            return candidates
        return [candidate for candidate in candidates if candidate.multiplicity in allowed_multiplicities]

    def _ensure_loaded(self) -> None:
        if self._by_smiles is not None:
            return

        by_smiles: dict[str, list[CompoundRecord]] = {}
        by_id: dict[str, CompoundRecord] = {}
        total_compounds = DatabaseManager.count_collection(self.manager_.compound_collection_)
        if self.jobs_ == 1:
            compounds = self.progress_.wrap(
                self.manager_.compound_collection_.iterate_all_compounds(),
                total=total_compounds,
                desc="Indexing compounds",
            )
            for db_compound in compounds:
                db_compound.link(self.manager_.compound_collection_)
                optimized_structure = self._get_lowest_energy_optimized_structure(db_compound)
                if optimized_structure is None:
                    continue
                constituents = extract_constituents_from_structure(optimized_structure)
                if len(constituents) != 1:
                    continue
                constituent = constituents[0]
                record = CompoundRecord(
                    compound_id=db_compound.get_id().string(),
                    smiles=constituent.smiles,
                    formula=constituent.formula,
                    multiplicity=optimized_structure.get_multiplicity(),
                    energy_hartree=self._get_energy(optimized_structure),
                )
                by_smiles.setdefault(record.smiles, []).append(record)
                by_id[record.compound_id] = record
        else:
            compound_ids = [
                db_compound.get_id().string()
                for db_compound in self.manager_.compound_collection_.iterate_all_compounds()
            ]
            database_config = _database_config_from_manager(self.manager_)
            model_config = _model_config_from_model(self.model_)
            with ProcessPoolExecutor(max_workers=self.jobs_) as executor:
                records = self.progress_.wrap(
                    executor.map(
                        _compound_record_worker,
                        compound_ids,
                        [database_config] * len(compound_ids),
                        [model_config] * len(compound_ids),
                        [self.energy_type_] * len(compound_ids),
                    ),
                    total=total_compounds,
                    desc="Indexing compounds",
                )
                for record in records:
                    if record is None:
                        continue
                    by_smiles.setdefault(record.smiles, []).append(record)
                    by_id[record.compound_id] = record

        for records in by_smiles.values():
            records.sort(key=lambda record: (record.multiplicity, record.compound_id))
        self._by_smiles = by_smiles
        self._by_id = by_id

    def _get_lowest_energy_optimized_structure(self, compound: db.Compound) -> db.Structure | None:
        best_structure: db.Structure | None = None
        best_energy: float | None = None
        for structure_id in compound.get_structures():
            structure = db.Structure(structure_id, self.manager_.structure_collection_)
            if structure.get_label() not in OPTIMIZED_STRUCTURE_LABELS:
                continue
            energy = self._get_energy(structure)
            if energy is None:
                continue
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_structure = structure
        return best_structure

    def _get_energy(self, structure: db.Structure) -> float | None:
        try:
            return dbfxn.get_energy_for_structure(
                structure,
                self.energy_type_,
                self.model_,
                self.manager_.structure_collection_,
                self.manager_.properties_collection_,
            )
        except Exception:
            return None


class ReactionEvaluator:
    def __init__(
        self,
        manager: DatabaseManager,
        model: Model,
        energy_type: str,
        temperature_k: float,
        progress: ProgressReporter | None = None,
        jobs: int = 1,
    ):
        self.manager_ = manager
        self.model_ = model
        self.energy_type_ = energy_type
        self.temperature_k_ = temperature_k
        self.progress_ = progress or ProgressReporter(False)
        self.jobs_ = max(1, jobs)

    def evaluate_all(self, aggregate_cache: AggregateCache) -> list[EvaluatedReaction]:
        reactions: list[EvaluatedReaction] = []
        total_reactions = DatabaseManager.count_collection(self.manager_.reaction_collection_)
        if self.jobs_ == 1:
            iterable = self.progress_.wrap(
                self.manager_.reaction_collection_.iterate_all_reactions(),
                total=total_reactions,
                desc="Evaluating reactions",
            )
            for db_reaction in iterable:
                db_reaction.link(self.manager_.reaction_collection_)
                evaluated = self._evaluate_one(db_reaction, aggregate_cache)
                if evaluated is not None:
                    reactions.append(evaluated)
        else:
            reaction_ids = [
                db_reaction.get_id().string()
                for db_reaction in self.manager_.reaction_collection_.iterate_all_reactions()
            ]
            database_config = _database_config_from_manager(self.manager_)
            model_config = _model_config_from_model(self.model_)
            with ProcessPoolExecutor(max_workers=self.jobs_) as executor:
                raw_reactions = self.progress_.wrap(
                    executor.map(
                        _evaluate_reaction_worker,
                        reaction_ids,
                        [database_config] * len(reaction_ids),
                        [model_config] * len(reaction_ids),
                        [self.energy_type_] * len(reaction_ids),
                    ),
                    total=total_reactions,
                    desc="Evaluating reactions",
                )
                for raw_reaction in raw_reactions:
                    if raw_reaction is not None:
                        reactions.append(self._from_raw(raw_reaction, aggregate_cache))
        return reactions

    def _from_raw(
        self,
        raw_reaction: RawEvaluatedReaction,
        aggregate_cache: AggregateCache,
    ) -> EvaluatedReaction:
        reactant_smiles = tuple(
            aggregate_cache.get(aggregate_id, aggregate_type).smiles
            for aggregate_id, aggregate_type in zip(raw_reaction.reactant_ids, raw_reaction.reactant_types)
        )
        product_smiles = tuple(
            aggregate_cache.get(aggregate_id, aggregate_type).smiles
            for aggregate_id, aggregate_type in zip(raw_reaction.product_ids, raw_reaction.product_types)
        )
        return EvaluatedReaction(
            reaction_id=raw_reaction.reaction_id,
            forward=ReactionDirection(
                direction_label="forward",
                reactant_ids=raw_reaction.reactant_ids,
                product_ids=raw_reaction.product_ids,
                reactant_types=raw_reaction.reactant_types,
                product_types=raw_reaction.product_types,
                reactant_smiles=reactant_smiles,
                product_smiles=product_smiles,
                barrier_kj_per_mol=raw_reaction.fwd_barrier_kj_per_mol,
                lhs_energy_hartree=raw_reaction.reactant_energy_hartree,
                rhs_energy_hartree=raw_reaction.product_energy_hartree,
            ),
            backward=ReactionDirection(
                direction_label="backward",
                reactant_ids=raw_reaction.product_ids,
                product_ids=raw_reaction.reactant_ids,
                reactant_types=raw_reaction.product_types,
                product_types=raw_reaction.reactant_types,
                reactant_smiles=product_smiles,
                product_smiles=reactant_smiles,
                barrier_kj_per_mol=raw_reaction.bwd_barrier_kj_per_mol,
                lhs_energy_hartree=raw_reaction.product_energy_hartree,
                rhs_energy_hartree=raw_reaction.reactant_energy_hartree,
            ),
        )

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


def build_formula_string(direction: ReactionDirection, aggregate_cache: AggregateCache) -> str:
    lhs = " + ".join(
        aggregate_cache.get(aggregate_id, aggregate_type).formula
        for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
    )
    rhs = " + ".join(
        aggregate_cache.get(aggregate_id, aggregate_type).formula
        for aggregate_id, aggregate_type in zip(direction.product_ids, direction.product_types)
    )
    return f"{lhs} -> {rhs}"


def build_structured_formula_string(direction: ReactionDirection, aggregate_cache: AggregateCache) -> str:
    lhs = " + ".join(
        aggregate_cache.get(aggregate_id, aggregate_type).structured_formula
        for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
    )
    rhs = " + ".join(
        aggregate_cache.get(aggregate_id, aggregate_type).structured_formula
        for aggregate_id, aggregate_type in zip(direction.product_ids, direction.product_types)
    )
    return f"{lhs} -> {rhs}"


def is_valid_smiles(smiles_values: Iterable[str]) -> bool:
    return all(smiles not in ("", "None") for smiles in smiles_values)


def molecular_formula_from_molecule(molecule: masm.Molecule) -> str:
    counts: Counter[str] = Counter()
    for element in molecule.graph.elements():
        symbol = base_element_symbol(element)
        counts[symbol] += 1

    if not counts:
        return "None"

    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
        if "H" in counts:
            ordered_symbols.append("H")
    ordered_symbols.extend(
        symbol for symbol in sorted(counts) if symbol not in ordered_symbols
    )
    return "".join(
        symbol if counts[symbol] == 1 else f"{symbol}{counts[symbol]}"
        for symbol in ordered_symbols
    )


def structured_formula_from_molecule(molecule: masm.Molecule) -> str:
    graph = molecule.graph
    atoms = list(graph.atoms())
    hydrogen_atoms = {atom for atom in atoms if base_element_symbol(graph.element_type(atom)) == "H"}
    heavy_atoms = [atom for atom in atoms if atom not in hydrogen_atoms]

    if not heavy_atoms:
        return molecular_formula_from_molecule(molecule)

    try:
        if graph.cycles.num_relevant_cycles() > 0:
            return molecular_formula_from_molecule(molecule)
    except Exception:
        pass

    heavy_adjacency = {
        atom: sorted(
            neighbor for neighbor in graph.adjacents(atom)
            if neighbor in heavy_atoms
        )
        for atom in heavy_atoms
    }
    hydrogen_counts = {
        atom: sum(1 for neighbor in graph.adjacents(atom) if neighbor in hydrogen_atoms)
        for atom in heavy_atoms
    }

    def atom_rank(atom: int) -> tuple[int, int]:
        symbol = base_element_symbol(graph.element_type(atom))
        priority = {"C": 0, "S": 1, "N": 2, "O": 3}.get(symbol, 4)
        return (priority, atom)

    root_candidates = [atom for atom in heavy_atoms if len(heavy_adjacency[atom]) <= 1]
    if not root_candidates:
        root_candidates = heavy_atoms
    root = min(root_candidates, key=atom_rank)

    subtree_sizes: dict[tuple[int, int | None], int] = {}

    def subtree_size(atom: int, parent: int | None) -> int:
        key = (atom, parent)
        if key in subtree_sizes:
            return subtree_sizes[key]
        size = 1 + sum(
            subtree_size(neighbor, atom)
            for neighbor in heavy_adjacency[atom]
            if neighbor != parent
        )
        subtree_sizes[key] = size
        return size

    def atom_group(atom: int) -> str:
        symbol = base_element_symbol(graph.element_type(atom))
        hydrogens = hydrogen_counts[atom]
        if hydrogens == 0:
            return symbol
        if hydrogens == 1:
            return f"{symbol}H"
        return f"{symbol}H{hydrogens}"

    def render(atom: int, parent: int | None) -> str:
        children = [neighbor for neighbor in heavy_adjacency[atom] if neighbor != parent]
        if not children:
            return atom_group(atom)

        children.sort(
            key=lambda child: (
                -subtree_size(child, atom),
                atom_rank(child),
            )
        )
        main_child = children[0]
        branch_children = children[1:]
        branch_string = "".join(f"({render(child, atom)})" for child in branch_children)
        return atom_group(atom) + branch_string + render(main_child, atom)

    return render(root, None)


def base_element_symbol(element: utils.ElementType) -> str:
    symbol = utils.ElementInfo.symbol(element)
    if symbol in {"D", "T"}:
        return "H"
    return re.sub(r"\d+", "", symbol)


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
    progress: ProgressReporter | None = None,
) -> tuple[set[str], list[tuple[str, ReactionDirection]]]:
    starting_set = set(starting_compound_ids)
    reachable_smiles = reachable_smiles_from_starting_ids(aggregate_cache, starting_set)
    reachable_aggregates = set(starting_set)
    feasible_directions: dict[str, tuple[str, ReactionDirection]] = {}
    progress = progress or ProgressReporter(False)

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        iterable = progress.wrap(
            evaluated_reactions,
            total=len(evaluated_reactions),
            desc=f"Propagating access (pass {iteration})",
        )
        for reaction in iterable:
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
    progress: ProgressReporter | None = None,
) -> list[tuple[str, ReactionDirection]]:
    progress = progress or ProgressReporter(False)
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
    iterable = progress.wrap(
        evaluated_reactions,
        total=len(evaluated_reactions),
        desc="Collecting subgraph reactions",
    )
    for reaction in iterable:
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
    manager: DatabaseManager,
    model: Model,
    energy_type: str,
    multiplicity_mode: str = "singlet-doublet",
    progress: ProgressReporter | None = None,
    jobs: int = 1,
):
    aggregate_types: dict[str, str] = {}
    for reaction in evaluated_reactions:
        for aggregate_id, aggregate_type in zip(reaction.forward.reactant_ids, reaction.forward.reactant_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)
        for aggregate_id, aggregate_type in zip(reaction.forward.product_ids, reaction.forward.product_types):
            aggregate_types.setdefault(aggregate_id, aggregate_type)

    allowed_multiplicities = None if multiplicity_mode == "all" else {1, 2}
    progress = progress or ProgressReporter(False)
    compound_index = CompoundIndex(manager, model, energy_type, progress, jobs)
    rows: list[MoleculeOutputRow] = []
    seen: set[tuple[str, str, str, int | None]] = set()

    for aggregate_id in sorted(set(aggregate_ids)):
        aggregate_type = aggregate_types.get(aggregate_id, "COMPOUND")
        record = aggregate_cache.get(aggregate_id, aggregate_type)
        if aggregate_type == "COMPOUND":
            match = compound_index.get_by_id(aggregate_id)
            if match is None:
                continue
            if allowed_multiplicities is not None and match.multiplicity not in allowed_multiplicities:
                continue
            key = (match.compound_id, match.smiles, match.formula, match.multiplicity)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                MoleculeOutputRow(
                    compound_id=match.compound_id,
                    smiles=match.smiles,
                    formula=match.formula,
                    multiplicity=match.multiplicity,
                    energy_hartree=match.energy_hartree,
                )
            )
            continue

        for constituent in record.constituents:
            matches = compound_index.find_by_smiles(constituent.smiles, allowed_multiplicities)
            if not matches:
                key = ("NO ID", constituent.smiles, constituent.formula, None)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    MoleculeOutputRow(
                        compound_id="NO ID",
                        smiles=constituent.smiles,
                        formula=constituent.formula,
                        multiplicity=None,
                        energy_hartree=None,
                    )
                )
                continue
            for match in matches:
                key = (match.compound_id, match.smiles, match.formula, match.multiplicity)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    MoleculeOutputRow(
                        compound_id=match.compound_id,
                        smiles=match.smiles,
                        formula=match.formula,
                        multiplicity=match.multiplicity,
                        energy_hartree=match.energy_hartree,
                    )
                )

    rows.sort(
        key=lambda row: (
            row.compound_id == "NO ID",
            row.formula,
            row.smiles,
            -1 if row.multiplicity is None else row.multiplicity,
            row.compound_id,
        )
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CompoundId", "SMILES", "ChemicalFormula", "Multiplicity", "Energy (Eh)"])
        for row in rows:
            writer.writerow(
                [
                    row.compound_id,
                    row.smiles,
                    row.formula,
                    "" if row.multiplicity is None else row.multiplicity,
                    "" if row.energy_hartree is None else row.energy_hartree,
                ]
            )


def write_reactions(
    output_path: str,
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
):
    with open(output_path, "w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ReactionId",
                "Reaction",
                "Chemical Equation",
                "Structured Chemical Equation",
                "Barrier (kJ/mol)",
                "Delta E (kJ/mol)",
            ]
        )
        for reaction_id, direction in reaction_directions:
            writer.writerow(
                [
                    f"{reaction_id};{direction.network_direction};",
                    build_reaction_string(direction),
                    build_formula_string(direction, aggregate_cache),
                    build_structured_formula_string(direction, aggregate_cache),
                    direction.barrier_kj_per_mol,
                    (direction.rhs_energy_hartree - direction.lhs_energy_hartree) * HARTREE_TO_KJ_PER_MOL,
                ]
            )
