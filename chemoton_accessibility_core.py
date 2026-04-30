from __future__ import annotations

import csv
import math
import re
import warnings
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
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
PLANCK_CONSTANT_J_S = 6.62607015e-34
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
    charge: int

    @property
    def constituent_smiles(self) -> tuple[str, ...]:
        return tuple(constituent.smiles for constituent in self.constituents)


@dataclass(frozen=True)
class ConstituentRecord:
    smiles: str
    formula: str
    structured_formula: str
    masm_cbor_graph: str


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
        serialized_components = tuple(component for component in graph.split(";") if component)
        molecules = [
            masm.JsonSerialization(
                masm.JsonSerialization.base_64_decode(component),
                masm.JsonSerialization.BinaryFormat.CBOR,
            ).to_molecule()
            for component in serialized_components
        ]
        constituents = tuple(
            ConstituentRecord(
                smiles=masm.io.experimental.emit_smiles(molecule),
                formula=molecular_formula_from_molecule(molecule),
                structured_formula=structured_formula_from_molecule(molecule),
                masm_cbor_graph=_standardized_masm_cbor_graph(molecule),
            )
            for molecule in molecules
        )
        if constituents:
            return constituents
    except Exception:
        pass
    return tuple()


def _standardized_masm_cbor_graph(molecule: masm.Molecule) -> str:
    serializer = masm.JsonSerialization(molecule)
    standardized = serializer.standardize()
    return masm.JsonSerialization.base_64_encode(
        standardized.to_binary(masm.JsonSerialization.BinaryFormat.CBOR)
    )


def _format_charge_suffix(charge: int) -> str:
    if charge == 0:
        return ""
    magnitude = abs(charge)
    sign = "+" if charge > 0 else "-"
    return f" [{'' if magnitude == 1 else magnitude}{sign}]"


def _append_charge_label(label: str, charge: int) -> str:
    if charge == 0 or label == "None":
        return label
    if " + " in label:
        return f"({label}){_format_charge_suffix(charge)}"
    return f"{label}{_format_charge_suffix(charge)}"


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
        charge = optimized_structure.get_charge()
        return CompoundRecord(
            compound_id=compound_id,
            smiles=constituent.smiles,
            formula=_append_charge_label(constituent.formula, charge),
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


@dataclass(frozen=True)
class CatalyticCycle:
    catalyst_id: str
    reaction_path: tuple[tuple[str, ReactionDirection], ...]
    tracked_species_path: tuple[str, ...]
    net_stoichiometry: tuple[tuple[str, int], ...]
    included_species: tuple[str, ...]
    max_barrier_kj_per_mol: float
    sum_delta_e_kj_per_mol: float


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
        charge = centroid.get_charge()
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
            formula=_append_charge_label(formula, charge),
            structured_formula=_append_charge_label(structured_formula, charge),
            constituents=constituents,
            multiplicity=centroid.get_multiplicity(),
            charge=charge,
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
                charge = optimized_structure.get_charge()
                record = CompoundRecord(
                    compound_id=db_compound.get_id().string(),
                    smiles=constituent.smiles,
                    formula=_append_charge_label(constituent.formula, charge),
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


def count_side_molecules(
    aggregate_cache: AggregateCache,
    aggregate_ids: Iterable[str],
    aggregate_types: Iterable[str],
) -> int:
    total = 0
    for aggregate_id, aggregate_type in zip(aggregate_ids, aggregate_types):
        total += len(aggregate_cache.get(aggregate_id, aggregate_type).constituents)
    return total


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


def exceeds_max_reactant_molecules(
    aggregate_cache: AggregateCache,
    direction: ReactionDirection,
    max_reactant_molecules: int | None,
) -> bool:
    if max_reactant_molecules is None:
        return False
    reactant_count = count_side_molecules(
        aggregate_cache, direction.reactant_ids, direction.reactant_types
    )
    return reactant_count > max_reactant_molecules


def exceeds_max_delta_e_kj_per_mol(
    direction: ReactionDirection,
    max_delta_e_kj_per_mol: float | None,
) -> bool:
    if max_delta_e_kj_per_mol is None:
        return False
    delta_e_kj_per_mol = (
        direction.rhs_energy_hartree - direction.lhs_energy_hartree
    ) * HARTREE_TO_KJ_PER_MOL
    return delta_e_kj_per_mol >= max_delta_e_kj_per_mol


def barrier_cutoff_from_minimum_rate_constant(
    temperature_k: float,
    minimum_rate_constant_s_inv: float,
) -> float:
    if minimum_rate_constant_s_inv <= 0.0:
        raise ValueError("minimum_rate_constant_s^-1 must be positive if set.")
    argument = (
        PLANCK_CONSTANT_J_S * minimum_rate_constant_s_inv
    ) / (BOLTZMANN_CONSTANT_J_PER_K * temperature_k)
    return -GAS_CONSTANT_KJ_PER_MOL_K * temperature_k * math.log(argument)


def delta_e_cutoff_from_minimum_equilibrium_constant(
    temperature_k: float,
    minimum_equilibrium_constant: float,
) -> float:
    if minimum_equilibrium_constant <= 0.0:
        raise ValueError("minimum_equilibrium_constant must be positive if set.")
    return -GAS_CONSTANT_KJ_PER_MOL_K * temperature_k * math.log(
        minimum_equilibrium_constant
    )


def resolve_effective_energy_cutoffs(
    temperature_k: float,
    max_barrier_kj_per_mol: float | None,
    max_delta_e_kj_per_mol: float | None,
    minimum_rate_constant_s_inv: float | None,
    minimum_equilibrium_constant: float | None,
) -> tuple[float | None, float | None]:
    effective_max_barrier = max_barrier_kj_per_mol
    effective_max_delta_e = max_delta_e_kj_per_mol

    if minimum_rate_constant_s_inv is not None:
        if max_barrier_kj_per_mol is not None:
            warnings.warn(
                "Both max_barrier_kj_per_mol and minimum_rate_constant_s^-1 are set; "
                "using the rate-derived barrier cutoff.",
                RuntimeWarning,
                stacklevel=2,
            )
        effective_max_barrier = barrier_cutoff_from_minimum_rate_constant(
            temperature_k, minimum_rate_constant_s_inv
        )

    if minimum_equilibrium_constant is not None:
        if max_delta_e_kj_per_mol is not None:
            warnings.warn(
                "Both max_delta_e_kj_per_mol and minimum_equilibrium_constant are set; "
                "using the equilibrium-derived Delta E cutoff.",
                RuntimeWarning,
                stacklevel=2,
            )
        effective_max_delta_e = delta_e_cutoff_from_minimum_equilibrium_constant(
            temperature_k, minimum_equilibrium_constant
        )

    return effective_max_barrier, effective_max_delta_e


def competition_gap_from_minimum_rate_ratio(
    temperature_k: float,
    minimum_competitive_rate_ratio: float,
) -> float:
    if minimum_competitive_rate_ratio <= 0.0 or minimum_competitive_rate_ratio > 1.0:
        raise ValueError("minimum_competitive_rate_ratio must satisfy 0 < ratio <= 1 if set.")
    return -GAS_CONSTANT_KJ_PER_MOL_K * temperature_k * math.log(
        minimum_competitive_rate_ratio
    )


def resolve_effective_competition_filter(
    temperature_k: float,
    competition_filter: float | None,
    minimum_competitive_rate_ratio: float | None,
) -> float:
    effective_competition_filter = 0.0 if competition_filter is None else competition_filter
    if minimum_competitive_rate_ratio is not None:
        if competition_filter is not None:
            warnings.warn(
                "Both competition_filter and minimum_competitive_rate_ratio are set; "
                "using the rate-ratio-derived competition gap.",
                RuntimeWarning,
                stacklevel=2,
            )
        effective_competition_filter = competition_gap_from_minimum_rate_ratio(
            temperature_k, minimum_competitive_rate_ratio
        )
    return effective_competition_filter


def screen_network(
    evaluated_reactions: list[EvaluatedReaction],
    aggregate_cache: AggregateCache,
    starting_compound_ids: Iterable[str],
    max_barrier: float,
    max_reactant_molecules: int | None = None,
    max_delta_e_kj_per_mol: float | None = None,
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
                if exceeds_max_reactant_molecules(
                    aggregate_cache, direction, max_reactant_molecules
                ):
                    continue
                if exceeds_max_delta_e_kj_per_mol(
                    direction, max_delta_e_kj_per_mol
                ):
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


def _direction_token(reaction_id: str, direction: ReactionDirection) -> str:
    return f"{reaction_id};{direction.network_direction};"


def _direction_delta_e_kj_per_mol(direction: ReactionDirection) -> float:
    return (
        direction.rhs_energy_hartree - direction.lhs_energy_hartree
    ) * HARTREE_TO_KJ_PER_MOL


def _sorted_directions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
) -> list[tuple[str, ReactionDirection]]:
    return sorted(
        reaction_directions,
        key=lambda item: (item[1].barrier_kj_per_mol, item[0], item[1].direction_label),
    )


def _reactant_competition_key(direction: ReactionDirection) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(zip(direction.reactant_ids, direction.reactant_types)))


def _reactant_filter_key(
    aggregate_cache: AggregateCache,
    direction: ReactionDirection,
    rotamer_filter: bool,
):
    if rotamer_filter:
        return _reactant_rotamer_key(aggregate_cache, direction)
    return _reactant_competition_key(direction)


def _aggregate_connectivity_key(
    aggregate_cache: AggregateCache,
    aggregate_id: str,
    aggregate_type: str,
) -> tuple[str, ...]:
    record = aggregate_cache.get(aggregate_id, aggregate_type)
    return tuple(sorted(constituent.masm_cbor_graph for constituent in record.constituents))


def _reactant_rotamer_key(
    aggregate_cache: AggregateCache,
    direction: ReactionDirection,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            _aggregate_connectivity_key(aggregate_cache, aggregate_id, aggregate_type)
            for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
        )
    )


def _deduplicate_direction_list(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
) -> list[tuple[str, ReactionDirection]]:
    unique: dict[str, tuple[str, ReactionDirection]] = {}
    for reaction_id, direction in reaction_directions:
        unique.setdefault(_direction_token(reaction_id, direction), (reaction_id, direction))
    return _sorted_directions(unique.values())


def _apply_rotamer_filter_to_directions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
) -> list[tuple[str, ReactionDirection]]:
    groups: dict[tuple[tuple[str, ...], ...], list[tuple[str, ReactionDirection]]] = {}
    for reaction_id, direction in reaction_directions:
        groups.setdefault(_reactant_rotamer_key(aggregate_cache, direction), []).append((reaction_id, direction))

    survivors: list[tuple[str, ReactionDirection]] = []
    for group in groups.values():
        min_barrier = min(item[1].barrier_kj_per_mol for item in group)
        survivors.extend(
            item for item in group if item[1].barrier_kj_per_mol == min_barrier
        )
    return _sorted_directions(survivors)


def apply_rotamer_filter_to_directions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
) -> list[tuple[str, ReactionDirection]]:
    return _apply_rotamer_filter_to_directions(reaction_directions, aggregate_cache)


def _apply_competition_filter_to_directions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
    rotamer_filter: bool,
    competition_filter: float,
) -> list[tuple[str, ReactionDirection]]:
    if competition_filter <= 0.0:
        return _sorted_directions(reaction_directions)

    groups: dict[object, list[tuple[str, ReactionDirection]]] = {}
    for reaction_id, direction in reaction_directions:
        groups.setdefault(
            _reactant_filter_key(aggregate_cache, direction, rotamer_filter),
            [],
        ).append((reaction_id, direction))

    survivors: list[tuple[str, ReactionDirection]] = []
    for group in groups.values():
        min_barrier = min(item[1].barrier_kj_per_mol for item in group)
        survivors.extend(
            item
            for item in group
            if item[1].barrier_kj_per_mol - min_barrier <= competition_filter
        )
    return _sorted_directions(survivors)


def apply_competition_filter_to_directions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
    rotamer_filter: bool,
    competition_filter: float,
) -> list[tuple[str, ReactionDirection]]:
    return _apply_competition_filter_to_directions(
        reaction_directions,
        aggregate_cache,
        rotamer_filter,
        competition_filter,
    )


def recompute_accessible_reactions(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
    starting_compound_ids: Iterable[str],
    progress: ProgressReporter | None = None,
) -> tuple[set[str], list[tuple[str, ReactionDirection]]]:
    progress = progress or ProgressReporter(False)
    candidate_directions = _deduplicate_direction_list(reaction_directions)
    starting_set = set(starting_compound_ids)
    reachable_smiles = reachable_smiles_from_starting_ids(aggregate_cache, starting_set)
    reachable_aggregates = set(starting_set)
    feasible_directions: dict[str, tuple[str, ReactionDirection]] = {}

    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        iterable = progress.wrap(
            candidate_directions,
            total=len(candidate_directions),
            desc=f"Recomputing access (pass {iteration})",
        )
        for reaction_id, direction in iterable:
            direction_key = _direction_token(reaction_id, direction)
            if direction_key in feasible_directions:
                continue
            if not all(
                aggregate_is_reachable(aggregate_cache, reachable_smiles, aggregate_id, aggregate_type)
                for aggregate_id, aggregate_type in zip(direction.reactant_ids, direction.reactant_types)
            ):
                continue
            feasible_directions[direction_key] = (reaction_id, direction)
            new_products = set(direction.product_ids) - reachable_aggregates
            if new_products:
                reachable_aggregates.update(new_products)
                for aggregate_id, aggregate_type in zip(direction.product_ids, direction.product_types):
                    reachable_smiles.update(aggregate_cache.get(aggregate_id, aggregate_type).constituent_smiles)
                changed = True

    return reachable_aggregates, _sorted_directions(feasible_directions.values())


def apply_secondary_accessibility_filters(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
    starting_compound_ids: Iterable[str],
    rotamer_filter: bool = False,
    competition_filter: float = 0.0,
    progress: ProgressReporter | None = None,
) -> tuple[set[str], list[tuple[str, ReactionDirection]]]:
    progress = progress or ProgressReporter(False)
    current_directions = _deduplicate_direction_list(reaction_directions)

    while True:
        filtered_directions = current_directions
        if rotamer_filter:
            filtered_directions = _apply_rotamer_filter_to_directions(
                filtered_directions, aggregate_cache
            )
        if competition_filter > 0.0:
            filtered_directions = _apply_competition_filter_to_directions(
                filtered_directions,
                aggregate_cache,
                rotamer_filter,
                competition_filter,
            )

        reachable_aggregates, accessible_directions = recompute_accessible_reactions(
            filtered_directions,
            aggregate_cache,
            starting_compound_ids,
            progress,
        )
        current_tokens = {
            _direction_token(reaction_id, direction)
            for reaction_id, direction in current_directions
        }
        accessible_tokens = {
            _direction_token(reaction_id, direction)
            for reaction_id, direction in accessible_directions
        }
        if accessible_tokens == current_tokens:
            return reachable_aggregates, accessible_directions
        current_directions = accessible_directions


def collect_accessible_subgraph_reactions(
    evaluated_reactions: list[EvaluatedReaction],
    aggregate_cache: AggregateCache,
    reachable_aggregates: Iterable[str],
    max_barrier: float,
    max_reactant_molecules: int | None = None,
    max_delta_e_kj_per_mol: float | None = None,
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
            if exceeds_max_reactant_molecules(
                aggregate_cache, direction, max_reactant_molecules
            ):
                continue
            if exceeds_max_delta_e_kj_per_mol(
                direction, max_delta_e_kj_per_mol
            ):
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


def collect_reactions_with_starting_reactants(
    evaluated_reactions: Iterable[EvaluatedReaction],
    aggregate_cache: AggregateCache,
    starting_compound_ids: Iterable[str],
    max_barrier: float | None = None,
    max_reactant_molecules: int | None = None,
    max_delta_e_kj_per_mol: float | None = None,
) -> list[tuple[str, ReactionDirection]]:
    starting_set = set(starting_compound_ids)
    matching_directions: list[tuple[str, ReactionDirection]] = []

    for reaction in evaluated_reactions:
        for direction in (reaction.forward, reaction.backward):
            if max_barrier is not None and direction.barrier_kj_per_mol > max_barrier:
                continue
            if is_trivial_flask_relabeling(aggregate_cache, direction):
                continue
            if exceeds_max_reactant_molecules(
                aggregate_cache, direction, max_reactant_molecules
            ):
                continue
            if exceeds_max_delta_e_kj_per_mol(
                direction, max_delta_e_kj_per_mol
            ):
                continue
            if not is_valid_smiles(direction.reactant_smiles) or not is_valid_smiles(direction.product_smiles):
                continue
            if not any(aggregate_id in starting_set for aggregate_id in direction.reactant_ids):
                continue
            matching_directions.append((reaction.reaction_id, direction))

    matching_directions.sort(
        key=lambda item: (item[1].barrier_kj_per_mol, item[0], item[1].direction_label)
    )
    return matching_directions


def direction_species_ids(direction: ReactionDirection) -> set[str]:
    return set(direction.reactant_ids) | set(direction.product_ids)


def path_species_ids(
    reaction_path: Iterable[tuple[str, ReactionDirection]],
) -> set[str]:
    species: set[str] = set()
    for _reaction_id, direction in reaction_path:
        species.update(direction_species_ids(direction))
    return species


def path_net_stoichiometry(
    reaction_path: Iterable[tuple[str, ReactionDirection]],
) -> Counter[str]:
    net: Counter[str] = Counter()
    for _reaction_id, direction in reaction_path:
        net.update(direction.product_ids)
        net.subtract(direction.reactant_ids)
    return net


def _catalytic_cycle_from_path(
    catalyst_id: str,
    reaction_path: tuple[tuple[str, ReactionDirection], ...],
    tracked_species_path: tuple[str, ...],
) -> CatalyticCycle:
    net = path_net_stoichiometry(reaction_path)
    included_species = tuple(sorted(path_species_ids(reaction_path)))
    return CatalyticCycle(
        catalyst_id=catalyst_id,
        reaction_path=reaction_path,
        tracked_species_path=tracked_species_path,
        net_stoichiometry=tuple(sorted((species_id, count) for species_id, count in net.items() if count != 0)),
        included_species=included_species,
        max_barrier_kj_per_mol=max(direction.barrier_kj_per_mol for _reaction_id, direction in reaction_path),
        sum_delta_e_kj_per_mol=sum(_direction_delta_e_kj_per_mol(direction) for _reaction_id, direction in reaction_path),
    )


def _cycle_satisfies_filters(
    cycle: CatalyticCycle,
    required_reactant_ids: set[str],
    required_product_ids: set[str],
    required_species_ids: set[str],
    forbidden_species_ids: set[str],
) -> bool:
    net = dict(cycle.net_stoichiometry)
    if net.get(cycle.catalyst_id, 0) != 0:
        return False
    if any(net.get(species_id, 0) >= 0 for species_id in required_reactant_ids):
        return False
    if any(net.get(species_id, 0) <= 0 for species_id in required_product_ids):
        return False
    included = set(cycle.included_species)
    if not required_species_ids.issubset(included):
        return False
    if forbidden_species_ids & included:
        return False
    return True


def _canonical_cycle_key(cycle: CatalyticCycle) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (
        cycle.catalyst_id,
        tuple(_direction_token(reaction_id, direction) for reaction_id, direction in cycle.reaction_path),
        cycle.tracked_species_path,
    )


def find_catalytic_cycles(
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    catalyst_ids: Iterable[str],
    max_steps: int,
    required_reactant_ids: Iterable[str] = (),
    required_product_ids: Iterable[str] = (),
    required_species_ids: Iterable[str] = (),
    forbidden_species_ids: Iterable[str] = (),
) -> list[CatalyticCycle]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    directions = _deduplicate_direction_list(reaction_directions)
    directions_by_reactant: dict[str, list[tuple[str, ReactionDirection]]] = {}
    for reaction_id, direction in directions:
        for reactant_id in set(direction.reactant_ids):
            directions_by_reactant.setdefault(reactant_id, []).append((reaction_id, direction))

    required_reactants = set(required_reactant_ids)
    required_products = set(required_product_ids)
    required_species = set(required_species_ids)
    forbidden_species = set(forbidden_species_ids)
    cycles_by_key: dict[tuple[str, tuple[str, ...], tuple[str, ...]], CatalyticCycle] = {}

    def search(
        catalyst_id: str,
        current_species_id: str,
        reaction_path: tuple[tuple[str, ReactionDirection], ...],
        tracked_species_path: tuple[str, ...],
        used_reaction_ids: set[str],
    ) -> None:
        if len(reaction_path) >= max_steps:
            return

        for reaction_id, direction in directions_by_reactant.get(current_species_id, []):
            if reaction_id in used_reaction_ids:
                continue
            next_reaction_path = reaction_path + ((reaction_id, direction),)
            for next_species_id in direction.product_ids:
                if next_species_id in tracked_species_path and next_species_id != catalyst_id:
                    continue
                next_tracked_species_path = tracked_species_path + (next_species_id,)
                if next_species_id == catalyst_id:
                    cycle = _catalytic_cycle_from_path(
                        catalyst_id,
                        next_reaction_path,
                        next_tracked_species_path,
                    )
                    if _cycle_satisfies_filters(
                        cycle,
                        required_reactants,
                        required_products,
                        required_species,
                        forbidden_species,
                    ):
                        cycles_by_key.setdefault(_canonical_cycle_key(cycle), cycle)
                    continue
                search(
                    catalyst_id,
                    next_species_id,
                    next_reaction_path,
                    next_tracked_species_path,
                    used_reaction_ids | {reaction_id},
                )

    for catalyst_id in sorted(set(catalyst_ids)):
        search(
            catalyst_id,
            catalyst_id,
            tuple(),
            (catalyst_id,),
            set(),
        )

    return sorted(
        cycles_by_key.values(),
        key=lambda cycle: (
            cycle.max_barrier_kj_per_mol,
            cycle.sum_delta_e_kj_per_mol,
            cycle.catalyst_id,
            len(cycle.reaction_path),
            tuple(_direction_token(reaction_id, direction) for reaction_id, direction in cycle.reaction_path),
        ),
    )


def _format_stoichiometry_entries(entries: Iterable[tuple[str, int]], sign: int) -> str:
    formatted: list[str] = []
    for species_id, count in entries:
        if count * sign <= 0:
            continue
        magnitude = abs(count)
        formatted.append(species_id if magnitude == 1 else f"{magnitude} {species_id}")
    return ";".join(formatted)


def write_catalytic_cycles(
    output_path: str,
    cycles: Iterable[CatalyticCycle],
    aggregate_cache: AggregateCache,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "CycleId",
                "CatalystId",
                "StepCount",
                "ReactionPath",
                "TrackedSpeciesPath",
                "Net Reactants",
                "Net Products",
                "Included Species",
                "Max Barrier (kJ/mol)",
                "Sum Delta E (kJ/mol)",
                "Chemical Equations",
            ]
        )
        for index, cycle in enumerate(cycles, start=1):
            writer.writerow(
                [
                    f"cycle_{index}",
                    cycle.catalyst_id,
                    len(cycle.reaction_path),
                    " > ".join(
                        _direction_token(reaction_id, direction)
                        for reaction_id, direction in cycle.reaction_path
                    ),
                    " > ".join(cycle.tracked_species_path),
                    _format_stoichiometry_entries(cycle.net_stoichiometry, -1),
                    _format_stoichiometry_entries(cycle.net_stoichiometry, 1),
                    ";".join(cycle.included_species),
                    cycle.max_barrier_kj_per_mol,
                    cycle.sum_delta_e_kj_per_mol,
                    " | ".join(
                        build_formula_string(direction, aggregate_cache)
                        for _reaction_id, direction in cycle.reaction_path
                    ),
                ]
            )


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
                "LHS Total Energy (kJ/mol)",
                "RHS Total Energy (kJ/mol)",
                "LHS IDs",
                "RHS IDs",
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
                    direction.lhs_energy_hartree * HARTREE_TO_KJ_PER_MOL,
                    direction.rhs_energy_hartree * HARTREE_TO_KJ_PER_MOL,
                    ";".join(direction.reactant_ids),
                    ";".join(direction.product_ids),
                    direction.barrier_kj_per_mol,
                    (direction.rhs_energy_hartree - direction.lhs_energy_hartree) * HARTREE_TO_KJ_PER_MOL,
                ]
            )


def write_reactions_with_opposite_barrier(
    output_path: str,
    reaction_directions: Iterable[tuple[str, ReactionDirection]],
    aggregate_cache: AggregateCache,
    evaluated_reactions: Iterable[EvaluatedReaction],
):
    reaction_lookup = {reaction.reaction_id: reaction for reaction in evaluated_reactions}

    with open(output_path, "w", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ReactionId",
                "Reaction",
                "Chemical Equation",
                "Structured Chemical Equation",
                "LHS Total Energy (kJ/mol)",
                "RHS Total Energy (kJ/mol)",
                "LHS IDs",
                "RHS IDs",
                "Barrier (kJ/mol)",
                "Opposite Direction Barrier (kJ/mol)",
                "Delta E (kJ/mol)",
            ]
        )
        for reaction_id, direction in reaction_directions:
            reaction = reaction_lookup[reaction_id]
            opposite_direction = reaction.backward if direction.direction_label == "forward" else reaction.forward
            writer.writerow(
                [
                    f"{reaction_id};{direction.network_direction};",
                    build_reaction_string(direction),
                    build_formula_string(direction, aggregate_cache),
                    build_structured_formula_string(direction, aggregate_cache),
                    direction.lhs_energy_hartree * HARTREE_TO_KJ_PER_MOL,
                    direction.rhs_energy_hartree * HARTREE_TO_KJ_PER_MOL,
                    ";".join(direction.reactant_ids),
                    ";".join(direction.product_ids),
                    direction.barrier_kj_per_mol,
                    opposite_direction.barrier_kj_per_mol,
                    (direction.rhs_energy_hartree - direction.lhs_energy_hartree) * HARTREE_TO_KJ_PER_MOL,
                ]
            )
