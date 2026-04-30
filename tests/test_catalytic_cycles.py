from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_scine_stubs() -> None:
    db = types.ModuleType("scine_database")

    class _Label:
        MINIMUM_OPTIMIZED = object()
        USER_OPTIMIZED = object()
        SURFACE_OPTIMIZED = object()
        COMPLEX_OPTIMIZED = object()
        USER_COMPLEX_OPTIMIZED = object()
        SURFACE_COMPLEX_OPTIMIZED = object()
        USER_SURFACE_OPTIMIZED = object()
        USER_SURFACE_COMPLEX_OPTIMIZED = object()

    class _Model:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Manager:
        pass

    class _Credentials:
        def __init__(self, *_args, **_kwargs):
            pass

    class _Collection:
        pass

    class _CompoundOrFlask:
        COMPOUND = object()
        FLASK = object()

    db.Label = _Label
    db.Model = _Model
    db.Manager = _Manager
    db.Credentials = _Credentials
    db.Collection = _Collection
    db.CompoundOrFlask = _CompoundOrFlask
    db.ID = lambda value: value
    db.Structure = object
    db.Reaction = object
    db.Compound = object

    dbfxn = types.ModuleType("scine_database.energy_query_functions")
    creation = types.ModuleType("scine_database.compound_and_flask_creation")
    creation.get_compound_or_flask = lambda *_args, **_kwargs: None

    masm = types.ModuleType("scine_molassembler")
    masm.JsonSerialization = object
    masm.Molecule = object
    masm.io = types.SimpleNamespace(experimental=types.SimpleNamespace(emit_smiles=lambda _molecule: ""))

    utils = types.ModuleType("scine_utilities")
    utils.ElementType = object
    utils.ElementInfo = types.SimpleNamespace(symbol=lambda element: str(element))

    sys.modules.setdefault("scine_database", db)
    sys.modules.setdefault("scine_database.energy_query_functions", dbfxn)
    sys.modules.setdefault("scine_database.compound_and_flask_creation", creation)
    sys.modules.setdefault("scine_molassembler", masm)
    sys.modules.setdefault("scine_utilities", utils)


def _load_core():
    _install_scine_stubs()
    module_path = Path(__file__).resolve().parents[1] / "chemoton_accessibility_core.py"
    spec = importlib.util.spec_from_file_location("chemoton_accessibility_core_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()


def direction(reactants, products, barrier=10.0, lhs=0.0, rhs=0.0):
    return core.ReactionDirection(
        direction_label="forward",
        reactant_ids=tuple(reactants),
        product_ids=tuple(products),
        reactant_types=tuple("COMPOUND" for _ in reactants),
        product_types=tuple("COMPOUND" for _ in products),
        reactant_smiles=tuple(reactants),
        product_smiles=tuple(products),
        barrier_kj_per_mol=barrier,
        lhs_energy_hartree=lhs,
        rhs_energy_hartree=rhs,
    )


def test_finds_catalyst_regenerating_cycle_with_required_net_species():
    reactions = [
        ("r1", direction(("C", "A"), ("I",), barrier=15.0)),
        ("r2", direction(("I", "B"), ("C", "D"), barrier=20.0)),
    ]

    cycles = core.find_catalytic_cycles(
        reactions,
        catalyst_ids=("C",),
        max_steps=2,
        required_reactant_ids=("A", "B"),
        required_product_ids=("D",),
    )

    assert len(cycles) == 1
    assert cycles[0].tracked_species_path == ("C", "I", "C")
    assert cycles[0].max_barrier_kj_per_mol == 20.0


def test_rejects_paths_that_do_not_regenerate_catalyst():
    reactions = [
        ("r1", direction(("C", "A"), ("I",))),
        ("r2", direction(("I", "B"), ("J", "D"))),
    ]

    cycles = core.find_catalytic_cycles(reactions, catalyst_ids=("C",), max_steps=2)

    assert cycles == []


def test_enforces_required_product():
    reactions = [
        ("r1", direction(("C", "A"), ("I",))),
        ("r2", direction(("I",), ("C", "E"))),
    ]

    cycles = core.find_catalytic_cycles(
        reactions,
        catalyst_ids=("C",),
        max_steps=2,
        required_product_ids=("D",),
    )

    assert cycles == []


def test_enforces_include_and_exclude_species_filters():
    reactions = [
        ("r1", direction(("C", "A"), ("I",))),
        ("r2", direction(("I",), ("C", "D"))),
    ]

    included = core.find_catalytic_cycles(
        reactions,
        catalyst_ids=("C",),
        max_steps=2,
        required_species_ids=("I",),
    )
    excluded = core.find_catalytic_cycles(
        reactions,
        catalyst_ids=("C",),
        max_steps=2,
        forbidden_species_ids=("I",),
    )

    assert len(included) == 1
    assert excluded == []


def test_respects_max_steps():
    reactions = [
        ("r1", direction(("C",), ("I",))),
        ("r2", direction(("I",), ("J",))),
        ("r3", direction(("J",), ("C",))),
    ]

    cycles = core.find_catalytic_cycles(reactions, catalyst_ids=("C",), max_steps=2)

    assert cycles == []


def test_rejects_reusing_same_database_reaction():
    reactions = [
        ("r1", direction(("C",), ("I",))),
        ("r1", direction(("I",), ("C",))),
    ]

    cycles = core.find_catalytic_cycles(reactions, catalyst_ids=("C",), max_steps=2)

    assert cycles == []
