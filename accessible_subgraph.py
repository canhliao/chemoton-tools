from __future__ import annotations

import argparse

from chemoton_accessibility_core import (
    AggregateCache,
    apply_secondary_accessibility_filters,
    DatabaseManager,
    Model,
    ProgressReporter,
    ReactionEvaluator,
    collect_accessible_subgraph_reactions,
    screen_network,
    write_molecules,
    write_reactions,
)
from user_input_config import (
    ACCESSIBILITY_DEFAULTS,
    ACCESSIBLE_SUBGRAPH_DEFAULTS,
    DATABASE_DEFAULTS,
    MODEL_DEFAULTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the accessible low-barrier aggregate subgraph of a Chemoton database."
    )
    parser.add_argument("--db-name", default=DATABASE_DEFAULTS["db_name"])
    parser.add_argument("--ip", default=DATABASE_DEFAULTS["ip"])
    parser.add_argument("--port", type=int, default=DATABASE_DEFAULTS["port"])
    parser.add_argument("--energy-type", default=ACCESSIBILITY_DEFAULTS["energy_type"])
    parser.add_argument("--temperature-k", type=float, default=ACCESSIBILITY_DEFAULTS["temperature_k"])
    parser.add_argument("--max-barrier-kj-per-mol", type=float, default=ACCESSIBILITY_DEFAULTS["max_barrier_kj_per_mol"])
    parser.add_argument("--method-family", default=MODEL_DEFAULTS["method_family"])
    parser.add_argument("--method", default=MODEL_DEFAULTS["method"])
    parser.add_argument("--basisset", default=MODEL_DEFAULTS["basisset"])
    parser.add_argument("--spin-mode", default=MODEL_DEFAULTS["spin_mode"])
    parser.add_argument("--program", default=MODEL_DEFAULTS["program"])
    parser.add_argument(
        "--starting-id",
        action="append",
        dest="starting_ids",
        default=None,
        help="Starting compound ID. Repeat to provide multiple IDs.",
    )
    parser.add_argument(
        "--compound-multiplicity-mode",
        choices=("singlet-doublet", "all"),
        default="singlet-doublet",
        help="Filter molecule-output rows to compounds with multiplicity 1 or 2, or include all multiplicities.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars for the database-heavy phases.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes to use for parallelizable phases. Default: 1.",
    )
    parser.add_argument("--molecule-output", default=ACCESSIBLE_SUBGRAPH_DEFAULTS["molecule_output"])
    parser.add_argument("--reaction-output", default=ACCESSIBLE_SUBGRAPH_DEFAULTS["reaction_output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = max(1, args.jobs)
    starting_ids = args.starting_ids if args.starting_ids else ACCESSIBILITY_DEFAULTS["starting_compound_ids"]

    manager = DatabaseManager(args.db_name, args.ip, args.port)
    manager.loadCollections()
    model = Model(args.method_family, args.method, args.basisset, args.spin_mode, args.program)
    progress = ProgressReporter(args.progress)

    aggregate_cache = AggregateCache(manager)
    evaluator = ReactionEvaluator(manager, model, args.energy_type, args.temperature_k, progress, jobs)

    print("Loading reactions.")
    evaluated_reactions = evaluator.evaluate_all(aggregate_cache)

    print("Propagating reachable aggregates from the starting set.")
    reachable_aggregates, _ = screen_network(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        starting_compound_ids=starting_ids,
        max_barrier=args.max_barrier_kj_per_mol,
        max_reactant_molecules=ACCESSIBILITY_DEFAULTS["max_reactant_molecules"],
        max_delta_e_kj_per_mol=ACCESSIBILITY_DEFAULTS["max_delta_e_kj_per_mol"],
        progress=progress,
    )

    print("Collecting all low-barrier reactions within the reachable aggregate subgraph.")
    accessible_reactions = collect_accessible_subgraph_reactions(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        reachable_aggregates=reachable_aggregates,
        max_barrier=args.max_barrier_kj_per_mol,
        max_reactant_molecules=ACCESSIBILITY_DEFAULTS["max_reactant_molecules"],
        max_delta_e_kj_per_mol=ACCESSIBILITY_DEFAULTS["max_delta_e_kj_per_mol"],
        progress=progress,
    )
    if ACCESSIBILITY_DEFAULTS["rotamer_filter"] or ACCESSIBILITY_DEFAULTS["competition_filter"] > 0.0:
        print("Applying secondary accessibility screening.")
        reachable_aggregates, accessible_reactions = apply_secondary_accessibility_filters(
            reaction_directions=accessible_reactions,
            aggregate_cache=aggregate_cache,
            starting_compound_ids=starting_ids,
            rotamer_filter=ACCESSIBILITY_DEFAULTS["rotamer_filter"],
            competition_filter=ACCESSIBILITY_DEFAULTS["competition_filter"],
            progress=progress,
        )

    write_molecules(
        args.molecule_output,
        reachable_aggregates,
        aggregate_cache,
        evaluated_reactions,
        manager,
        model,
        args.energy_type,
        args.compound_multiplicity_mode,
        progress,
        jobs,
    )
    write_reactions(args.reaction_output, accessible_reactions, aggregate_cache)

    print(f"Accessible aggregates: {len(reachable_aggregates)}")
    print(f"Accessible subgraph reactions: {len(accessible_reactions)}")


if __name__ == "__main__":
    main()
