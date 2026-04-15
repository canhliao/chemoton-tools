from __future__ import annotations

import argparse

from chemoton_accessibility_core import (
    AggregateCache,
    DEFAULT_CONFIG,
    DatabaseManager,
    Model,
    ProgressReporter,
    ReactionEvaluator,
    screen_network,
    write_molecules,
    write_reactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find accessible aggregates in a Chemoton database from a starting set of compound IDs."
    )
    parser.add_argument("--db-name", default=DEFAULT_CONFIG["db_name"])
    parser.add_argument("--ip", default=DEFAULT_CONFIG["ip"])
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"])
    parser.add_argument("--energy-type", default=DEFAULT_CONFIG["energy_type"])
    parser.add_argument("--temperature-k", type=float, default=DEFAULT_CONFIG["temperature_k"])
    parser.add_argument("--max-barrier-kj-per-mol", type=float, default=DEFAULT_CONFIG["max_barrier_kj_per_mol"])
    parser.add_argument("--method-family", default="dft")
    parser.add_argument("--method", default="m062x")
    parser.add_argument("--basisset", default="6-311+G**")
    parser.add_argument("--spin-mode", default="unrestricted")
    parser.add_argument("--program", default="orca")
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
    parser.add_argument("--molecule-output", default=DEFAULT_CONFIG["molecule_output"])
    parser.add_argument("--reaction-output", default=DEFAULT_CONFIG["reaction_output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    starting_ids = args.starting_ids if args.starting_ids else DEFAULT_CONFIG["starting_compound_ids"]

    manager = DatabaseManager(args.db_name, args.ip, args.port)
    manager.loadCollections()
    model = Model(args.method_family, args.method, args.basisset, args.spin_mode, args.program)
    progress = ProgressReporter(args.progress)

    aggregate_cache = AggregateCache(manager)
    evaluator = ReactionEvaluator(manager, model, args.energy_type, args.temperature_k, progress)

    print("Loading and screening reactions.")
    evaluated_reactions = evaluator.evaluate_all(aggregate_cache)

    print("Propagating accessible aggregates from the starting set.")
    accessible_aggregates, accessible_reactions = screen_network(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        starting_compound_ids=starting_ids,
        max_barrier=args.max_barrier_kj_per_mol,
        progress=progress,
    )

    write_molecules(
        args.molecule_output,
        accessible_aggregates,
        aggregate_cache,
        evaluated_reactions,
        manager,
        model,
        args.energy_type,
        args.compound_multiplicity_mode,
        progress,
    )
    write_reactions(args.reaction_output, accessible_reactions, aggregate_cache)

    print(f"Accessible aggregates: {len(accessible_aggregates)}")
    print(f"Accessible directional reactions: {len(accessible_reactions)}")


if __name__ == "__main__":
    main()
