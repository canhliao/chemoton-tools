from __future__ import annotations

import argparse

from chemoton_accessibility_core import (
    AggregateCache,
    DatabaseManager,
    Model,
    ProgressReporter,
    ReactionEvaluator,
    collect_reactions_with_starting_reactants,
    resolve_effective_energy_cutoffs,
    write_reactions_with_opposite_barrier,
)
from user_input_config import (
    ACCESSIBILITY_DEFAULTS,
    DATABASE_DEFAULTS,
    MODEL_DEFAULTS,
    STARTING_REACTANT_REACTIONS_DEFAULTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List directional reactions whose reactant side contains at least one of the starting compound IDs."
        )
    )
    parser.add_argument("--db-name", default=DATABASE_DEFAULTS["db_name"])
    parser.add_argument("--ip", default=DATABASE_DEFAULTS["ip"])
    parser.add_argument("--port", type=int, default=DATABASE_DEFAULTS["port"])
    parser.add_argument("--energy-type", default=ACCESSIBILITY_DEFAULTS["energy_type"])
    parser.add_argument("--temperature-k", type=float, default=ACCESSIBILITY_DEFAULTS["temperature_k"])
    parser.add_argument(
        "--max-barrier-kj-per-mol",
        type=float,
        default=ACCESSIBILITY_DEFAULTS["max_barrier_kj_per_mol"],
        help="Maximum directional barrier to include. Default matches the package accessibility cutoff.",
    )
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
        "--progress",
        action="store_true",
        help="Show progress bars for the database-heavy phases.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes to use for reaction evaluation. Default: 1.",
    )
    parser.add_argument("--reaction-output", default=STARTING_REACTANT_REACTIONS_DEFAULTS["reaction_output"])
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
    effective_max_barrier, effective_max_delta_e = resolve_effective_energy_cutoffs(
        temperature_k=args.temperature_k,
        max_barrier_kj_per_mol=args.max_barrier_kj_per_mol,
        max_delta_e_kj_per_mol=ACCESSIBILITY_DEFAULTS["max_delta_e_kj_per_mol"],
        minimum_rate_constant_s_inv=ACCESSIBILITY_DEFAULTS["minimum_rate_constant_s^-1"],
        minimum_equilibrium_constant=ACCESSIBILITY_DEFAULTS["minimum_equilibrium_constant"],
    )

    print("Loading reactions.")
    evaluated_reactions = evaluator.evaluate_all(aggregate_cache)

    print("Collecting directional reactions that use a starting compound as a reactant.")
    matching_reactions = collect_reactions_with_starting_reactants(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        starting_compound_ids=starting_ids,
        max_barrier=effective_max_barrier,
        max_reactant_molecules=ACCESSIBILITY_DEFAULTS["max_reactant_molecules"],
        max_delta_e_kj_per_mol=effective_max_delta_e,
    )

    write_reactions_with_opposite_barrier(
        args.reaction_output,
        matching_reactions,
        aggregate_cache,
        evaluated_reactions,
    )

    print(f"Starting compounds provided: {len(starting_ids)}")
    print(f"Matching directional reactions: {len(matching_reactions)}")


if __name__ == "__main__":
    main()
