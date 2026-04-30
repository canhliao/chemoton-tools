from __future__ import annotations

import argparse

from chemoton_accessibility_core import (
    AggregateCache,
    apply_secondary_accessibility_filters,
    collect_accessible_subgraph_reactions,
    DatabaseManager,
    find_catalytic_cycles,
    Model,
    ProgressReporter,
    ReactionEvaluator,
    resolve_effective_competition_filter,
    resolve_effective_energy_cutoffs,
    screen_network,
    write_catalytic_cycles,
)
from user_input_config import (
    ACCESSIBILITY_DEFAULTS,
    CATALYTIC_CYCLE_DEFAULTS,
    DATABASE_DEFAULTS,
    MODEL_DEFAULTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find catalyst-regenerating cyclic paths in the accessible low-barrier subgraph."
        )
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
        help="Starting compound ID for accessibility. Repeat to provide multiple IDs.",
    )
    parser.add_argument(
        "--catalyst-id",
        action="append",
        dest="catalyst_ids",
        required=True,
        help="Catalyst aggregate ID that must be regenerated. Repeat to search multiple catalysts.",
    )
    parser.add_argument(
        "--reactant-id",
        action="append",
        dest="reactant_ids",
        default=None,
        help="Aggregate ID that must be net consumed by a reported cycle. Repeatable.",
    )
    parser.add_argument(
        "--product-id",
        action="append",
        dest="product_ids",
        default=None,
        help="Aggregate ID that must be net produced by a reported cycle. Repeatable.",
    )
    parser.add_argument(
        "--include-species-id",
        action="append",
        dest="include_species_ids",
        default=None,
        help="Aggregate ID that must appear anywhere in a reported cycle. Repeatable.",
    )
    parser.add_argument(
        "--exclude-species-id",
        action="append",
        dest="exclude_species_ids",
        default=None,
        help="Aggregate ID that must not appear anywhere in a reported cycle. Repeatable.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=CATALYTIC_CYCLE_DEFAULTS["max_steps"],
        help="Maximum number of directional reaction steps per cycle.",
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
    parser.add_argument("--cycle-output", default=CATALYTIC_CYCLE_DEFAULTS["cycle_output"])
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
    effective_competition_filter = resolve_effective_competition_filter(
        temperature_k=args.temperature_k,
        competition_filter=ACCESSIBILITY_DEFAULTS["competition_filter"],
        minimum_competitive_rate_ratio=ACCESSIBILITY_DEFAULTS["minimum_competitive_rate_ratio"],
    )

    print("Loading reactions.")
    evaluated_reactions = evaluator.evaluate_all(aggregate_cache)

    print("Propagating reachable aggregates from the starting set.")
    reachable_aggregates, _ = screen_network(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        starting_compound_ids=starting_ids,
        max_barrier=effective_max_barrier,
        max_reactant_molecules=ACCESSIBILITY_DEFAULTS["max_reactant_molecules"],
        max_delta_e_kj_per_mol=effective_max_delta_e,
        progress=progress,
    )

    print("Collecting all low-barrier reactions within the reachable aggregate subgraph.")
    accessible_reactions = collect_accessible_subgraph_reactions(
        evaluated_reactions=evaluated_reactions,
        aggregate_cache=aggregate_cache,
        reachable_aggregates=reachable_aggregates,
        max_barrier=effective_max_barrier,
        max_reactant_molecules=ACCESSIBILITY_DEFAULTS["max_reactant_molecules"],
        max_delta_e_kj_per_mol=effective_max_delta_e,
        progress=progress,
    )
    if ACCESSIBILITY_DEFAULTS["rotamer_filter"] or effective_competition_filter > 0.0:
        print("Applying secondary accessibility screening.")
        _reachable_aggregates, accessible_reactions = apply_secondary_accessibility_filters(
            reaction_directions=accessible_reactions,
            aggregate_cache=aggregate_cache,
            starting_compound_ids=starting_ids,
            rotamer_filter=ACCESSIBILITY_DEFAULTS["rotamer_filter"],
            competition_filter=effective_competition_filter,
            progress=progress,
        )

    print("Searching catalyst-regenerating cyclic paths.")
    cycles = find_catalytic_cycles(
        reaction_directions=accessible_reactions,
        catalyst_ids=args.catalyst_ids,
        max_steps=args.max_steps,
        required_reactant_ids=args.reactant_ids or (),
        required_product_ids=args.product_ids or (),
        required_species_ids=args.include_species_ids or (),
        forbidden_species_ids=args.exclude_species_ids or (),
    )
    write_catalytic_cycles(args.cycle_output, cycles, aggregate_cache)

    print(f"Accessible subgraph reactions: {len(accessible_reactions)}")
    print(f"Catalytic cycles: {len(cycles)}")


if __name__ == "__main__":
    main()
