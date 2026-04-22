from __future__ import annotations

DATABASE_DEFAULTS = {
    "db_name": "c6h7+",
    "ip": "localhost",
    "port": 27017,
}

MODEL_DEFAULTS = {
    "method_family": "dft",
    "method": "m06",
    "basisset": "def2-svp",
    "spin_mode": "unrestricted",
    "program": "orca",
}

ACCESSIBILITY_DEFAULTS = {
    "energy_type": "electronic_energy",
    "temperature_k": 300.0,
    "max_barrier_kj_per_mol": 10000.0,
    "starting_compound_ids": [
        "69dfb60ed57f03007d05a0f5"
    ]
}

ACCESSIBLE_NETWORK_DEFAULTS = {
    "molecule_output": "molecules.csv",
    "reaction_output": "reactions.csv",
}

ACCESSIBLE_SUBGRAPH_DEFAULTS = {
    "molecule_output": "accessible_subgraph_molecules.csv",
    "reaction_output": "accessible_subgraph_reactions.csv",
}

STARTING_REACTANT_REACTIONS_DEFAULTS = {
    "reaction_output": "starting_reactant_reactions.csv",
}

GIF_RENDER_DEFAULTS = {
    "frames": 48,
    "fps": 10,
    "render_dim": "3d",
    "view_elev": 18.0,
    "view_azim": 38.0,
    "rotate_azim_deg": 0.0,
    "rotate_elev_deg": 0.0,
    "render_quality": "high",
    "output_dir": "rendered_reactions",
}

INTERACTIVE_RENDER_DEFAULTS = {
    "frames": 48,
    "render_quality": "high",
    "output_dir": "interactive_reactions",
}
