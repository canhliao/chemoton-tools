from __future__ import annotations

DATABASE_DEFAULTS = {
    "db_name": "ch3sh-ch2sh",
    "ip": "localhost",
    "port": 27017,
}

MODEL_DEFAULTS = {
    "method_family": "dft",
    "method": "m062x",
    "basisset": "6-311+G**",
    "spin_mode": "unrestricted",
    "program": "orca",
}

ACCESSIBILITY_DEFAULTS = {
    "energy_type": "electronic_energy",
    "temperature_k": 300.0,
    "max_barrier_kj_per_mol": 150.0,
    "max_delta_e_kj_per_mol": 20.0,
    "max_reactant_molecules": 2,
    "starting_compound_ids": [
        "69c1abfbdf7e55117102846a",
        "69c290cd54afd82e0701e3f3",
        "69c2a6b754afd82e0701e3ff",
        "69c2e5d854afd82e0701e43d",
        "69c2f41054afd82e0701e449",
        "69c2f7da54afd82e0701e459",
        "69c1ac9bdf7e55117102846c"
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
