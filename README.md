# Chemoton Tools

Utilities for querying a SCINE/Chemoton reaction-network database, identifying accessible aggregates and reactions from a starting set, and rendering elementary-step trajectories to GIFs.

Most CLI default inputs now live in `user_input_config.py`. Edit that file if you want to change the package-wide defaults without passing command-line flags each time.

## Files

- [chemoton_accessibility_core.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/chemoton_accessibility_core.py:1)
  Shared library code for:
  - database/model setup
  - aggregate caching
  - flask decomposition from `masm_cbor_graph`
  - reaction evaluation
  - accessibility propagation
  - accessible subgraph collection
  - text output writers

- [user_input_config.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/user_input_config.py:1)
  Central location for user-editable default inputs used by the CLI scripts:
  - database connection defaults
  - electronic-structure model defaults
  - accessibility cutoffs and starting IDs
  - catalytic-cycle species and path-filter defaults
  - default output filenames
  - renderer defaults

- [accessible_network.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_network.py:1)
  CLI entry point for the propagation-based accessibility analysis.

- [accessible_subgraph.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_subgraph.py:1)
  CLI entry point for collecting the low-barrier reaction subgraph induced by the reachable chemistry.

- [catalytic_cycles.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/catalytic_cycles.py:1)
  CLI entry point for collecting the accessible subgraph and identifying catalyst-regenerating cyclic paths.

- [starting_reactant_reactions.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/starting_reactant_reactions.py:1)
  CLI entry point for collecting directional reactions where at least one starting compound ID appears on the reactant side.

- [render_lowest_step_gif.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/render_lowest_step_gif.py:1)
  Renders the lowest-barrier elementary step for requested reaction directions to:
  - GIF
  - XYZ trajectory
  - VMD loader script

- [render_interactive_3d.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/render_interactive_3d.py:1)
  Renders requested reaction directions as standalone interactive HTML trajectory viewers with:
  - drag-to-rotate 3D view
  - frame slider
  - play/pause controls

- [render_reaction_common.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/render_reaction_common.py:1)
  Shared helper code for the renderers:
  - reaction-id parsing
  - lowest-barrier elementary-step selection per direction
  - trajectory sampling from spline/path
  - atom/bond presentation helpers

- [example_chemoton_script.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/example_chemoton_script.py:1)
  Example Chemoton workflow script already present in the repo.

## Concepts

### Reaction directions

Directional reaction IDs are written as:

- `reaction_id;0;` for forward
- `reaction_id;1;` for backward

### Compounds vs flasks

- A `Compound` is treated as a single connected molecular aggregate.
- A `Flask` is treated as a multi-molecule aggregate.

Flask constituent-compound IDs are unreliable in this database, so flask contents are inferred from the centroid structure’s `masm_cbor_graph`.

### Accessibility logic

Current accessibility is based on:

- barrier cutoff only
- SMILES-based chemistry reachability, not strict starting aggregate ID matching

Starting aggregates are converted to constituent SMILES first. A compound or flask is considered reachable if all of its constituent SMILES are already present in the reachable SMILES pool.

Reactions that are only trivial flask/compound regroupings are excluded if:

- at least one side contains a flask, and
- the full multiset of constituent SMILES is identical on both sides

An optional reactant-side molecule-count limit can also be set in `user_input_config.py`.
When `ACCESSIBILITY_DEFAULTS["max_reactant_molecules"]` is an integer, a directional reaction is excluded if its reactant side expands to more molecules than that limit after flask decomposition.
A flask counts as the number of constituent molecules recovered from its centroid `masm_cbor_graph`.
Set the value to `None` to disable this filter.

An optional `Delta E` limit can also be set in `user_input_config.py`.
When `ACCESSIBILITY_DEFAULTS["max_delta_e_kj_per_mol"]` is a number, a directional reaction is excluded if its `Delta E` is greater than or equal to that value in kJ/mol.
Set the value to `None` to disable this filter.

Two optional thermodynamic/kinetic inputs can also be set in `user_input_config.py`.
When `ACCESSIBILITY_DEFAULTS["minimum_rate_constant_s^-1"]` is a positive number, the maximum barrier is approximated as `-R*T*ln(h*k/(k_B*T))` in `kJ/mol`.
When `ACCESSIBILITY_DEFAULTS["minimum_equilibrium_constant"]` is a positive number, the maximum product-reactant energy difference is approximated as `-R*T*ln(K)` in `kJ/mol`.
If both the direct energy cutoff and the corresponding `k` or `K` input are set, the code prints a warning and uses the `k`/`K`-derived cutoff.

Two optional second-stage accessibility screens can also be set in `user_input_config.py`.
When `ACCESSIBILITY_DEFAULTS["rotamer_filter"]` is `True`, accessible directions are grouped by reactant-side `masm_cbor_graph` connectivity and only the lowest-barrier member of each rotamer-equivalent group is kept, with exact barrier ties preserved.
When `ACCESSIBILITY_DEFAULTS["competition_filter"]` is greater than zero, an accessible direction is removed if another accessible direction in the same reactant group is lower in barrier by more than that threshold in kJ/mol. The reactant group is defined by exact reactant aggregate IDs when `rotamer_filter` is off, and by reactant-side `masm_cbor_graph` connectivity when `rotamer_filter` is on.
When `ACCESSIBILITY_DEFAULTS["minimum_competitive_rate_ratio"]` is set, the competition energy gap is approximated as `-R*T*ln(ratio)` in `kJ/mol`, where the allowed ratio is `0 < ratio <= 1` and the ratio means `k_reaction / k_best`.
If both `competition_filter` and `minimum_competitive_rate_ratio` are set, the code prints a warning and uses the rate-ratio-derived competition gap.
After either second-stage filter is applied, reachability is recomputed and any reactions or molecules no longer reachable are removed from the final outputs.

## Requirements

These scripts assume:

- access to the Chemoton MongoDB database
- the `chemoton` conda environment
- installed SCINE Python packages
- `matplotlib` and `Pillow` for GIF generation
- `plotly` for standalone interactive HTML rendering
- `vmd-python` is available in the environment, but the renderer currently uses Python/matplotlib for GIF creation and writes a VMD script for inspection

## Accessible Network

This is the propagation-based analysis.

It:

1. starts from a set of starting compound IDs
2. evaluates all reactions in the database
3. accepts a reaction direction if:
   - its barrier is below the cutoff
   - all reactants are chemically reachable
   - it is not a trivial flask regrouping
   - and its reactant side does not exceed `ACCESSIBILITY_DEFAULTS["max_reactant_molecules"]` when that limit is set
   - and its `Delta E` is below the effective `Delta E` cutoff, taken from `minimum_equilibrium_constant` when set or `max_delta_e_kj_per_mol` otherwise
4. adds the products’ constituent SMILES to the reachable chemistry pool
5. repeats until no new reachable aggregates appear
6. if `ACCESSIBILITY_DEFAULTS["rotamer_filter"]` or `ACCESSIBILITY_DEFAULTS["competition_filter"]` is active, prunes the initially accessible directions, recomputes reachability, and removes any downstream reactions and molecules that are no longer accessible

Run:

```bash
python accessible_network.py
```

Override starting IDs:

```bash
python accessible_network.py \
  --starting-id 69c1abfbdf7e55117102846a \
  --starting-id 69c290cd54afd82e0701e3f3
```

Useful options:

```bash
python accessible_network.py \
  --db-name ch3sh-ch2sh \
  --ip 172.31.55.219 \
  --port 27017 \
  --energy-type electronic_energy \
  --max-barrier-kj-per-mol 150 \
  --molecule-output molecules.csv \
  --reaction-output reactions.csv
```

By default, the molecule table is filtered to compounds with multiplicity 1 or 2. To include all multiplicities:

```bash
python accessible_network.py --compound-multiplicity-mode all
```

Add `--progress` to show progress bars for the database-heavy phases.
Add `--jobs N` to parallelize reaction evaluation and compound indexing. The default is `--jobs 1`.
Edit `ACCESSIBILITY_DEFAULTS["max_reactant_molecules"]` in `user_input_config.py` to limit accepted reactant sides by molecule count. This restriction does not apply to product sides.
Edit `ACCESSIBILITY_DEFAULTS["minimum_rate_constant_s^-1"]` and `ACCESSIBILITY_DEFAULTS["minimum_equilibrium_constant"]` in `user_input_config.py` to drive the barrier and `Delta E` cutoffs through the approximate `k` and `K` formulas. The older `max_barrier_kj_per_mol` and `max_delta_e_kj_per_mol` settings remain as fallback inputs.
Edit `ACCESSIBILITY_DEFAULTS["rotamer_filter"]`, `ACCESSIBILITY_DEFAULTS["competition_filter"]`, and `ACCESSIBILITY_DEFAULTS["minimum_competitive_rate_ratio"]` in `user_input_config.py` to control the secondary pruning pass on initially accessible reactions.

Outputs:

- molecule table:
  - `CompoundId`
  - `SMILES`
  - `ChemicalFormula`
  - `Multiplicity`
  - `Energy (Eh)`

  Flask rows are not written directly. Reachable flasks are decomposed from the centroid `masm_cbor_graph`, each constituent is resolved back to matching compound IDs through the same graph-derived SMILES convention, and unresolved constituents are written with `NO ID`.

- reaction table:
  - `ReactionId`
  - `Reaction`
  - `Chemical Equation`
  - `Structured Chemical Equation`
  - `Barrier (kJ/mol)`
  - `Delta E (kJ/mol)`

## Accessible Subgraph

This is the induced low-barrier subgraph view.

It:

1. computes the reachable aggregate set using the same propagation logic as `accessible_network.py`
2. collects all low-barrier reaction directions whose reactants and products are chemically reachable within that induced chemistry
3. if `ACCESSIBILITY_DEFAULTS["rotamer_filter"]` or `ACCESSIBILITY_DEFAULTS["competition_filter"]` is active, applies the same secondary pruning pass and recomputes the final reachable aggregates before writing reactions and molecules

Run:

```bash
python accessible_subgraph.py
```

Outputs by default:

- `accessible_subgraph_molecules.csv`
- `accessible_subgraph_reactions.csv`

`accessible_subgraph.py` supports the same `--compound-multiplicity-mode {singlet-doublet,all}` option for the molecule output.
It also supports `--progress`.
It also supports `--jobs N`, with default `--jobs 1`.
It also honors `ACCESSIBILITY_DEFAULTS["max_reactant_molecules"]` from `user_input_config.py`.
It also honors the effective barrier and `Delta E` cutoffs resolved from `minimum_rate_constant_s^-1` / `minimum_equilibrium_constant`, with `max_barrier_kj_per_mol` / `max_delta_e_kj_per_mol` as fallback inputs.
It also honors `ACCESSIBILITY_DEFAULTS["rotamer_filter"]`, `ACCESSIBILITY_DEFAULTS["competition_filter"]`, and `ACCESSIBILITY_DEFAULTS["minimum_competitive_rate_ratio"]` from `user_input_config.py`.

## Catalytic Cycles

This finds catalyst-regenerating cyclic paths inside the same accessible low-barrier subgraph used by `accessible_subgraph.py`.

It:

1. computes the reachable aggregate set using the same propagation logic as `accessible_network.py`
2. collects all low-barrier reaction directions within that reachable chemistry
3. applies the same optional secondary rotamer/competition filters
4. searches for directed paths where a requested catalyst aggregate ID appears as a reactant, moves through product aggregate IDs as catalyst-state intermediates, and is regenerated within `--max-steps`
5. filters cycles by optional net-consumed reactants, net-produced products, and species that must or must not appear anywhere in the path

Run:

```bash
python catalytic_cycles.py \
  --catalyst-id 69c1abfbdf7e55117102846a
```

Or set the defaults in `user_input_config.py`:

```python
CATALYTIC_CYCLE_DEFAULTS = {
    "cycle_output": "catalytic_cycles.csv",
    "max_steps": 8,
    "catalyst_ids": ["69c2a6b754afd82e0701e3ff"],
    "reactant_ids": ["69c1abfbdf7e55117102846a", "69c290cd54afd82e0701e3f3"],
    "product_ids": ["69c2e5d854afd82e0701e43d"],
    "include_species_ids": [],
    "exclude_species_ids": [],
}
```

Require starting reactants and a product:

```bash
python catalytic_cycles.py \
  --starting-id 69c1abfbdf7e55117102846a \
  --starting-id 69c290cd54afd82e0701e3f3 \
  --starting-id 69c2a6b754afd82e0701e3ff \
  --catalyst-id 69c2a6b754afd82e0701e3ff \
  --reactant-id 69c1abfbdf7e55117102846a \
  --reactant-id 69c290cd54afd82e0701e3f3 \
  --product-id 69c2e5d854afd82e0701e43d
```

Useful path filters:

```bash
python catalytic_cycles.py \
  --catalyst-id 69c2a6b754afd82e0701e3ff \
  --include-species-id 69c50ae865f50a833301e4d5 \
  --exclude-species-id 69c3e55c65f50a833301e42d \
  --max-steps 10 \
  --cycle-output catalytic_cycles.csv
```

Outputs by default:

- `catalytic_cycles.csv`

The cycle table includes:

- `CycleId`
- `CatalystId`
- `StepCount`
- `ReactionPath`
- `TrackedSpeciesPath`
- `Net Reactants`
- `Net Products`
- `Included Species`
- `Max Barrier (kJ/mol)`
- `Sum Delta E (kJ/mol)`
- `Chemical Equations`

`catalytic_cycles.py` matches species by exact aggregate ID. A reported cycle must have zero net stoichiometry for the catalyst ID. `--reactant-id` entries must be net consumed, `--product-id` entries must be net produced, `--include-species-id` entries must appear anywhere in the path, and `--exclude-species-id` entries must not appear anywhere in the path.
Command-line species options override the corresponding `CATALYTIC_CYCLE_DEFAULTS` list. At least one catalyst ID must be provided either by `--catalyst-id` or `CATALYTIC_CYCLE_DEFAULTS["catalyst_ids"]`.

## Starting Reactant Reactions

This lists directional reactions where at least one of the provided starting compound IDs appears directly on the reactant side.

Run:

```bash
python starting_reactant_reactions.py
```

Override starting IDs:

```bash
python starting_reactant_reactions.py \
  --starting-id 69c1abfbdf7e55117102846a \
  --starting-id 69c290cd54afd82e0701e3f3
```

Useful options:

```bash
python starting_reactant_reactions.py \
  --db-name ch3sh-ch2sh \
  --ip 172.31.55.219 \
  --port 27017 \
  --max-barrier-kj-per-mol 150 \
  --reaction-output starting_reactant_reactions.csv
```

`starting_reactant_reactions.py` also honors `ACCESSIBILITY_DEFAULTS["max_reactant_molecules"]` from `user_input_config.py`.
`starting_reactant_reactions.py` also honors the same resolved barrier and `Delta E` cutoffs from `user_input_config.py`.
`starting_reactant_reactions.py` also applies conformer screening from `ACCESSIBILITY_DEFAULTS["rotamer_filter"]`.
`starting_reactant_reactions.py` also applies the resolved competing-reaction filter from `ACCESSIBILITY_DEFAULTS["competition_filter"]` or `ACCESSIBILITY_DEFAULTS["minimum_competitive_rate_ratio"]`.

## Rendering Reaction Trajectories

`render_lowest_step_gif.py` renders every requested directional reaction using the lowest-barrier elementary step for that requested direction.

### Accepted inputs

You can provide:

- a text file with one directional reaction ID per line
- one or more `--reaction-id` flags
- or both

Examples of valid tokens:

- `69c50ae865f50a833301e4d5;0;`
- `69c50ae865f50a833301e4d5;1;`
- bare `69c50ae865f50a833301e4d5`

A bare reaction ID expands to both directions.

Important:

- if you pass `;0;` or `;1;` on the shell command line, quote it

Example:

```bash
python render_lowest_step_gif.py --reaction-id '69c50ae865f50a833301e4d5;1;'
```

### File input

Example `reaction_ids.txt`:

```text
69c50ae865f50a833301e4d5;1;
69c3e55c65f50a833301e42d;0;
```

Run:

```bash
python render_lowest_step_gif.py reaction_ids.txt
```

Add `--progress` to show progress bars across requested reactions and sampled frames.
Add `--jobs N` to render multiple requested reactions in parallel. The default is `--jobs 1`.

### Flag input

Render both directions for a reaction:

```bash
python render_lowest_step_gif.py --reaction-id 69c50ae865f50a833301e4d5
```

Render a specific direction:

```bash
python render_lowest_step_gif.py --reaction-id '69c50ae865f50a833301e4d5;0;'
```

Multiple flags:

```bash
python render_lowest_step_gif.py \
  --reaction-id 69c50ae865f50a833301e4d5 \
  --reaction-id '69c3e55c65f50a833301e42d;0;'
```

### Renderer behavior

For each requested direction, the script:

1. loads the corresponding database reaction
2. inspects all elementary steps attached to that reaction
3. computes the barrier for the requested direction for each step
4. chooses the lowest-barrier step that has renderable data (`spline` or `path`)
5. samples frames
6. writes:
   - GIF
   - XYZ trajectory
   - VMD loader script

The GIF renderer supports:

- explicit `2d` or `3d` render mode
- configurable initial 3D camera angle
- optional automatic fixed-view camera selection for 3D renders
- optional camera rotation across the animation

Default output directory:

- `rendered_reactions/`

For a request like `69c2f4b065f50a833301e3bd;0;`, the outputs are:

- `rendered_reactions/69c2f4b065f50a833301e3bd_0_3d_high.gif` for `--render-dim 3d --render-quality high`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0_3d_low.gif` for `--render-dim 3d --render-quality low`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0_2d_high.gif` for `--render-dim 2d --render-quality high`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0.xyz`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0.vmd.tcl`

### Renderer options

```bash
python render_lowest_step_gif.py reaction_ids.txt \
  --frames 48 \
  --fps 10 \
  --render-dim 3d \
  --auto-view \
  --render-quality high \
  --view-elev 18 \
  --view-azim 38 \
  --rotate-azim-deg 120 \
  --output-dir rendered_reactions
```

Notes:

- the final GIF playback currently runs at two-thirds of the nominal `--fps` value
- the script uses database spline/path data to construct the trajectory
- GIF rendering is done with `matplotlib`
- the generated `.vmd.tcl` file is for loading the exported XYZ movie into VMD for inspection
- `--render-dim 3d` is the default
- `--render-dim 2d` renders the same trajectory projected onto the XY plane
- `--render-quality high` is the default
- `--render-quality low` restores the earlier simple marker/line appearance
- `--auto-view` chooses a fixed 3D camera angle that tries to make bond changes visible while keeping the full system in frame
- `--view-elev` and `--view-azim` set the starting camera angle for 3D GIFs
- if `--auto-view` is used, `--view-elev` and `--view-azim` become fallback defaults for degenerate cases
- `--rotate-azim-deg` and `--rotate-elev-deg` apply a smooth camera sweep over the full animation
- if `--auto-view` is combined with rotation, the sweep is applied around the auto-selected base view

## Interactive 3D Rendering

`render_interactive_3d.py` renders requested directional reactions as standalone interactive HTML files.

It uses the same selection logic as the GIF renderer:

1. parse requested directional reaction IDs from a file and/or `--reaction-id`
2. expand a bare reaction ID to both directions
3. inspect all elementary steps for each requested direction
4. select the lowest-barrier step for that direction that has renderable data
5. sample spline/path frames through the same direction-corrected shared renderer path used by GIF rendering
6. write one interactive HTML file per requested direction

### Interaction model

Each HTML output provides:

- 3D drag-to-rotate camera controls
- frame slider
- play button
- pause button

The scene bounds are fixed across all frames so the trajectory does not rescale while animating.

### Input examples

File input:

```bash
python render_interactive_3d.py reaction_ids.txt
```

Add `--progress` to show progress bars across requested reactions and generated frames.
Add `--jobs N` to render multiple requested reactions in parallel. The default is `--jobs 1`.

Specific direction:

```bash
python render_interactive_3d.py --reaction-id '69c50ae865f50a833301e4d5;1;'
```

Bare reaction ID expands to both directions:

```bash
python render_interactive_3d.py --reaction-id 69c50ae865f50a833301e4d5
```

Multiple requests:

```bash
python render_interactive_3d.py \
  --reaction-id 69c50ae865f50a833301e4d5 \
  --reaction-id '69c3e55c65f50a833301e42d;0;'
```

### Outputs

Default output directory:

- `interactive_reactions/`

For a request like `69c2f4b065f50a833301e3bd;0;`, the output is:

- `interactive_reactions/69c2f4b065f50a833301e3bd_0_high.html` by default
- `interactive_reactions/69c2f4b065f50a833301e3bd_0_low.html` for `--render-quality low`

### Interactive renderer options

```bash
python render_interactive_3d.py reaction_ids.txt \
  --frames 48 \
  --render-quality high \
  --output-dir interactive_reactions
```

Notes:

- output is standalone HTML, not a notebook widget
- interaction is handled by Plotly in the browser
- no separate web server is required
- the GIF renderer remains available for non-interactive exports
- interactive rendering uses the same reaction-direction frame orientation logic as GIF rendering, so the first frame matches the requested reactants and the last frame matches the requested products when endpoint matching succeeds
- `--render-quality high` uses mesh-based spheres and cylindrical bonds
- `--render-quality low` uses the earlier simple Plotly marker/line style

## Python usage

The shared module can be imported directly:

```python
from chemoton_accessibility_core import (
    DatabaseManager,
    Model,
    AggregateCache,
    ReactionEvaluator,
    screen_network,
    collect_accessible_subgraph_reactions,
)
```

Typical workflow:

1. create `DatabaseManager`
2. call `loadCollections()`
3. construct `Model`
4. build `AggregateCache`
5. evaluate reactions with `ReactionEvaluator`
6. run `screen_network(...)` or `collect_accessible_subgraph_reactions(...)`

## Sanity checks

Basic syntax checks:

```bash
python -m py_compile \
  chemoton_accessibility_core.py \
  accessible_network.py \
  accessible_subgraph.py \
  render_lowest_step_gif.py
```

## Current limitations

- flask constituent compound IDs are not reliably populated in this database, so flask composition is inferred from structure graphs
- the GIF renderer writes VMD loader scripts, but image rendering itself is currently handled in Python rather than directly via headless VMD snapshot output
- the renderer chooses the lowest-barrier renderable elementary step for a requested direction, which may differ from the step selected elsewhere if a lower-barrier step has no usable spline/path
