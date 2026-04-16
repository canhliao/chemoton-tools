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
  - default output filenames
  - renderer defaults

- [accessible_network.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_network.py:1)
  CLI entry point for the propagation-based accessibility analysis.

- [accessible_subgraph.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_subgraph.py:1)
  CLI entry point for collecting the low-barrier reaction subgraph induced by the reachable chemistry.

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
4. adds the products’ constituent SMILES to the reachable chemistry pool
5. repeats until no new reachable aggregates appear

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
- `--view-elev` and `--view-azim` set the starting camera angle for 3D GIFs
- `--rotate-azim-deg` and `--rotate-elev-deg` apply a smooth camera sweep over the full animation

## Interactive 3D Rendering

`render_interactive_3d.py` renders requested directional reactions as standalone interactive HTML files.

It uses the same selection logic as the GIF renderer:

1. parse requested directional reaction IDs from a file and/or `--reaction-id`
2. expand a bare reaction ID to both directions
3. inspect all elementary steps for each requested direction
4. select the lowest-barrier step for that direction that has renderable data
5. sample spline/path frames
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
