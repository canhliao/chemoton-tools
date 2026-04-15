# Chemoton Tools

Utilities for querying a SCINE/Chemoton reaction-network database, identifying accessible aggregates and reactions from a starting set, and rendering elementary-step trajectories to GIFs.

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

- [accessible_network.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_network.py:1)
  CLI entry point for the propagation-based accessibility analysis.

- [accessible_subgraph.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/accessible_subgraph.py:1)
  CLI entry point for collecting the low-barrier reaction subgraph induced by the reachable chemistry.

- [render_lowest_step_gif.py](/scratch/caliao/astrochemistry/ch3sh-ch2sh/chemoton-tools/render_lowest_step_gif.py:1)
  Renders the lowest-barrier elementary step for requested reaction directions to:
  - GIF
  - XYZ trajectory
  - VMD loader script

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
  --molecule-output molecules.txt \
  --reaction-output reactions.txt
```

Outputs:

- molecule table:
  - `AggregateId`
  - `Type`
  - `SMILES`
  - `Multiplicity`

- reaction table:
  - `ReactionId`
  - `Reaction`
  - `Barrier (kJ/mol)`
  - `LHS Energy (Eh)`
  - `RHS Energy (Eh)`

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

- `accessible_subgraph_molecules.txt`
- `accessible_subgraph_reactions.txt`

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

Default output directory:

- `rendered_reactions/`

For a request like `69c2f4b065f50a833301e3bd;0;`, the outputs are:

- `rendered_reactions/69c2f4b065f50a833301e3bd_0.gif`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0.xyz`
- `rendered_reactions/69c2f4b065f50a833301e3bd_0.vmd.tcl`

### Renderer options

```bash
python render_lowest_step_gif.py reaction_ids.txt \
  --frames 48 \
  --fps 10 \
  --output-dir rendered_reactions
```

Notes:

- the final GIF playback is intentionally slowed to one-third of the nominal `--fps` value
- the script uses database spline/path data to construct the trajectory
- GIF rendering is done with `matplotlib`
- the generated `.vmd.tcl` file is for loading the exported XYZ movie into VMD for inspection

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
