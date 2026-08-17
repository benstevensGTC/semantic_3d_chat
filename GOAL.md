# The goal

Build a **fully local** system in which Gemma turns complete visual scans of a
3D space into a **persistent semantic 3D representation**, and then **reasons
over that representation directly** — to answer spatial questions about the
room, and to drive a rover by choosing its own coordinates, headings, and when
to stop.

## What "directly" means, precisely

The language model's answer must be caused by the 3D semantic field itself, fed
in as continuous embeddings. Specifically **not**:

- a textual scene graph or object list handed to the model,
- a caption or description of the room,
- a single rendered picture standing in for the space,
- an object detector feeding labels into a planner,
- deterministic code choosing where the robot goes.

The test is causal, not rhetorical: swap the scene for another room's, scramble
its spatial layout, or zero it, and the answers must change accordingly. An
answer that survives those controls was not grounded in the scene.

## Constraints

- Everything runs on this machine. No cloud inference.
- On-device training kept to a minimum; prefer zero, and only train if there is
  no other way.
- The user authors the 3D space themselves.
- Perception is visual: the model looks at the room. Author-supplied names for
  the furniture never reach it.
- Deterministic code may act as the robot's body — exact motion, collision
  rejection, safety — but not as its planner or its semantics.

## Status

**Achieved (zero training).** `scripts/lens_ask_3d.py` fuses 30 RGB-D views
through Gemma's vision encoder into a semantic voxel map, pools it into a
bird's-eye grid of 256 tokens that live in Gemma's own vision-projector output
space, and injects them through the decoder's native image pathway. Asked to
list what is in the room, Gemma answers from the 3D field alone — correctly
naming the teal bed, television, wooden table, cylindrical lamp and bookshelf —
with no scene graph, object list or photograph in the prompt. Zeroing the
tokens produces "I cannot discern any specific objects"; scrambling their layout
degrades the answer to generic nouns without colours or shapes.

**Working but weaker.** Rover navigation currently reasons over a *text*
rendering of the metric scene graph, which is the notes-shaped compromise this
goal rejects. It reaches 5/7 objects with Gemma, and solves the hard
obstacle-detour case with a larger local model plus visit memory.

**Measured dead end.** Training a control head on the soft scene prefix (V15)
does not produce grounded behaviour: the geometry is present in the tokens
(R² = 0.99 for position, 86% room identity) but the readout discards it.

**Measured limit of the 3D pathway.** Reading a *position* out of the same
tokens does not work. Asked which grid cell each object occupies, Gemma scores
14% within 1 m — identical to the scrambled and zeroed controls, and at the
10.5% random baseline
([evidence](reports/gemma4/metrics/spatial_lens_studio_locate3d.json)). So the
field supports "what is in this room" but not "where exactly", and navigation
was deliberately *not* moved onto it, because that would have been a regression.

This converges with the V15 result from the other direction. There, a probe
showed position is present in the scene tokens (R² = 0.99) while the control
head could not use it. Here, a frozen decoder can pool the same tokens into an
accurate description but cannot index them spatially. Both say the same thing:
**the geometry is in the representation; what is missing is a readout trained to
address it.**

## Next

Two candidates, in order of expected value:

1. Train a small readout that addresses scene tokens by position -- a
   cross-attention head over the token grid, supervised on "which cell holds
   this object". The per-room tokens cache, so this is minutes of training, not
   hours, and it is the one thing both experiments point at.
2. Failing that, keep navigation on the metric scene graph, which is itself
   derived from the 3D map, and be explicit that the LLM reads notes for
   metric work and the field for semantic work.
