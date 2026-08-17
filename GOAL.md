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

## Next

Move rover navigation onto the same 3D-token pathway that already works for
question answering, so the whole system reasons over the semantic 3D map rather
than over notes derived from it.
