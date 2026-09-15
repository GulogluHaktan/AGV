# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the working resources folder for an academic paper project (not yet a software
codebase — no code exists here yet). It currently contains a single source document:

- `Mission/Symmetry_Makale_Fikirleri.docx` — a Turkish-language report proposing 5 candidate
  article ideas for the journal *Symmetry* (MDPI, ISSN 2073-8994), "Computer" section. Each
  idea includes research questions, contributions, required materials/tools, method flow,
  expected results, and a figure/table presentation plan.

**The team has selected Paper Idea 5** from that report as the active project:

> **"Symmetry-Exploiting Deep Reinforcement Learning for Sample-Efficient AGV Path Planning
> and Sim-to-Real Transfer"**

Core idea: encode the reflective/rotational symmetry of a warehouse/corridor navigation
environment into a DRL policy (via symmetric data augmentation and/or an E(2)-steerable
equivariant policy network) and compare it against a symmetry-agnostic PPO/SAC baseline on:
(a) sample efficiency (episodes to reach target success rate), (b) generalization to unseen
mirrored/rotated map layouts, and (c) sim-to-real transfer gap, with the gap partly attributed
to broken symmetry assumptions (asymmetric sensor noise, uneven floor friction).

Read `Mission/Symmetry_Makale_Fikirleri.docx` (Makale Fikri 5 section) before doing
substantive planning or implementation work — it contains the full research questions,
materials list, and method steps that should drive the project below.

## Project workflow (current plan)

The project proceeds in these phases, in order:

1. **Literature review** — survey prior work on symmetry-aware/equivariant RL, AGV/AMR path
   planning, and sim-to-real transfer for mobile robots, to ground the paper's contribution
   claims and related-work section.
2. **Simulation build** — implement the navigation environment (symmetric and asymmetric
   corridor/warehouse layouts) in the selected simulator.
3. **RL training** — train and compare the three conditions: baseline PPO/SAC, symmetric
   data-augmented PPO, and an equivariant (E(2)-steerable) policy network.
4. **Results generation** — produce the result tables and figures needed for the paper
   (learning curves, generalization heatmaps on mirrored/rotated maps, sim-to-real success
   rates with error decomposition, hyperparameter/statistical-significance tables).

No simulator, RL library, or code structure has been committed to this repo yet — those
choices should follow the materials list in the source document (PyBullet, Gazebo, or NVIDIA
Isaac Sim; Stable-Baselines3/RLlib; `escnn` for the equivariant policy network; a
Gymnasium-style interface) unless the user directs otherwise.

## Working in this repo

- There is no build/lint/test tooling yet because there is no code yet. Do not invent or
  assume tooling — check what's actually present before suggesting commands.
- This is not a git repository. Confirm with the user before assuming version control
  conventions (branches, commits) apply.
- For the literature-review phase, the `academic-research-skills` plugin and its
  `deep-research` / `literatur-taramasi` skills are available and appropriate — use them
  rather than ad hoc web searches when the user asks for a literature survey.
- For eventual paper drafting (once results exist), `makale-latex-yazici` or
  `makale-docx-yazici` are the appropriate skills depending on the deliverable format the
  target venue/template requires.
