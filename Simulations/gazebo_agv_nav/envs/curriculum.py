"""The curriculum ladder, in one place.

Lives here rather than in `scripts/curriculum_train.py` because the gate test
and the evaluation scripts have to use the identical ladder -- when the stage
table was duplicated across scripts they were free to drift, and a gate test
run against different distances than training uses proves nothing.

Each stage samples the start-goal distance from a BAND [min_goal_dist,
max_goal_dist]. Bands, not just ceilings, because the old ceiling-only sampler
produced a ladder that barely climbed: measured over the map pool, the stages
capped at 12/16/None all averaged 5.8-6.7 m, the nominally hardest full-random
stage came out easier than the one before it, and 19% of s1 episodes began
inside the goal radius -- already solved (HANDOFF §14). Raising the floor with
the ceiling makes each stage strictly harder than the last and keeps every
episode a real navigation problem.

max_goal_dist=None on the last stage means "no ceiling"; with a floor of 18 m
it is now genuinely the hardest stage rather than merely unbounded.

`max_steps` is roughly 2.5x the straight-line minimum for the band's ceiling,
(dist - goal_radius) / 0.11 m per step, which leaves room for turning: the
scripted controller needed about 1.5x in practice.
"""

STAGES = [
    # s0 exists so learning can start at all. Success at s1 (2-5 m) is close to
    # unreachable by chance -- a random policy scored 0/8 there -- so SAC gets
    # no first success to bootstrap from and 16 of 17 runs settled into a
    # zero-mean, maximum-entropy "stand still" policy instead. s0's trips are
    # short enough (0.1-1.3 m of closing distance, against ~0.85 m of random-walk
    # displacement over its step budget) that chance successes are frequent.
    # The floor stays above goal_radius=1.2 so no episode begins already solved:
    # that was the flaw in the pre-band sampler, which inflated the reported
    # metric. Bootstrapping and honest measurement are separable, and this
    # separates them.
    dict(name="s0", min_goal_dist=1.3,  max_goal_dist=2.5,  max_steps=60,  timesteps=40_000),
    dict(name="s1", min_goal_dist=2.0,  max_goal_dist=5.0,  max_steps=90,  timesteps=60_000),
    dict(name="s2", min_goal_dist=5.0,  max_goal_dist=9.0,  max_steps=180, timesteps=120_000),
    dict(name="s3", min_goal_dist=9.0,  max_goal_dist=13.0, max_steps=270, timesteps=150_000),
    dict(name="s4", min_goal_dist=13.0, max_goal_dist=18.0, max_steps=380, timesteps=200_000),
    dict(name="s5", min_goal_dist=18.0, max_goal_dist=None, max_steps=600, timesteps=300_000),
]

# world/pool geometry these stages assume
GRID_SIZE = 24
N_OBSTACLE_PAIRS = 14
N_OBSTACLE_SLOTS = 2 * N_OBSTACLE_PAIRS
N_MAPS = 24


def stage_by_name(name: str) -> dict:
    for s in STAGES:
        if s["name"] == name:
            return s
    raise KeyError(f"unknown stage {name!r}; have {[s['name'] for s in STAGES]}")
