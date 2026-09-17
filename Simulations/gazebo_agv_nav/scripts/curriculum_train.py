"""Single coherent curriculum run: max_goal_dist AND max_steps both scale up
together across stages (the earlier ad-hoc runs kept max_steps fixed at 120,
which silently capped every stage past dist~8 -- ep_len_mean sitting at ~119
the whole time was the tell: episodes were timing out on the clock, not on
a bad policy). Evaluates at the end of every stage, at that stage's own
difficulty, so progress is visible immediately instead of only at the end.
"""
import os, sys, argparse, json
from collections import deque
# repo root for `envs` imports: /workspace inside the Docker image, or the
# project directory itself when running natively
sys.path.insert(0, os.environ.get("AGV_WORKSPACE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from envs.curriculum import (STAGES, GRID_SIZE, N_MAPS, N_OBSTACLE_PAIRS,
                             N_OBSTACLE_SLOTS)
from envs.map_generator import make_map_pool
from envs.gazebo_agv_env import GazeboAGVEnv
from envs.wrappers import SymmetricAugmentationWrapper


class SuccessRateCallback(BaseCallback):
    """ep_rew_mean/ep_len_mean alone can't tell you whether episodes are
    actually reaching the goal (terminated) vs just timing out closer than
    they started (truncated) -- this prints a rolling reached-vs-timeout rate
    so that's directly visible instead of inferred."""
    def __init__(self, log_every=2000, window=50):
        super().__init__()
        self.log_every = log_every
        self.window = window
        self.recent = deque(maxlen=window)
        self.actions = deque(maxlen=500)  # (linear_v, angular_v) raw actions

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                truncated = info.get("TimeLimit.truncated", False)
                self.recent.append(0 if truncated else 1)
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.atleast_2d(actions):
                self.actions.append((float(a[0]), float(a[1])))
        if self.recent and self.num_timesteps % self.log_every == 0:
            rate = sum(self.recent) / len(self.recent)
            # timestep on the line so learning curves read straight out of the
            # log; without it a parser has to bracket each line between the
            # surrounding rollout blocks to recover x
            print(f"[success-rate] t={self.num_timesteps} "
                  f"last {len(self.recent)} episodes: "
                  f"{rate:.1%} reached goal", flush=True)
            if self.actions:
                arr = np.array(self.actions)
                lin, ang = arr[:, 0], arr[:, 1]
                spin_frac = float(np.mean((np.abs(ang) > 0.5) & (np.abs(lin) < 0.2)))
                print(f"[actions] last {len(arr)} steps: "
                      f"linear mean={lin.mean():+.2f} std={lin.std():.2f} | "
                      f"angular mean={ang.mean():+.2f} std={ang.std():.2f} | "
                      f"spin-in-place frac={spin_frac:.1%}", flush=True)
        return True



def make_env(pool, grid_size, seed, stage, arm, collision_coef=0.05,
             include_position=True, reward_scale=1.0):
    env = GazeboAGVEnv(grid=pool, grid_size=grid_size, seed=seed,
                        min_goal_dist=stage["min_goal_dist"],
                        max_goal_dist=stage["max_goal_dist"],
                        max_steps=stage["max_steps"],
                        collision_coef=collision_coef,
                        include_position=include_position,
                        reward_scale=reward_scale,
                        n_obstacle_slots=N_OBSTACLE_SLOTS)
    if arm == "symmetric_augmentation":
        env = SymmetricAugmentationWrapper(env, seed=seed)
    return env


def evaluate(model, pool, grid_size, seed, stage, n_episodes=20,
             collision_coef=0.05, include_position=True):
    """Returns (deterministic_sr, stochastic_sr). Both are reported because the
    deterministic mean action lags well behind the stochastic behavior early in
    training (eval_diag.py on sac_baseline_v2_s1: det 5% vs stoch 25% — the
    train-vs-eval gap of HANDOFF §6.9 is mean-action miscalibration, not a
    metric artifact), and the stochastic number is the one comparable to the
    train-time rolling success rate."""
    env = GazeboAGVEnv(grid=pool, grid_size=grid_size, seed=seed,
                        min_goal_dist=stage["min_goal_dist"],
                        max_goal_dist=stage["max_goal_dist"],
                        max_steps=stage["max_steps"],
                        collision_coef=collision_coef,
                        include_position=include_position,
                        n_obstacle_slots=N_OBSTACLE_SLOTS)
    rates = []
    for deterministic in (True, False):
        successes = 0
        for ep in range(n_episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
            successes += int(terminated)
        rates.append(successes / n_episodes)
    env.close()
    return rates[0], rates[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "symmetric_augmentation", "equivariant"], default="baseline")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    ap.add_argument("--grid_size", type=int, default=GRID_SIZE)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_maps", type=int, default=N_MAPS,
                    help="canonical-orientation training layouts; the env samples "
                         "one per episode. A single fixed map makes the occupancy "
                         "observation a constant, which is exactly the input both "
                         "symmetry arms operate on -- see HANDOFF 12b.")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--out_prefix", default="/workspace/curr2")
    ap.add_argument("--start_stage", type=int, default=0)
    ap.add_argument("--end_stage", type=int, default=len(STAGES),
                    help="exclusive; --end_stage 1 runs s1 only (for ablations)")
    ap.add_argument("--collision_coef", type=float, default=0.05,
                    help="obstacle-proximity penalty weight. Early in training "
                         "progress is ~0, so this term dominates and freezing "
                         "can be locally optimal -- 0 disables it.")
    ap.add_argument("--ent_coef", default="0.02",
                    help="SAC entropy coefficient; 'auto' tunes it")
    ap.add_argument("--reward_scale", type=float, default=1.0,
                    help="multiplies the reward; sets the task's weight against "
                         "SAC's entropy bonus")
    ap.add_argument("--no_position", action="store_true",
                    help="ablation: drop the position observation")
    args = ap.parse_args()
    Algo = SAC if args.algo == "sac" else PPO

    # Training distribution: canonical-orientation layouts only. The D4
    # images of these, plus fresh unseen layouts, are held back for the G3
    # generalization evaluation and must never be sampled here.
    pool = make_map_pool(size=args.grid_size, n_maps=args.n_maps,
                     seed=args.seed, n_obstacle_pairs=N_OBSTACLE_PAIRS)

    ckpt = args.resume_from
    results = {}
    model = None
    # only the very first stage of a fresh run resets SB3's step counter
    first_stage = args.resume_from is None
    for idx, stage in enumerate(STAGES):
        if idx < args.start_stage or idx >= args.end_stage:
            continue
        print(f"=== STAGE {stage['name']}: dist {stage['min_goal_dist']}-{stage['max_goal_dist']} "
              f"steps={stage['max_steps']} timesteps={stage['timesteps']} ===", flush=True)
        env = make_env(pool, args.grid_size, args.seed, stage, args.arm,
                       collision_coef=args.collision_coef,
                       include_position=not args.no_position,
                       reward_scale=args.reward_scale)
        policy_kwargs = {}
        if args.arm == "equivariant":
            from envs.equivariant_extractor import D4EquivariantExtractor
            policy_kwargs = {"features_extractor_class": D4EquivariantExtractor,
                              "features_extractor_kwargs": {"grid_feat_dim": 64}}

        if model is not None:
            # Keep ONE model across stages and just swap the env, instead of
            # reloading from the checkpoint. Algo.load() restores the weights
            # but hands back an EMPTY replay buffer, and because
            # reset_num_timesteps=False leaves num_timesteps well above
            # learning_starts, SB3 then began gradient updates on the very
            # first step of the new stage -- drawing 256-sample batches from a
            # buffer holding a handful of transitions. Every stage boundary
            # hammered the network with near-duplicate data, and the damage
            # compounded as the stages got harder: s1 reached 90% (fresh model,
            # so learning_starts was honoured), then s2 degraded and s3/s4
            # collapsed to 0% -- far below the 70% a map-blind scripted
            # controller gets. Keeping the model keeps its buffer, its critic,
            # and its optimizer state. (This is also the real cause of the
            # rise-then-decline pattern section 6.7 attributed to the UTD
            # ratio.)
            model.set_env(env)
            print(f"continuing with in-memory model and replay buffer "
                  f"({model.replay_buffer.size()} transitions)", flush=True)
        elif ckpt:
            model = Algo.load(ckpt, env=env)
            buf = f"{os.path.splitext(ckpt)[0]}_buffer.pkl"
            if os.path.exists(buf):
                model.load_replay_buffer(buf)
                print(f"resumed from {ckpt} + {model.replay_buffer.size()} "
                      f"buffered transitions", flush=True)
            else:
                print(f"resumed from {ckpt} WITHOUT a replay buffer -- the "
                      f"first updates of this stage will overfit a nearly "
                      f"empty buffer; expect a dip", flush=True)
        elif args.algo == "sac":
            # fixed (not "auto") ent_coef: auto-tuning collapsed to ~0.0003
            # within the first stage, killing exploration far too early and
            # leaving the policy stuck around a ~15-20% success rate.
            # gradient_steps=1 (not 4): the same rise-to-~22%-then-decline-to
            # -~10% pattern reproduced under two different ent_coef values,
            # which rules out entropy as the cause -- a high update-to-data
            # (UTD) ratio on a small, still-filling replay buffer is a known
            # SAC failure mode ("primacy bias": the critic overfits early,
            # narrow experience and then can't incorporate new data). Back to
            # the standard 1:1 ratio.
            model = SAC("MultiInputPolicy", env, seed=args.seed, verbose=1,
                        buffer_size=100_000, learning_starts=1000,
                        batch_size=256, train_freq=1, gradient_steps=1,
                        tau=0.005,
                        ent_coef=(args.ent_coef if args.ent_coef == "auto"
                                  else float(args.ent_coef)),
                        policy_kwargs=policy_kwargs)
        else:
            model = PPO("MultiInputPolicy", env, seed=args.seed, verbose=1,
                         n_steps=256, batch_size=64, ent_coef=0.01,
                         policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=first_stage,
                    callback=SuccessRateCallback(log_every=2000))
        first_stage = False
        ckpt = f"{args.out_prefix}_{stage['name']}.zip"
        model.save(ckpt)
        # the buffer is saved alongside so --resume_from can restore it too;
        # without it a resumed run hits the same empty-buffer cliff
        model.save_replay_buffer(f"{args.out_prefix}_{stage['name']}_buffer.pkl")
        env.close()

        sr_det, sr_stoch = evaluate(model, pool, args.grid_size, args.seed,
                                    stage, collision_coef=args.collision_coef,
                                    include_position=not args.no_position,
                       reward_scale=args.reward_scale)
        print(f"=== STAGE {stage['name']} DONE: success_rate={sr_det:.2%} "
              f"stochastic={sr_stoch:.2%} (own difficulty) ===", flush=True)
        results[stage["name"]] = {"deterministic": sr_det, "stochastic": sr_stoch}

    last = STAGES[min(args.end_stage, len(STAGES)) - 1]
    sr_det, sr_stoch = evaluate(model, pool, args.grid_size, args.seed,
                                 last, n_episodes=30,
                                 collision_coef=args.collision_coef,
                       include_position=not args.no_position,
                       reward_scale=args.reward_scale)
    print(f"=== FINAL ({last['name']}) success_rate={sr_det:.2%} "
          f"stochastic={sr_stoch:.2%} ===", flush=True)
    results["final_full_random"] = {"deterministic": sr_det, "stochastic": sr_stoch}
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
