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
from envs.map_generator import symmetric_corridor
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
            print(f"[success-rate] last {len(self.recent)} episodes: "
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

STAGES = [
    dict(name="s1", max_goal_dist=4,    max_steps=80,  timesteps=80_000),
    dict(name="s2", max_goal_dist=8,    max_steps=140, timesteps=120_000),
    dict(name="s3", max_goal_dist=12,   max_steps=200, timesteps=150_000),
    dict(name="s4", max_goal_dist=16,   max_steps=260, timesteps=180_000),
    dict(name="s5", max_goal_dist=None, max_steps=320, timesteps=250_000),
]


def make_env(grid, grid_size, seed, max_goal_dist, max_steps, arm):
    env = GazeboAGVEnv(grid=grid, grid_size=grid_size, seed=seed,
                        max_goal_dist=max_goal_dist, max_steps=max_steps)
    if arm == "symmetric_augmentation":
        env = SymmetricAugmentationWrapper(env, seed=seed)
    return env


def evaluate(model, grid, grid_size, seed, max_goal_dist, max_steps, n_episodes=20):
    """Returns (deterministic_sr, stochastic_sr). Both are reported because the
    deterministic mean action lags well behind the stochastic behavior early in
    training (eval_diag.py on sac_baseline_v2_s1: det 5% vs stoch 25% — the
    train-vs-eval gap of HANDOFF §6.9 is mean-action miscalibration, not a
    metric artifact), and the stochastic number is the one comparable to the
    train-time rolling success rate."""
    env = GazeboAGVEnv(grid=grid, grid_size=grid_size, seed=seed,
                        max_goal_dist=max_goal_dist, max_steps=max_steps)
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
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--out_prefix", default="/workspace/curr2")
    ap.add_argument("--start_stage", type=int, default=0)
    args = ap.parse_args()
    Algo = SAC if args.algo == "sac" else PPO

    rng = np.random.default_rng(args.seed)
    grid = symmetric_corridor(size=args.grid_size, rng=rng)

    ckpt = args.resume_from
    results = {}
    model = None
    for idx, stage in enumerate(STAGES):
        if idx < args.start_stage:
            continue
        print(f"=== STAGE {stage['name']}: dist<={stage['max_goal_dist']} "
              f"steps={stage['max_steps']} timesteps={stage['timesteps']} ===", flush=True)
        env = make_env(grid, args.grid_size, args.seed, stage["max_goal_dist"],
                        stage["max_steps"], args.arm)
        policy_kwargs = {}
        if args.arm == "equivariant":
            from envs.equivariant_extractor import D4EquivariantExtractor
            policy_kwargs = {"features_extractor_class": D4EquivariantExtractor,
                              "features_extractor_kwargs": {"grid_feat_dim": 64}}

        if ckpt:
            model = Algo.load(ckpt, env=env)
            print(f"resumed from {ckpt}", flush=True)
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
                        tau=0.005, ent_coef=0.02, policy_kwargs=policy_kwargs)
        else:
            model = PPO("MultiInputPolicy", env, seed=args.seed, verbose=1,
                         n_steps=256, batch_size=64, ent_coef=0.01,
                         policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=stage["timesteps"], reset_num_timesteps=(ckpt is None),
                    callback=SuccessRateCallback(log_every=2000))
        ckpt = f"{args.out_prefix}_{stage['name']}.zip"
        model.save(ckpt)
        env.close()

        sr_det, sr_stoch = evaluate(model, grid, args.grid_size, args.seed,
                                     stage["max_goal_dist"], stage["max_steps"])
        print(f"=== STAGE {stage['name']} DONE: success_rate={sr_det:.2%} "
              f"stochastic={sr_stoch:.2%} (own difficulty) ===", flush=True)
        results[stage["name"]] = {"deterministic": sr_det, "stochastic": sr_stoch}

    sr_det, sr_stoch = evaluate(model, grid, args.grid_size, args.seed, None,
                                 STAGES[-1]["max_steps"], n_episodes=30)
    print(f"=== FINAL full-random task success_rate={sr_det:.2%} "
          f"stochastic={sr_stoch:.2%} ===", flush=True)
    results["final_full_random"] = {"deterministic": sr_det, "stochastic": sr_stoch}
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
