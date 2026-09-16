"""Turn a training log into tidy learning-curve data (paper Figure 2).

Reads the rolling success rate, episode reward, and the per-stage evaluation
results out of a `curriculum_train.py` log and writes them as CSV + JSON, so
the figures are generated from the run's own record rather than from numbers
copied by hand.

Handles both log formats: newer runs print the timestep on the success-rate
line, older ones do not, and for those the timestep is recovered by bracketing
each line between the surrounding rollout blocks (SB3 prints
`total_timesteps` in every block, so a success-rate line sits between two
known values and is attributed to the preceding one).

Both a cumulative and a per-stage timestep column are emitted. SB3's
`total_timesteps` is already cumulative across stages here -- only the first
stage passes `reset_num_timesteps=True`, so the counter runs straight through
(verified in the logs: a stage boundary reads 79982 -> 80284, not 79982 -> 0)
-- so the per-stage figure is what has to be derived, by subtracting the
counter's value when the stage began.

    python3 scripts/extract_results.py sac_baseline_v5.log --out results_v5
"""
import argparse, csv, json, os, re, sys

RE_STAGE = re.compile(r"^=== STAGE (\w+): dist ([\d.]+)-(\S+) steps=(\d+) timesteps=(\d+) ===")
RE_STAGE_OLD = re.compile(r"^=== STAGE (\w+): dist<=(\S+) steps=(\d+) timesteps=(\d+) ===")
RE_DONE = re.compile(r"^=== STAGE (\w+) DONE: success_rate=([\d.]+)%(?: stochastic=([\d.]+)%)?")
RE_FINAL = re.compile(r"^=== FINAL.*?success_rate=([\d.]+)%(?: stochastic=([\d.]+)%)?")
RE_SUCCESS = re.compile(r"^\[success-rate\](?: t=(\d+))? last (\d+) episodes: ([\d.]+)% reached")
RE_TIMESTEPS = re.compile(r"^\|\s+total_timesteps\s+\|\s+(\d+)")
RE_EPREW = re.compile(r"^\|\s+ep_rew_mean\s+\|\s+(\S+)")
RE_EPLEN = re.compile(r"^\|\s+ep_len_mean\s+\|\s+(\S+)")


def parse(path):
    curve, stage_evals = [], []
    stage = None
    stage_start_t = None      # counter value when the current stage began
    last_t = 0
    ep_rew = ep_len = None
    final = None

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")

            m = RE_STAGE.match(line) or RE_STAGE_OLD.match(line)
            if m:
                stage = m.group(1)
                # the counter keeps running, so the stage's origin is wherever
                # it stood at the boundary
                stage_start_t = last_t
                continue

            m = RE_TIMESTEPS.match(line)
            if m:
                last_t = int(m.group(1))
                continue
            m = RE_EPREW.match(line)
            if m:
                ep_rew = float(m.group(1))
                continue
            m = RE_EPLEN.match(line)
            if m:
                ep_len = float(m.group(1))
                continue

            m = RE_SUCCESS.match(line)
            if m:
                t = int(m.group(1)) if m.group(1) else last_t
                curve.append({
                    "stage": stage,
                    "timesteps_in_stage": t - (stage_start_t or 0),
                    "timesteps_cumulative": t,
                    "window_episodes": int(m.group(2)),
                    "success_rate": float(m.group(3)) / 100.0,
                    "ep_rew_mean": ep_rew,
                    "ep_len_mean": ep_len,
                })
                continue

            m = RE_DONE.match(line)
            if m:
                stage_evals.append({
                    "stage": m.group(1),
                    "eval_deterministic": float(m.group(2)) / 100.0,
                    "eval_stochastic": (float(m.group(3)) / 100.0
                                        if m.group(3) else None),
                })
                continue

            m = RE_FINAL.match(line)
            if m:
                final = {
                    "eval_deterministic": float(m.group(1)) / 100.0,
                    "eval_stochastic": (float(m.group(2)) / 100.0
                                        if m.group(2) else None),
                }

    return curve, stage_evals, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None,
                    help="output prefix (default: the log path without .log)")
    args = ap.parse_args()
    if not os.path.exists(args.log):
        sys.exit(f"no such log: {args.log}")
    prefix = args.out or os.path.splitext(args.log)[0]

    curve, stage_evals, final = parse(args.log)
    if not curve:
        sys.exit(f"no success-rate lines found in {args.log} -- wrong file?")

    with open(f"{prefix}_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    summary = {"log": args.log, "stage_evals": stage_evals, "final": final,
               "curve_points": len(curve)}
    with open(f"{prefix}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{len(curve)} curve points -> {prefix}_curve.csv")
    print(f"stage evals + final -> {prefix}_summary.json\n")
    print(f"{'stage':6s} {'points':>6s} {'peak train':>11s} "
          f"{'eval det':>9s} {'eval stoch':>11s}")
    for ev in stage_evals:
        pts = [c for c in curve if c["stage"] == ev["stage"]]
        peak = max((c["success_rate"] for c in pts), default=float("nan"))
        stoch = ev["eval_stochastic"]
        print(f"{ev['stage']:6s} {len(pts):6d} {peak:10.0%} "
              f"{ev['eval_deterministic']:9.0%} "
              f"{stoch if stoch is None else format(stoch, '.0%'):>11}")
    if final:
        stoch = final["eval_stochastic"]
        print(f"{'FINAL':6s} {'':6s} {'':>10} {final['eval_deterministic']:9.0%} "
              f"{stoch if stoch is None else format(stoch, '.0%'):>11}")


if __name__ == "__main__":
    main()
