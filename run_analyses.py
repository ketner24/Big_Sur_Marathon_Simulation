"""
Run Monte Carlo + bike placement + corral sensitivity analyses.
Saves three figures and prints summary tables.
"""
import os, sys, time
sys.path.insert(0, "/home/claude")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from big_sur_simpy import (
    CorralPlan, MedicalConfig,
    run_monte_carlo, monte_carlo_summary,
    bike_placement_sweep, corral_sensitivity_sweep,
    make_recommended_aid_configs, StartAreaConfig,
)

HERE = "/home/claude"

# ============================================================
# (1) Monte Carlo baseline run -- 20 reps with the recommended config
# ============================================================
print("=" * 70)
print("(1) Monte Carlo on recommended config (10 reps x 1500 runners)")
print("=" * 70)
t0 = time.time()
stations  = make_recommended_aid_configs(n_runners=1500)
for s in stations:
    s.porta_johns = 8
start_cfg = StartAreaConfig(porta_johns=100)
med_cfg   = MedicalConfig(num_bike_medics=20, num_ambulances=5)

mc_df = run_monte_carlo(n_reps=10, n_runners=1500,
                        station_cfgs=stations,
                        start_area_cfg=start_cfg,
                        med_cfg=med_cfg)
print(f"  ... took {time.time()-t0:.1f} sec\n")
summary = monte_carlo_summary(mc_df)
print(summary[["metric", "mean", "std", "p5", "p95",
               "pass_rate", "fail_rate"]].to_string(index=False))

# Plot: box plot of key metrics across reps
key_cols = ["aid_wait_p95_sec", "aid_wait_max_sec",
            "porta_wait_p95_min", "med_resp_p90_minor_min",
            "med_resp_p90_high_min", "amb_resp_p90_min",
            "bus_wait_p95_min", "start_porta_p95_min"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flat, key_cols):
    ax.boxplot([mc_df[col].values], showmeans=True)
    ax.set_title(col, fontsize=9)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3)
fig.suptitle("Monte Carlo distributions across 20 replications "
             "(recommended config, 2000 runners)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mc_distributions.png"), dpi=110, bbox_inches="tight")
plt.close()
print(f"\nSaved mc_distributions.png")


# ============================================================
# (2) Bike placement sweep
# ============================================================
print("\n" + "=" * 70)
print("(2) Bike placement sweep (8 reps x 1200 runners)")
print("=" * 70)
t0 = time.time()
placements = {
    "4_quartiles":            (3.0, 10.0, 17.0, 24.0),
    "8_evenly_spaced":        (1.5, 4.6, 7.7, 10.8, 13.9, 17.0, 20.1, 23.2),
    "12_co-located_with_aid": (2.5, 4.8, 7.8, 10.4, 12.2, 14.7, 16.9, 19.0, 21.2, 23.0, 24.5, 25.5),
    "16_dense_early":         (1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 24.5, 25.5),
}
bs_df = bike_placement_sweep(placements, n_reps=8, n_runners=1200)
print(f"  ... took {time.time()-t0:.1f} sec\n")
bs_agg = bs_df.groupby("placement").agg(
    n_bikes=("n_bikes", "first"),
    p90_minor_mean=("p90_minor", "mean"),
    p90_medium_mean=("p90_medium", "mean"),
    p90_high_mean=("p90_high", "mean"),
    p90_high_std=("p90_high", "std"),
    max_response_mean=("max_response", "mean"),
).reset_index()
print(bs_agg.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(bs_agg))
w = 0.25
ax.bar(x - w, bs_agg["p90_minor_mean"],  w, label="p90 minor",   color="seagreen", alpha=0.85)
ax.bar(x,     bs_agg["p90_medium_mean"], w, label="p90 medium",  color="orange",  alpha=0.85)
ax.bar(x + w, bs_agg["p90_high_mean"],   w, label="p90 high",    color="firebrick", alpha=0.85, yerr=bs_agg["p90_high_std"], capsize=4)
for i, (xi, n) in enumerate(zip(x, bs_agg["n_bikes"])):
    ax.text(xi, 0.5, f"n={n}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.axhline(5, ls="--", color="green",   lw=1.5, label="5-min target")
ax.axhline(10, ls="--", color="firebrick", lw=1.5, label="10-min failure")
ax.set_xticks(x)
ax.set_xticklabels(bs_agg["placement"], rotation=10, fontsize=9)
ax.set_ylabel("First-responder p90 response time (min)")
ax.set_title("Bike medic placement comparison (mean of 12 reps)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "bike_placement.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\nSaved bike_placement.png")


# ============================================================
# (3) Corral sensitivity sweep
# ============================================================
print("\n" + "=" * 70)
print("(3) Corral sensitivity sweep (6 reps x 1200 runners)")
print("=" * 70)
t0 = time.time()
plans = {
    "3 corrals @ 0/5/10 (baseline)":  CorralPlan.baseline_3x5(),
    "3 corrals @ 0/3/6":              CorralPlan.custom(3, 3.0),
    "5 corrals @ 0/2/4/6/8":          CorralPlan.five_corrals(interval_min=2.0),
    "5 corrals @ 0/3/6/9/12":         CorralPlan.five_corrals(interval_min=3.0),
    "8 corrals @ 0..14 step 2":       CorralPlan.custom(8, 2.0),
}
cs_df = corral_sensitivity_sweep(plans, n_reps=6, n_runners=1200)
print(f"  ... took {time.time()-t0:.1f} sec\n")

# Compute finish times of the *last* runner in each rep (using last_at_station as proxy)
# Actually we don't have last_finish directly; reuse n_finishers and use late_to_corral
cs_agg = cs_df.groupby("plan").agg(
    n_corrals=("n_corrals", "first"),
    max_offset=("max_offset", "first"),
    aid_wait_p95_mean=("aid_wait_p95_sec", "mean"),
    aid_wait_p95_std =("aid_wait_p95_sec", "std"),
    aid_wait_max_mean=("aid_wait_max_sec", "mean"),
    late_mean=("late_to_corral_count", "mean"),
    bus_wait_p95_mean=("bus_wait_p95_min", "mean"),
    stockouts_mean=("stockout_count", "mean"),
    n_finishers_mean=("n_finishers", "mean"),
).reset_index()
# Order by max_offset for plotting
cs_agg = cs_agg.sort_values("max_offset").reset_index(drop=True)
print(cs_agg.to_string(index=False))

# Plot: aid wait p95 by corral plan
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(cs_agg))

axes[0].bar(x, cs_agg["aid_wait_p95_mean"], yerr=cs_agg["aid_wait_p95_std"],
            color="steelblue", alpha=0.85, capsize=4)
axes[0].axhline(10, ls="--", color="green",   lw=1.5, label="10-sec target")
axes[0].axhline(30, ls="--", color="firebrick", lw=1.5, label="30-sec failure")
axes[0].set_xticks(x)
axes[0].set_xticklabels(cs_agg["plan"], rotation=20, fontsize=8, ha="right")
axes[0].set_ylabel("Aid wait p95 (sec)")
axes[0].set_title("Aid station congestion by corral plan")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# Second plot: trade-off scatter
axes[1].scatter(cs_agg["max_offset"], cs_agg["aid_wait_p95_mean"],
                s=100 + 30 * cs_agg["n_corrals"],
                c=cs_agg["n_corrals"], cmap="viridis", alpha=0.85, edgecolors="black")
for _, row in cs_agg.iterrows():
    axes[1].annotate(f"{row['n_corrals']}c", (row["max_offset"], row["aid_wait_p95_mean"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
axes[1].axhline(10, ls="--", color="green",  lw=1.5)
axes[1].axhline(30, ls="--", color="firebrick", lw=1.5)
axes[1].set_xlabel("Last corral starts (min after first)")
axes[1].set_ylabel("Aid wait p95 (sec)")
axes[1].set_title("Trade-off: total start window vs. congestion")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Corral structure sensitivity (12 reps, 1500 runners)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "corral_sensitivity.png"), dpi=110, bbox_inches="tight")
plt.close()
print("\nSaved corral_sensitivity.png")

print("\nAll analyses complete.")
