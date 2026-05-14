"""
Big Sur Marathon -- visualization helpers
Pairs with big_sur_simpy.py. Three plot types:
   1. Queue length over time per resource (from the QueueMonitor)
   2. Aid wait distribution per station (from metrics.aid_wait_min if you
      store station IDs; demonstrated as a simple histogram here)
   3. Sketch: animation of runner positions along the course over time

USAGE: put this file and big_sur_simpy.py in the same directory, then run
    python big_sur_visuals.py
Outputs are saved next to this script.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from big_sur_simpy import (
    run_marathon_full, AidStationConfig, AID_STATION_MILES, COURSE_MILES,
    elevation_at_mile,
)

# All plots saved next to this script
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# (1) Queue length over time
# ---------------------------------------------------------------------------
def plot_queue_lengths(monitor, names_to_plot=None, output_path="queues.png",
                       title="Queue lengths over simulation time"):
    """Plot selected queue length time series.
    `names_to_plot`: list of monitor.series keys; defaults to all."""
    fig, ax = plt.subplots(figsize=(13, 6))
    keys = names_to_plot or sorted(monitor.series.keys())
    for k in keys:
        ax.plot(monitor.times, monitor.series[k], lw=1.5, label=k, alpha=0.85)
    ax.axvline(0, ls="--", color="black", alpha=0.5, label="race start (t=0)")
    ax.set_xlabel("Sim time (min, race start = 0)")
    ax.set_ylabel("Queue length (entities waiting)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# (2) Aid wait distribution (one histogram across all stations)
# ---------------------------------------------------------------------------
def plot_aid_wait_distribution(metrics, output_path="aid_wait_dist.png",
                               target_sec=10, fail_sec=30):
    if not metrics.aid_wait_min:
        print("No aid waits recorded.")
        return
    aw_sec = np.array(metrics.aid_wait_min) * 60.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(aw_sec, bins=60, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(target_sec, ls="--", color="green",    lw=2, label=f"target ≤ {target_sec} sec")
    ax.axvline(fail_sec,   ls="--", color="firebrick", lw=2, label=f"failure > {fail_sec} sec")
    p_under_target = float((aw_sec <= target_sec).mean())
    p_over_fail    = float((aw_sec >  fail_sec).mean())
    ax.set_xlabel("Aid station wait time (sec)")
    ax.set_ylabel("# visits")
    ax.set_title(f"Aid station wait distribution -- "
                 f"{p_under_target:.1%} meet target, {p_over_fail:.1%} fail")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# (3) Server sweep with target lines drawn in
# ---------------------------------------------------------------------------
def plot_server_sweep_with_targets(server_counts, reps=3, n_runners=2000,
                                   target_sec=10, fail_sec=30,
                                   output_path="server_sweep_targeted.png"):
    """Run sweep and plot p95 and max with target/failure lines."""
    rows = []
    for ns in server_counts:
        for rep in range(reps):
            stations = [AidStationConfig(mile=m, servers=ns) for m in AID_STATION_MILES]
            metrics, _ = run_marathon_full(seed=42 + rep, n_runners=n_runners,
                                           station_cfgs=stations,
                                           monitor_interval_min=999)  # disable monitor
            aw_sec = np.array(metrics.aid_wait_min) * 60.0 if metrics.aid_wait_min else np.array([0.0])
            rows.append({
                "servers": ns,
                "p50": float(np.percentile(aw_sec, 50)),
                "p95": float(np.percentile(aw_sec, 95)),
                "max": float(aw_sec.max()),
            })
    import pandas as pd
    df = pd.DataFrame(rows).groupby("servers").mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["servers"], df["p50"], marker="o", lw=2, label="median wait")
    ax.plot(df["servers"], df["p95"], marker="s", lw=2, label="p95 wait")
    ax.plot(df["servers"], df["max"], marker="^", lw=2, label="max wait", alpha=0.7)
    ax.axhline(target_sec, ls="--", color="green",    lw=2)
    ax.axhline(fail_sec,   ls="--", color="firebrick", lw=2)
    ax.text(df["servers"].max(), target_sec, f"  target ({target_sec}s)", color="green", va="center")
    ax.text(df["servers"].max(), fail_sec,   f"  failure ({fail_sec}s)", color="firebrick", va="center")
    ax.set_xlabel("Servers per aid station")
    ax.set_ylabel("Aid wait (sec)")
    ax.set_yscale("log")
    ax.set_title(f"Aid wait vs server count ({n_runners} runners, {reps} reps)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")
    return df


# ---------------------------------------------------------------------------
# (4) Sketch: animation of runners along the course
# ---------------------------------------------------------------------------
# Plan if you want to build this:
#
# Step A. Add lightweight position logging to _race_segment:
#   record a (t, bib, mile) tuple every leg boundary plus every aid station.
#   In Python: append to a list passed into the function. 4500 runners * ~12 events
#   each = 54k tuples, ~ couple MB.
#
# Step B. Reconstruct position(t, bib) by linear interpolation between recorded
#   waypoints. Vectorize this with numpy: a (n_runners, n_waypoints) array of
#   times and miles per runner.
#
# Step C. Use matplotlib.animation.FuncAnimation:
#     fig, ax = plt.subplots()
#     ax.plot(course_x, course_y)       # course as a 1D line, x=mile, y=0
#     scatter = ax.scatter([], [])
#     def update(t):
#         miles_at_t = [position(t, bib) for bib in range(n_runners)]
#         scatter.set_offsets(np.c_[miles_at_t, np.zeros(n_runners)])
#         return scatter,
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)
#     anim.save("runners.mp4", fps=20)
#
# Step D. (Nicer) plot the elevation profile as the y-axis so runners visibly
#   climb Hurricane Point. Use elevation_at_mile() to compute y.
#
# Step E. (Even nicer) color runners by corral, or by injury status.
#
# Equivalent for the bus phase: stack the three bus stops vertically and animate
# the queue building up and emptying.


# ---------------------------------------------------------------------------
# (5) Event timeline (Gantt-style)
# ---------------------------------------------------------------------------
def plot_event_timeline(metrics, output_path="event_timeline.png"):
    """Gantt-style timeline of the entire race day from bus loading to cutoff."""
    from big_sur_simpy import (
        DEFAULT_BUS_STOPS, CORRAL_START_MIN, AID_STATION_MILES, CUTOFF_MIN,
    )
    fig, ax = plt.subplots(figsize=(14, 9))
    y = 0
    yticks, yticklabels = [], []

    # Bus stops: pickup window (light blue) + bus arrival range at start (dark blue)
    for stop in DEFAULT_BUS_STOPS:
        ax.barh(y, stop.arrival_end_min - stop.arrival_start_min,
                left=stop.arrival_start_min, height=0.45,
                color="lightblue", edgecolor="steelblue",
                label="bus pickup window" if y == 0 else None)
        if stop.name in metrics.first_bus_at_start:
            t0 = metrics.first_bus_at_start[stop.name]
            t1 = metrics.last_bus_at_start[stop.name]
            ax.barh(y, t1 - t0, left=t0, height=0.45,
                    color="navy", alpha=0.6,
                    label="bus arrivals at start" if y == 0 else None)
        yticks.append(y); yticklabels.append(f"Bus {stop.name}"); y += 1

    y += 0.5
    if metrics.start_porta_wait_min:
        ax.barh(y, 55, left=-60, height=0.5, color="orange", alpha=0.7,
                label="start porta-john usage")
        yticks.append(y); yticklabels.append("Start porta-johns"); y += 1

    y += 0.5
    corral_y = y
    for corral, t in CORRAL_START_MIN.items():
        ax.plot(t, corral_y, "v", markersize=14, color="darkgreen",
                label="corral start" if corral == "A" else None)
        ax.annotate(corral, (t, corral_y), xytext=(t, corral_y - 0.35),
                    ha="center", fontsize=9, fontweight="bold")
    yticks.append(corral_y); yticklabels.append("Corral starts"); y += 1

    y += 0.5
    for mile in AID_STATION_MILES:
        if mile in metrics.first_at_station:
            t0 = metrics.first_at_station[mile]
            t1 = metrics.last_at_station[mile]
            ax.barh(y, t1 - t0, left=t0, height=0.45,
                    color="forestgreen", alpha=0.55,
                    label="aid station active" if mile == AID_STATION_MILES[0] else None)
        yticks.append(y); yticklabels.append(f"Aid @{mile}"); y += 1

    y += 0.5
    sev_color = {"minor": "gold", "medium": "orange", "high": "firebrick"}
    for sev in ("minor", "medium", "high"):
        ts = [t for t, _, s in metrics.injury_events if s == sev]
        if ts:
            ax.scatter(ts, [y] * len(ts), c=sev_color[sev], s=12, alpha=0.6,
                       label=f"injury: {sev}")
    yticks.append(y); yticklabels.append("Injuries"); y += 1.5

    ax.axvline(0,          ls="--", color="black",     lw=2, label="race start")
    ax.axvline(CUTOFF_MIN, ls="--", color="firebrick", lw=2, label="6-hr cutoff")

    ax.set_xlabel("Minutes from race start (race start = 0; 6:45 AM)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_title("Event Timeline: bus loading -> race -> cutoff")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(-230, CUTOFF_MIN + 50)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# (6) Monte Carlo and sweep comparison plots
# ---------------------------------------------------------------------------
def plot_monte_carlo_distributions(df, metrics_to_plot=None,
                                   output_path="mc_distributions.png"):
    """Box+strip plot of MC results across reps for selected metrics."""
    from big_sur_simpy import DEFAULT_TARGETS
    if metrics_to_plot is None:
        metrics_to_plot = ["aid_wait_p95_sec", "porta_wait_p95_min",
                           "med_resp_p90_minor_min", "med_resp_p90_high_min",
                           "bus_wait_p95_min", "start_porta_p95_min"]
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(14, 8))
    axes = axes.flatten()
    for i, key in enumerate(metrics_to_plot):
        if key not in df.columns:
            continue
        vals = df[key].values
        ax = axes[i]
        ax.boxplot([vals], vert=True, widths=0.5)
        ax.scatter(np.random.normal(1, 0.04, len(vals)), vals,
                   alpha=0.5, s=20, color="steelblue")
        tgt = DEFAULT_TARGETS.get(key)
        if tgt:
            ax.axhline(tgt.target,  ls="--", color="green",     lw=1.5,
                       label=f"target {tgt.target}")
            ax.axhline(tgt.failure, ls="--", color="firebrick", lw=1.5,
                       label=f"fail {tgt.failure}")
            ax.set_title(f"{tgt.name}\nmean={vals.mean():.2f}, "
                         f"pass={float((vals<=tgt.target).mean()):.0%}",
                         fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.set_title(key)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Monte Carlo distributions across {len(df)} reps", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_bike_placement_comparison(df, output_path="bike_placement.png"):
    """Compare bike placements by p90 response time, grouped by severity."""
    placements = df["placement"].unique()
    sevs = ["minor", "medium", "high"]
    colors = {"minor": "gold", "medium": "orange", "high": "firebrick"}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(placements))
    w = 0.25
    for i, sev in enumerate(sevs):
        means = [df[df["placement"] == p][f"p90_{sev}"].mean() for p in placements]
        stds  = [df[df["placement"] == p][f"p90_{sev}"].std()  for p in placements]
        ax.bar(x + (i-1)*w, means, w, yerr=stds, capsize=3,
               color=colors[sev], alpha=0.85, label=f"{sev} p90")
    ax.axhline(5,  ls="--", color="green",     alpha=0.7, label="target 5 min")
    ax.axhline(10, ls="--", color="firebrick", alpha=0.7, label="fail 10 min")
    ax.set_xticks(x)
    n_bikes = [df[df["placement"]==p]["n_bikes"].iloc[0] for p in placements]
    ax.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(placements, n_bikes)],
                       fontsize=9)
    ax.set_ylabel("p90 medical response time (min)")
    ax.set_title("Bike placement comparison (mean of MC reps; error bars = 1 SD)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_corral_comparison(df, output_path="corral_comparison.png"):
    """Compare corral plans on aid wait, finish time spread, late starts."""
    plans = df["plan"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Aid wait p95
    data = [df[df["plan"] == p]["aid_wait_p95_sec"].values for p in plans]
    axes[0].boxplot(data, labels=plans, vert=True)
    axes[0].axhline(10, ls="--", color="green", lw=1.5, label="target 10s")
    axes[0].axhline(30, ls="--", color="firebrick", lw=1.5, label="fail 30s")
    axes[0].set_ylabel("Aid wait p95 (sec)")
    axes[0].set_title("Aid station wait")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Late to corral
    data = [df[df["plan"] == p]["late_to_corral_count"].values for p in plans]
    axes[1].boxplot(data, labels=plans, vert=True)
    axes[1].set_ylabel("# runners late to corral")
    axes[1].set_title("Late-to-corral count")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(True, alpha=0.3)

    # Trash overflow count
    data = [df[df["plan"] == p]["trash_overflow_count"].values for p in plans]
    axes[2].boxplot(data, labels=plans, vert=True)
    axes[2].set_ylabel("# trash overflow events")
    axes[2].set_title("Trash overflow")
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Corral plan sensitivity (boxplot across MC reps)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    # Run one replication, then make all three plots
    print("Running simulation (this takes ~10 seconds for 2500 runners)...")
    metrics, monitor = run_marathon_full(seed=42, n_runners=2500,
                                         monitor_interval_min=1.0)

    # Plot 1: queue lengths at a few interesting resources
    plot_queue_lengths(
        monitor,
        names_to_plot=[
            "bus:Monterey", "bus:Carmel", "bus:ParkRideA",
            "start_porta",
            "aid_serv@2.5", "aid_serv@10.4", "aid_serv@19.0",
            "bike_medics", "ambulances",
        ],
        output_path=os.path.join(HERE, "queue_lengths.png"),
    )

    # Plot 2: aid wait histogram
    plot_aid_wait_distribution(
        metrics, output_path=os.path.join(HERE, "aid_wait_dist.png"),
        target_sec=10, fail_sec=30,
    )

    # Plot 3: server sweep with target lines
    print("Running server sweep (this takes ~30 seconds)...")
    df = plot_server_sweep_with_targets(
        server_counts=[4, 6, 8, 10, 12, 16, 20],
        reps=2, n_runners=2000,
        target_sec=10, fail_sec=30,
        output_path=os.path.join(HERE, "server_sweep_targeted.png"),
    )
    print(df.to_string(index=False))
