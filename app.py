"""
Big Sur Marathon Simulator -- Streamlit UI
Run with:  streamlit run app.py

Tabs:
  1. Single Scenario   -- set parameters, run, see metrics + plots
  2. Compare Scenarios -- paired-seed comparison of two configs
  3. Sensitivity Sweep -- vary one parameter across a range
"""
from __future__ import annotations
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from big_sur_simpy import (
    run_marathon_full, run_monte_carlo, monte_carlo_summary,
    paired_seed_compare, evaluate_targets, _extract_metrics,
    AidStationConfig, MedicalConfig, StartAreaConfig, CorralPlan,
    AID_STATION_MILES, DEFAULT_TARGETS, DEFAULT_BUS_STOPS, BusStopConfig,
)

st.set_page_config(page_title="Big Sur Marathon Simulator", layout="wide")
st.title("🏃 Big Sur Marathon Simulator")
st.caption("SimPy discrete-event model. Adjust parameters in the sidebar; "
           "run scenarios to see how logistics decisions affect runner experience.")


# =============================================================================
# Sidebar: parameters used by all tabs
# =============================================================================
def parameter_sidebar(key_prefix: str = ""):
    """Returns a dict of run_marathon_full kwargs from sidebar inputs."""
    p = lambda k: f"{key_prefix}_{k}"

    st.sidebar.header("Field & seed")
    n_runners = st.sidebar.slider("Number of runners",  500, 4500, 1500, 250,
                                   key=p("n_runners"))
    base_seed = st.sidebar.number_input("Base seed",     value=42, key=p("seed"))

    st.sidebar.header("Corrals")
    n_corrals = st.sidebar.slider("Number of corrals",   1, 10, 3, key=p("nc"))
    interval  = st.sidebar.slider("Interval between corrals (min)",
                                   0.0, 15.0, 5.0, 0.5, key=p("ci"))
    corral_plan = CorralPlan.custom(n_corrals, interval)

    st.sidebar.header("Aid stations")
    serv_early = st.sidebar.slider("Servers (mile < 8)",   4, 25, 16, key=p("se"))
    serv_mid   = st.sidebar.slider("Servers (mile 8-18)",  4, 20, 10, key=p("sm"))
    serv_late  = st.sidebar.slider("Servers (mile > 18)",  4, 15,  6, key=p("sl"))
    aid_pj     = st.sidebar.slider("Porta-johns per station", 2, 16, 8, key=p("apj"))
    prep_w     = st.sidebar.slider("Water prep rate (cups/min, early)",
                                    50, 500, 300, key=p("pw"))
    init_buf   = st.sidebar.slider("Initial water buffer (cups)",
                                    20, 200, 80, key=p("ib"))

    st.sidebar.header("Medical")
    n_bikes      = st.sidebar.slider("Bike medics",       4, 24, 12, key=p("nb"))
    n_ambulances = st.sidebar.slider("Ambulances",        2, 10,  4, key=p("na"))
    bike_strategy = st.sidebar.selectbox(
        "Bike placement strategy",
        ["even", "at_aid", "hurricane_heavy"], key=p("bs"))

    st.sidebar.header("Pre-race")
    start_pj   = st.sidebar.slider("Porta-johns at start", 30, 200, 100, key=p("spj"))
    bus_freq   = st.sidebar.slider("Bus dispatch every N min (15-min stops)",
                                    1, 4, 2, key=p("bf"))

    # ------- Build config -------
    stations = []
    for m in AID_STATION_MILES:
        if m < 8:    sv = serv_early
        elif m < 18: sv = serv_mid
        else:        sv = serv_late
        stations.append(AidStationConfig(
            mile=m, servers=sv, porta_johns=aid_pj,
            water_ready_init=init_buf,
            water_buffer_capacity=max(init_buf*2, 150),
            water_prep_per_min=prep_w,
            trash_capacity=int(1.6 * n_runners),
        ))

    if bike_strategy == "even":
        bike_locs = tuple(np.round(np.linspace(2, 25, n_bikes), 1))
    elif bike_strategy == "at_aid":
        bike_locs = tuple(AID_STATION_MILES[:n_bikes]) if n_bikes <= 11 \
                    else tuple(AID_STATION_MILES) + tuple(np.linspace(3, 24, n_bikes-11))
    else:  # hurricane_heavy
        center_count = max(4, n_bikes // 2)
        spread_count = n_bikes - center_count
        bike_locs = (tuple(np.linspace(9, 13, center_count)) +
                     tuple(np.linspace(2, 25, spread_count)))
    med = MedicalConfig(num_bike_medics=n_bikes, num_ambulances=n_ambulances,
                        bike_locations=bike_locs)

    start = StartAreaConfig(porta_johns=start_pj)

    bus_stops = []
    for s in DEFAULT_BUS_STOPS:
        win = s.arrival_end_min - s.arrival_start_min
        new_dispatches = tuple(range(int(s.bus_dispatch_times[0]),
                                      int(s.bus_dispatch_times[-1]) + 1,
                                      bus_freq if win <= 16 else 2))
        bus_stops.append(BusStopConfig(
            name=s.name, n_runners=s.n_runners,
            arrival_start_min=s.arrival_start_min,
            arrival_end_min=s.arrival_end_min,
            travel_time_mean_min=s.travel_time_mean_min,
            travel_time_std_min=s.travel_time_std_min,
            bus_dispatch_times=new_dispatches,
        ))

    return dict(
        n_runners=n_runners, base_seed=int(base_seed),
        station_cfgs=stations, med_cfg=med, start_area_cfg=start,
        bus_stop_cfgs=tuple(bus_stops), corral_plan=corral_plan,
    ), corral_plan


# =============================================================================
# Result rendering helpers
# =============================================================================
def render_status_table(rows):
    """Color-code PASS/WARN/FAIL."""
    df = pd.DataFrame(rows)
    def color_status(val):
        if val == "PASS": return "background-color: #d4edda"
        if val == "WARN": return "background-color: #fff3cd"
        if val == "FAIL": return "background-color: #f8d7da"
        return ""
    return df.style.map(color_status, subset=["status"]).format({
        "value": "{:.2f}", "target": "{:.1f}", "failure": "{:.1f}",
    })


def plot_metric_distribution(df, metric_key, target=None, failure=None,
                             title=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df[metric_key].values
    ax.boxplot([vals], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.scatter(np.random.normal(1, 0.04, len(vals)), vals, alpha=0.6, s=30)
    if target is not None:
        ax.axhline(target, ls="--", color="green", label=f"target {target}")
    if failure is not None:
        ax.axhline(failure, ls="--", color="firebrick", label=f"fail {failure}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# Tab 1: Single Scenario
# =============================================================================
def tab_single_scenario():
    st.header("Single Scenario")
    st.write("Choose parameters in the sidebar, then run a Monte Carlo of N reps "
             "with that configuration.")
    cfg, plan = parameter_sidebar(key_prefix="single")
    n_reps = st.slider("Monte Carlo reps", 1, 25, 5, key="mc_reps")

    if st.button("🏃 Run Monte Carlo", type="primary"):
        with st.spinner(f"Running {n_reps} replications..."):
            t0 = time.time()
            prog = st.progress(0)
            rows = []
            for i in range(n_reps):
                m, _ = run_marathon_full(seed=cfg["base_seed"] + i,
                                         n_runners=cfg["n_runners"],
                                         station_cfgs=cfg["station_cfgs"],
                                         med_cfg=cfg["med_cfg"],
                                         start_area_cfg=cfg["start_area_cfg"],
                                         bus_stop_cfgs=cfg["bus_stop_cfgs"],
                                         corral_plan=cfg["corral_plan"],
                                         monitor_interval_min=9999.0)
                d = _extract_metrics(m)
                d["rep"] = i
                d["n_finishers"] = len(m.finishes)
                rows.append(d)
                prog.progress((i + 1) / n_reps)
            mc_df = pd.DataFrame(rows)
            elapsed = time.time() - t0
        st.success(f"Ran {n_reps} reps in {elapsed:.1f}s")

        # Use the last replication's metrics for the status check display
        last_metrics = m
        status_rows = evaluate_targets(last_metrics)
        st.subheader("Targets (last replication)")
        st.dataframe(render_status_table(status_rows), use_container_width=True)

        st.subheader(f"Monte Carlo summary ({n_reps} reps)")
        summary = monte_carlo_summary(mc_df)
        cols = [c for c in ["metric", "mean", "std", "p5", "p50", "p95", "pass_rate"]
                if c in summary.columns]
        st.dataframe(summary[cols], use_container_width=True)

        # Plot key metrics
        st.subheader("Distributions")
        key_metrics = [
            ("aid_wait_p95_sec",       "Aid wait p95 (sec)",          10,   30),
            ("porta_wait_p95_min",     "Aid porta-john p95 (min)",    5,    15),
            ("med_resp_p90_high_min",  "Med response high p90 (min)", 5,    10),
            ("aid_balk_count",         "Aid balks",                   0,    200),
            ("porta_balk_count",       "Porta-john balks",            0,    100),
            ("stockout_count",         "Supply stockouts",            0,    100),
        ]
        cols_per_row = 3
        for i in range(0, len(key_metrics), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, (key, title, tgt, fail) in enumerate(key_metrics[i:i+cols_per_row]):
                with row_cols[j]:
                    if key in mc_df.columns:
                        st.pyplot(plot_metric_distribution(
                            mc_df, key, target=tgt, failure=fail, title=title))

        # Download raw data
        csv = mc_df.to_csv(index=False)
        st.download_button("⬇ Download raw CSV", csv,
                           file_name="single_scenario.csv", mime="text/csv")


# =============================================================================
# Tab 2: Compare Scenarios
# =============================================================================
def tab_compare():
    st.header("Compare Two Scenarios (paired-seed)")
    st.write("Two parameter columns; paired-seed comparison uses common random "
             "numbers, so the p-values are statistically defensible with few reps.")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Scenario A")
        cfg_a, _ = parameter_sidebar(key_prefix="compA")
    with col_b:
        st.subheader("Scenario B")
        # Slight different defaults for B
        cfg_b, _ = parameter_sidebar(key_prefix="compB")

    n_pairs = st.slider("Number of paired reps", 5, 30, 10, key="pair_reps")

    if st.button("⚖ Run paired comparison", type="primary"):
        cfg_a_run = {k: v for k, v in cfg_a.items()
                     if k not in {"n_runners", "base_seed"}}
        cfg_b_run = {k: v for k, v in cfg_b.items()
                     if k not in {"n_runners", "base_seed"}}
        with st.spinner(f"Running {n_pairs} paired reps (each = 2 runs)..."):
            df = paired_seed_compare(cfg_a_run, cfg_b_run, n_pairs=n_pairs,
                                     n_runners=cfg_a["n_runners"],
                                     base_seed=cfg_a["base_seed"],
                                     label_A="A", label_B="B")
        st.success("Done.")
        st.dataframe(df.style.format({
            "A_mean": "{:.2f}", "B_mean": "{:.2f}", "B_minus_A": "{:+.2f}",
            "diff_ci_lo": "{:+.2f}", "diff_ci_hi": "{:+.2f}",
            "t_stat": "{:.2f}", "p_value": "{:.4f}",
        }), use_container_width=True)


# =============================================================================
# Tab 3: Sensitivity sweep
# =============================================================================
def tab_sensitivity():
    st.header("Sensitivity Sweep")
    st.write("Vary one parameter across a range and watch outputs change.")

    parameter = st.selectbox("Parameter to sweep", [
        "Servers per station (uniform)",
        "Porta-johns per aid station",
        "Number of bike medics",
        "Number of ambulances",
        "Water prep rate (cups/min)",
        "Initial water buffer (cups)",
    ])
    n_runners = st.slider("Number of runners",  500, 4500, 1500, 250, key="sw_n")
    n_reps    = st.slider("Reps per value",      1,   10,    3,    key="sw_r")

    if parameter == "Servers per station (uniform)":
        values = st.multiselect("Values", [4,6,8,10,12,14,16,20], default=[6,8,10,12,16])
    elif parameter == "Porta-johns per aid station":
        values = st.multiselect("Values", [2,4,6,8,10,12,14,16], default=[4,6,8,10,12])
    elif parameter == "Number of bike medics":
        values = st.multiselect("Values", [4,6,8,10,12,14,16,20], default=[6,8,12,16])
    elif parameter == "Number of ambulances":
        values = st.multiselect("Values", [2,3,4,5,6,8,10], default=[3,4,5,6])
    elif parameter == "Water prep rate (cups/min)":
        values = st.multiselect("Values", [50,100,150,200,250,300,400], default=[100,200,300])
    else:
        values = st.multiselect("Values", [20,40,60,80,100,150,200], default=[40,80,120])

    if st.button("🔬 Run sweep", type="primary"):
        progress = st.progress(0)
        rows = []
        total = len(values) * n_reps
        i = 0
        for v in values:
            # Build the modified config
            stations = []
            for m in AID_STATION_MILES:
                if m < 8:    sv = 16
                elif m < 18: sv = 10
                else:        sv = 6
                s = AidStationConfig(mile=m, servers=sv, porta_johns=8,
                                     trash_capacity=int(1.6 * n_runners))
                stations.append(s)
            med = MedicalConfig()
            start = StartAreaConfig()
            if parameter == "Servers per station (uniform)":
                for s in stations: s.servers = v
            elif parameter == "Porta-johns per aid station":
                for s in stations: s.porta_johns = v
            elif parameter == "Number of bike medics":
                med = MedicalConfig(num_bike_medics=v,
                    bike_locations=tuple(np.round(np.linspace(2, 25, v), 1)))
            elif parameter == "Number of ambulances":
                med = MedicalConfig(num_ambulances=v)
            elif parameter == "Water prep rate (cups/min)":
                for s in stations: s.water_prep_per_min = v
            else:  # Initial water buffer
                for s in stations:
                    s.water_ready_init = v
                    s.water_buffer_capacity = max(v*2, 150)

            for rep in range(n_reps):
                m_obj, _ = run_marathon_full(seed=42 + rep, n_runners=n_runners,
                                              station_cfgs=stations, med_cfg=med,
                                              start_area_cfg=start,
                                              monitor_interval_min=9999.0)
                d = _extract_metrics(m_obj)
                d["param_value"] = v
                d["rep"] = rep
                rows.append(d)
                i += 1
                progress.progress(i / total)
        sweep_df = pd.DataFrame(rows)
        st.success(f"Ran {total} sims.")

        agg = sweep_df.groupby("param_value").agg(
            aid_p95=("aid_wait_p95_sec", "mean"),
            porta_p95=("porta_wait_p95_min", "mean"),
            med_high=("med_resp_p90_high_min", "mean"),
            balks_aid=("aid_balk_count", "mean"),
            balks_porta=("porta_balk_count", "mean"),
            stockouts=("stockout_count", "mean"),
        ).round(2)
        st.subheader(f"Aggregated by {parameter}")
        st.dataframe(agg, use_container_width=True)

        # Quick line plot of key metrics
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in ["aid_p95", "balks_porta", "balks_aid"]:
            ax.plot(agg.index, agg[col], marker="o", label=col)
        ax.set_xlabel(parameter)
        ax.set_ylabel("metric value")
        ax.set_title(f"Sensitivity: {parameter}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)


# =============================================================================
# Main: tabs
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Single Scenario", "⚖ Compare Scenarios",
                             "🔬 Sensitivity Sweep"])
with tab1:
    tab_single_scenario()
with tab2:
    tab_compare()
with tab3:
    tab_sensitivity()

st.sidebar.markdown("---")
st.sidebar.caption("Bixby Bridge Beer Bottleneck — Big Sur Marathon Simulator")
