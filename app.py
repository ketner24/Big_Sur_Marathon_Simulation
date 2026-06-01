"""Big Sur Marathon Simulator — Streamlit UI

Three tabs:
  1. Single Scenario — Monte Carlo run with full parameter control
  2. Compare Scenarios — paired-seed (CRN) comparison of two presets
  3. Sensitivity Sweep — one-factor sweep to find elbow points

Reflects the final model:
  • Bus check-in is a mandatory queue (no balking)
  • Bike teams patrol BSIM zones with time windows, or evenly-spaced full coverage
  • Bike patrol speed (12 mph) is separate from response speed (16 mph)
  • Minor/medium injuries: runner continues. Severe (HIGH): DNF.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from big_sur_simpy import (
    AidStationConfig, MedicalConfig, StartAreaConfig, CorralPlan,
    BusStopConfig, BikeTeamConfig, DEFAULT_BUS_STOPS, DEFAULT_BIKE_TEAMS,
    make_even_bike_teams,
    run_marathon_full, run_monte_carlo, paired_seed_compare, _extract_metrics,
    AID_STATION_MILES, DEFAULT_TARGETS,
)

st.set_page_config(page_title="Big Sur Marathon Simulator", layout="wide")
st.title("🏃 Big Sur Marathon Simulator")
st.caption("SimPy discrete-event model • GMM-3 finish times (23,631 BSIM records) • "
           "Mandatory bus queue • BSIM zone-based bike patrol • "
           "**DNF rate calibrated to BSIM 0-4/yr historical baseline**")

# =============================================================================
# Presets — match the values used in the final analysis
# =============================================================================
PRESETS = {
    "Real BSIM 2026 (as provisioned)": dict(
        # Aid: 15-30+ volunteers documented; 6 porta-johns per station
        servers_early=26, servers_mid=22, servers_late=18,
        aid_pj=6, water_prep=250, electrolyte_prep=90,
        # Medical: BSIM zone-based teams (8 teams across 7 zones, time-windowed)
        bike_mode="BSIM zones (time-windowed, 8 teams)",
        n_even_teams=8,
        bike_patrol_speed=12, bike_response_speed=16,
        n_ambulances=4,
        # Start, corrals
        start_pj=146, n_corrals=3, interval=5.0,
        # Bus: 3 check-in volunteers (calibrated to user's 10-15 min wait observation)
        bus_volunteers=3, bus_checkin_sec=18,
    ),
    "Optimal (recommended)": dict(
        servers_early=26, servers_mid=22, servers_late=18,
        aid_pj=12, water_prep=450, electrolyte_prep=200,
        # 32 evenly-spaced teams on full race coverage
        bike_mode="Evenly-spaced (full race coverage)",
        n_even_teams=32,
        bike_patrol_speed=12, bike_response_speed=16,
        n_ambulances=4,
        start_pj=146, n_corrals=3, interval=5.0,
        # 5 bus volunteers meets 15-min p95 target
        bus_volunteers=5, bus_checkin_sec=18,
    ),
    "Custom (set sliders manually)": None,
}

# =============================================================================
# Sidebar parameter builder
# =============================================================================
def parameter_sidebar(key_prefix: str = "single"):
    p = lambda k: f"{key_prefix}_{k}"

    st.sidebar.header("⚡ Preset configuration")
    preset_name = st.sidebar.selectbox(
        "Start from:", list(PRESETS.keys()), index=1, key=p("preset"),
        help="Real = documented BSIM 2026 with zone-based bikes and 3 bus volunteers. "
             "Optimal = simulation recommendation (32 even-spaced bikes, 5 bus vol). "
             "Custom = use sliders as-is."
    )
    preset = PRESETS[preset_name]

    def slv(label, lo, hi, default_key, step=1, help_text=None):
        """Slider that pulls default from preset if available."""
        default = preset[default_key] if preset is not None else None
        if default is None:
            return st.sidebar.slider(label, lo, hi, key=p(default_key), step=step, help=help_text)
        slider_key = f"{p(default_key)}_{preset_name}"
        return st.sidebar.slider(label, lo, hi, value=default, key=slider_key, step=step, help=help_text)

    st.sidebar.header("Field & seed")
    n_runners = st.sidebar.slider("Marathon starters", 1500, 4500, 3500, 250, key=p("n"),
                                   help="BSIM documents 3,000-3,500. We model 3,500.")
    base_seed = st.sidebar.number_input("Base seed", value=42, key=p("seed"))

    st.sidebar.header("Corrals")
    n_corrals = slv("Number of corrals", 1, 10, "n_corrals")
    interval  = slv("Interval between corrals (min)", 0.0, 15.0, "interval", step=0.5)

    st.sidebar.header("Aid stations")
    serv_early = slv("Servers, mile < 8 (bunched pack)", 4, 30, "servers_early")
    serv_mid   = slv("Servers, mile 8-18",               4, 28, "servers_mid")
    serv_late  = slv("Servers, mile > 18",               4, 24, "servers_late")
    aid_pj     = slv("Porta-johns per station",          2, 16, "aid_pj")
    prep_w     = slv("Water prep rate (cups/min, early)",     80, 600, "water_prep", step=10)
    prep_e     = slv("Electrolyte prep rate (cups/min, early)", 50, 350, "electrolyte_prep", step=10,
                     help="Mile 2.5 is the binding stockout location.")
    init_buf   = st.sidebar.slider("Initial water buffer (cups)", 50, 400, 200, 20, key=p("buf"))

    st.sidebar.header("🚑 Medical (bike teams)")
    bike_mode_options = [
        "BSIM zones (time-windowed, 8 teams)",
        "Evenly-spaced (full race coverage)",
    ]
    default_mode_idx = 0
    if preset is not None and preset.get("bike_mode", "").startswith("Even"):
        default_mode_idx = 1
    bike_mode = st.sidebar.selectbox(
        "Bike team configuration", bike_mode_options,
        index=default_mode_idx, key=p("bike_mode"),
        help="BSIM zones = 7 zones (Z4 has 2 teams) with time-windowed coverage. "
             "Evenly-spaced = N teams on duty entire race, equally distributed.")
    if bike_mode.startswith("Even"):
        n_even_teams = slv("Number of teams (even-spaced)", 8, 40, "n_even_teams")
    else:
        n_even_teams = 8   # BSIM has 8 fixed teams
        st.sidebar.caption("ℹ️ BSIM mode uses 8 fixed teams: Z1, Z2, Z3, Z4S, Z4N, Z5, Z6, Z7 with documented time windows.")
    patrol_sp   = slv("Bike patrol speed (mph)",     8, 18, "bike_patrol_speed",
                     help="Cruising pace while patrolling assigned zone.")
    response_sp = slv("Bike response speed (mph)", 10, 24, "bike_response_speed",
                     help="Sprint pace when called to an injury.")
    n_ambulances = slv("Ambulances", 2, 10, "n_ambulances",
                       help="Ambulances transport HIGH-severity injuries (counted as DNF).")

    # ---- Advanced: injury severity (calibrated by default) ----
    with st.sidebar.expander("⚙️ Advanced: injury severity (calibrated)"):
        st.caption("Defaults match BSIM history: ~3-4 DNFs per race (vs documented 0-4/yr).")
        p_high = st.slider("p(HIGH severity)  •  → DNF", 0.005, 0.10, 0.02, 0.005,
                            key=p("p_high"), format="%.3f",
                            help="Fraction of injuries that become DNF. "
                                 "Default 0.02 (2%) calibrates to BSIM 0-4 DNF/yr history. "
                                 "Set to 0.10 (legacy default) to see the over-prediction.")
        p_medium = st.slider("p(MEDIUM severity)", 0.05, 0.40, 0.10, 0.01,
                              key=p("p_medium"), format="%.2f",
                              help="On-scene treatment; runner continues.")
        st.caption(f"p(MINOR) = {1 - p_high - p_medium:.2f}  (remaining)")

    st.sidebar.header("🚌 Start area & bus")
    start_pj = slv("Porta-johns at start (real: 146)", 30, 200, "start_pj")
    st.sidebar.markdown("**Bus check-in (mandatory queue)**")
    bus_vol  = slv("Volunteers per stop", 1, 8, "bus_volunteers",
                  help="Each volunteer checks ticket + name. Runners CANNOT leave the queue.")
    bus_svc  = slv("Check-in service time (sec/runner)", 5, 30, "bus_checkin_sec",
                  help="18 sec = manual ticket + name. ~5 sec = barcode scanner.")

    # ---- Build configs ----
    if bike_mode.startswith("Even"):
        bike_teams = make_even_bike_teams(n_even_teams)
    else:
        bike_teams = DEFAULT_BIKE_TEAMS

    def build_stations():
        out = []
        for m in AID_STATION_MILES:
            if   m < 8:  sv, wp, ep, bf = serv_early, prep_w, prep_e, init_buf
            elif m < 18: sv, wp, ep, bf = serv_mid, max(150, int(prep_w*0.55)), max(80, int(prep_e*0.6)), max(150, int(init_buf*0.9))
            else:        sv, wp, ep, bf = serv_late, max(100, int(prep_w*0.35)), max(60, int(prep_e*0.4)), max(120, int(init_buf*0.75))
            out.append(AidStationConfig(
                mile=m, servers=sv, porta_johns=aid_pj,
                trash_capacity=int(1.6 * n_runners),
                water_prep_per_min=wp, electrolyte_prep_per_min=ep, snacks_prep_per_min=110,
                water_ready_init=bf, water_buffer_capacity=max(bf*2, 250)))
        return out

    def build_bus_stops():
        return tuple(BusStopConfig(
            name=s.name, n_runners=s.n_runners,
            arrival_start_min=s.arrival_start_min, arrival_end_min=s.arrival_end_min,
            travel_time_mean_min=s.travel_time_mean_min, travel_time_std_min=s.travel_time_std_min,
            bus_dispatch_times=s.bus_dispatch_times, bus_capacity=s.bus_capacity,
            check_in_volunteers=int(bus_vol), check_in_mean_sec=float(bus_svc))
            for s in DEFAULT_BUS_STOPS)

    # Build severity tuple from the user-set p_high and p_medium
    p_minor = max(0.01, 1.0 - p_high - p_medium)
    severity_tuple = (("minor", p_minor), ("medium", p_medium), ("high", p_high))

    return dict(
        n_runners=n_runners, base_seed=int(base_seed),
        station_cfgs=build_stations(),
        med_cfg=MedicalConfig(
            bike_teams=bike_teams,
            bike_patrol_speed_mph=float(patrol_sp),
            bike_response_speed_mph=float(response_sp),
            num_ambulances=n_ambulances,
            num_bike_medics=len(bike_teams),
            severity_prob=severity_tuple),
        start_area_cfg=StartAreaConfig(porta_johns=start_pj),
        corral_plan=CorralPlan.custom(n_corrals, interval) if interval > 0
                    else CorralPlan.custom(1, 0.0),
        bus_stop_cfgs=build_bus_stops(),
        preset_name=preset_name,
        bike_mode=bike_mode,
    )

# =============================================================================
# Pretty target evaluation table
# =============================================================================
def render_target_table(metrics: dict, container=st):
    """Show targets with PASS/WARN/FAIL coloring."""
    container.subheader("Performance against targets")
    rows = []
    color_map = {"PASS": "✅", "WARN": "🟨", "FAIL": "❌"}
    for key, tgt in DEFAULT_TARGETS.items():
        if key not in metrics: continue
        v = metrics[key]
        if v <= tgt.target_value:   status = "PASS"
        elif v <= tgt.failure_value: status = "WARN"
        else:                       status = "FAIL"
        rows.append({
            "Status": color_map[status],
            "Metric": tgt.name,
            "Value":  f"{v:.2f}",
            "Target ≤": f"{tgt.target_value}",
            "Fail ≥":   f"{tgt.failure_value}",
        })
    container.table(pd.DataFrame(rows))

# =============================================================================
# TAB 1 — Single Scenario
# =============================================================================
def tab_single():
    st.markdown("### Single scenario — Monte Carlo")
    st.caption("Set parameters in the sidebar; run a Monte Carlo across many simulated race days.")
    cfg = parameter_sidebar("single")
    bike_mode = cfg.pop("bike_mode")
    n_teams = len(cfg["med_cfg"].bike_teams)
    st.info(f"**Preset:** {cfg.pop('preset_name')}  •  "
            f"Field: {cfg['n_runners']} marathoners  •  "
            f"Bikes: {n_teams} teams ({bike_mode.split(' (')[0]})  •  "
            f"Bus: {cfg['bus_stop_cfgs'][0].check_in_volunteers} vol/stop @ {cfg['bus_stop_cfgs'][0].check_in_mean_sec:.0f} sec")

    n_reps = st.slider("Monte Carlo replications", 1, 25, 6, key="single_reps",
                       help="More reps = tighter confidence intervals. ~5-8 sec/rep at 3,500 runners.")
    if st.button("Run scenario", type="primary"):
        with st.spinner(f"Running {n_reps} replications..."):
            t0 = time.time()
            df = run_monte_carlo(n_reps=n_reps, seed=cfg["base_seed"],
                                 n_runners=cfg["n_runners"], station_cfgs=cfg["station_cfgs"],
                                 med_cfg=cfg["med_cfg"], start_area_cfg=cfg["start_area_cfg"],
                                 corral_plan=cfg["corral_plan"],
                                 bus_stop_cfgs=cfg["bus_stop_cfgs"],
                                 monitor_interval_min=9999.0)
            elapsed = time.time() - t0
        st.success(f"Completed in {elapsed:.1f} s")

        # Top-line summary cards
        m = {c: df[c].mean() for c in df.columns if c not in ("rep", "n_finishers")}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Med HIGH p90", f"{m.get('med_resp_p90_high_min', 0):.1f} min",
                  delta=f"target ≤2", delta_color="off")
        c2.metric("Bus check-in AVG", f"{m.get('bus_checkin_wait_mean_min', 0):.1f} min",
                  delta=f"p95: {m.get('bus_checkin_wait_p95_min', 0):.0f} min", delta_color="off")
        c3.metric("DNF (severe injury)", f"{m.get('dnf_due_to_injury_count', 0):.1f} runners",
                  delta="BSIM history: 0-4/yr", delta_color="off")
        c4.metric("Late to corral", f"{m.get('late_to_corral_count', 0):.0f} runners",
                  delta="bus → start timing", delta_color="off")

        metrics = {c: df[c].mean() for c in df.columns if c not in ("rep", "n_finishers")}
        render_target_table(metrics)

        # Distribution plots — updated to surface the new metrics
        st.subheader("Distribution of outcomes across reps")
        cols_to_plot = [
            ("med_resp_p90_high_min",     "Medical HIGH (p90 min)",       2.0),
            ("med_resp_p90_minor_min",    "Medical MINOR (p90 min)",      5.0),
            ("bus_checkin_wait_mean_min", "Bus check-in AVG (min)",       None),
            ("bus_checkin_wait_p95_min",  "Bus check-in p95 (min)",       15.0),
            ("dnf_due_to_injury_count",   "DNF (severe injury, runners)", None),
            ("overflow_bus_count",        "Overflow bus boardings",       None),
            ("porta_balk_count",          "Porta-john balks",             None),
            ("stockout_count",            "Supply stockouts",             None),
        ]
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
        for ax, (col, title, tgt) in zip(axes.flat, cols_to_plot):
            if col not in df.columns:
                ax.set_visible(False); continue
            v = df[col].values
            ax.violinplot([v], showmeans=True, showextrema=True)
            ax.scatter(np.random.normal(1, 0.04, len(v)), v, s=30, alpha=0.7, color="#0D2845")
            if tgt is not None:
                ax.axhline(tgt, ls="--", color="green", lw=1.5, label=f"target {tgt}")
                ax.legend(fontsize=8)
            ax.set_title(f"{title}\nmean {v.mean():.1f}", fontsize=9.5)
            ax.set_xticks([]); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

        st.download_button("Download raw results (CSV)",
                           df.to_csv(index=False), file_name="single_scenario.csv")

# =============================================================================
# TAB 2 — Compare Scenarios (paired-seed)
# =============================================================================
def build_config_from_preset(preset_name: str, n_runners: int) -> dict:
    """Construct a full run config from a preset name."""
    p = PRESETS[preset_name]
    if p is None:
        raise ValueError("Custom preset not supported in compare; use Single tab.")
    # Bike teams
    if p["bike_mode"].startswith("Even"):
        bike_teams = make_even_bike_teams(p["n_even_teams"])
    else:
        bike_teams = DEFAULT_BIKE_TEAMS
    # Aid stations
    stations = []
    for m in AID_STATION_MILES:
        if   m < 8:  sv, wp, ep, bf = p["servers_early"], p["water_prep"], p["electrolyte_prep"], 200
        elif m < 18: sv, wp, ep, bf = p["servers_mid"], max(150, int(p["water_prep"]*0.55)), max(80, int(p["electrolyte_prep"]*0.6)), 180
        else:        sv, wp, ep, bf = p["servers_late"], max(100, int(p["water_prep"]*0.35)), max(60, int(p["electrolyte_prep"]*0.4)), 150
        stations.append(AidStationConfig(
            mile=m, servers=sv, porta_johns=p["aid_pj"],
            trash_capacity=int(1.6 * n_runners),
            water_prep_per_min=wp, electrolyte_prep_per_min=ep, snacks_prep_per_min=110,
            water_ready_init=bf, water_buffer_capacity=max(bf*2, 250)))
    # Bus stops with preset's check-in staffing
    bus_stops = tuple(BusStopConfig(
        name=s.name, n_runners=s.n_runners,
        arrival_start_min=s.arrival_start_min, arrival_end_min=s.arrival_end_min,
        travel_time_mean_min=s.travel_time_mean_min, travel_time_std_min=s.travel_time_std_min,
        bus_dispatch_times=s.bus_dispatch_times, bus_capacity=s.bus_capacity,
        check_in_volunteers=p["bus_volunteers"], check_in_mean_sec=float(p["bus_checkin_sec"]))
        for s in DEFAULT_BUS_STOPS)
    return dict(
        station_cfgs=stations,
        med_cfg=MedicalConfig(
            bike_teams=bike_teams,
            bike_patrol_speed_mph=float(p["bike_patrol_speed"]),
            bike_response_speed_mph=float(p["bike_response_speed"]),
            num_ambulances=p["n_ambulances"],
            num_bike_medics=len(bike_teams)),
        start_area_cfg=StartAreaConfig(porta_johns=p["start_pj"]),
        corral_plan=CorralPlan.custom(p["n_corrals"], p["interval"]),
        bus_stop_cfgs=bus_stops,
    )

def _preset_summary(name: str) -> str:
    p = PRESETS[name]
    if p is None: return ""
    bike_desc = (f"{p['n_even_teams']} even-spaced teams" if p["bike_mode"].startswith("Even")
                 else "8 BSIM zone teams (time-windowed)")
    return (f"Aid {p['servers_early']}/{p['servers_mid']}/{p['servers_late']} svr, "
            f"{p['aid_pj']} porta. {bike_desc}, {p['n_ambulances']} amb. "
            f"Bus {p['bus_volunteers']} vol @ {p['bus_checkin_sec']}s.")

def tab_compare():
    st.markdown("### Compare two scenarios with paired seeds")
    st.caption("Common Random Numbers: both configs run at the same seeds, so the difference "
               "comes from the **change**, not noise.")
    preset_names = [k for k, v in PRESETS.items() if v is not None]
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Configuration A")
        nameA = st.selectbox("Preset A", preset_names, index=0, key="cmp_A")
        st.caption(_preset_summary(nameA))
    with cB:
        st.subheader("Configuration B")
        nameB = st.selectbox("Preset B", preset_names, index=1, key="cmp_B")
        st.caption(_preset_summary(nameB))

    c1, c2 = st.columns([1, 1])
    with c1:
        n_runners = st.slider("Field size (both configs)", 1500, 4500, 3500, 250, key="cmp_n")
    with c2:
        n_pairs = st.slider("Paired replications", 4, 25, 8, key="cmp_pairs")

    metric_keys = st.multiselect(
        "Metrics to compare (statistical tests run on these)",
        ["med_resp_p90_high_min", "med_resp_p90_minor_min", "aid_wait_p95_sec",
         "porta_balk_count", "stockout_count",
         "bus_checkin_wait_mean_min", "bus_checkin_wait_p95_min",
         "dnf_due_to_injury_count", "overflow_bus_count"],
        default=["med_resp_p90_high_min", "bus_checkin_wait_mean_min",
                 "bus_checkin_wait_p95_min", "dnf_due_to_injury_count", "overflow_bus_count"],
        key="cmp_metrics")

    if st.button("Run paired comparison", type="primary"):
        with st.spinner(f"Running {n_pairs} paired reps for each config..."):
            t0 = time.time()
            cfgA = build_config_from_preset(nameA, n_runners)
            cfgB = build_config_from_preset(nameB, n_runners)
            df = paired_seed_compare(cfgA, cfgB, n_pairs=n_pairs, n_runners=n_runners,
                                     label_A=nameA.split(" (")[0][:12],
                                     label_B=nameB.split(" (")[0][:12],
                                     metric_keys=metric_keys)
            elapsed = time.time() - t0
        st.success(f"Completed in {elapsed:.0f} s")
        st.subheader("Paired-seed results")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download (CSV)", df.to_csv(index=False),
                           file_name="paired_comparison.csv")

# =============================================================================
# TAB 3 — Sensitivity Sweep
# =============================================================================
SWEEP_DEFAULTS = {
    "Bike teams (even-spaced count)":  "8, 16, 24, 32, 40",
    "Bus check-in volunteers/stop":    "1, 2, 3, 4, 5, 6",
    "Bus check-in service time (sec)": "5, 10, 15, 18, 25",
    "Porta-johns per station":         "4, 6, 8, 10, 12, 16",
    "Aid servers, early (mile < 8)":   "6, 12, 18, 24, 30",
    "Electrolyte prep rate (early)":   "60, 90, 150, 200, 300",
    "Bike response speed (mph)":       "12, 14, 16, 18, 20",
    "p(HIGH severity)  •  → DNF":      "0.01, 0.02, 0.04, 0.06, 0.10",
}

def tab_sweep():
    st.markdown("### Sensitivity sweep")
    st.caption("Pick a parameter, pick a range, see how the outcome metric responds. "
               "Find the elbow point.")
    cfg = parameter_sidebar("sweep")
    cfg.pop("bike_mode")

    parameter = st.selectbox("Parameter to sweep", list(SWEEP_DEFAULTS.keys()))
    vals_txt = st.text_input("Values to test (comma-separated)", value=SWEEP_DEFAULTS[parameter])
    n_reps = st.slider("Reps per value", 2, 8, 3, key="sweep_reps")
    metric = st.selectbox("Metric to plot",
        ["med_resp_p90_high_min", "med_resp_p90_minor_min",
         "bus_checkin_wait_mean_min", "bus_checkin_wait_p95_min",
         "dnf_due_to_injury_count", "overflow_bus_count",
         "porta_balk_count", "stockout_count", "aid_wait_p95_sec"])

    if st.button("Run sweep", type="primary"):
        try:
            vals = [float(v.strip()) for v in vals_txt.split(",") if v.strip()]
        except ValueError:
            st.error("Couldn't parse values."); return
        rows = []
        prog = st.progress(0, text="Starting sweep...")
        total = len(vals) * n_reps
        done = 0
        for v in vals:
            # Build a copy of cfg with the swept parameter modified
            local = {k: c for k, c in cfg.items() if k not in ("base_seed", "preset_name")}
            if parameter == "Bike teams (even-spaced count)":
                nt = int(v)
                old = local["med_cfg"]
                local["med_cfg"] = MedicalConfig(
                    bike_teams=make_even_bike_teams(nt),
                    bike_patrol_speed_mph=old.bike_patrol_speed_mph,
                    bike_response_speed_mph=old.bike_response_speed_mph,
                    num_ambulances=old.num_ambulances,
                    num_bike_medics=nt)
            elif parameter == "Bus check-in volunteers/stop":
                local["bus_stop_cfgs"] = tuple(BusStopConfig(
                    name=s.name, n_runners=s.n_runners,
                    arrival_start_min=s.arrival_start_min, arrival_end_min=s.arrival_end_min,
                    travel_time_mean_min=s.travel_time_mean_min, travel_time_std_min=s.travel_time_std_min,
                    bus_dispatch_times=s.bus_dispatch_times, bus_capacity=s.bus_capacity,
                    check_in_volunteers=int(v), check_in_mean_sec=s.check_in_mean_sec)
                    for s in local["bus_stop_cfgs"])
            elif parameter == "Bus check-in service time (sec)":
                local["bus_stop_cfgs"] = tuple(BusStopConfig(
                    name=s.name, n_runners=s.n_runners,
                    arrival_start_min=s.arrival_start_min, arrival_end_min=s.arrival_end_min,
                    travel_time_mean_min=s.travel_time_mean_min, travel_time_std_min=s.travel_time_std_min,
                    bus_dispatch_times=s.bus_dispatch_times, bus_capacity=s.bus_capacity,
                    check_in_volunteers=s.check_in_volunteers, check_in_mean_sec=float(v))
                    for s in local["bus_stop_cfgs"])
            elif parameter == "Porta-johns per station":
                local["station_cfgs"] = [
                    AidStationConfig(mile=s.mile, servers=s.servers, porta_johns=int(v),
                        trash_capacity=s.trash_capacity, water_prep_per_min=s.water_prep_per_min,
                        electrolyte_prep_per_min=s.electrolyte_prep_per_min,
                        snacks_prep_per_min=s.snacks_prep_per_min,
                        water_ready_init=s.water_ready_init,
                        water_buffer_capacity=s.water_buffer_capacity)
                    for s in local["station_cfgs"]]
            elif parameter == "Aid servers, early (mile < 8)":
                local["station_cfgs"] = [
                    AidStationConfig(mile=s.mile, servers=int(v) if s.mile<8 else s.servers,
                        porta_johns=s.porta_johns, trash_capacity=s.trash_capacity,
                        water_prep_per_min=s.water_prep_per_min,
                        electrolyte_prep_per_min=s.electrolyte_prep_per_min,
                        snacks_prep_per_min=s.snacks_prep_per_min,
                        water_ready_init=s.water_ready_init,
                        water_buffer_capacity=s.water_buffer_capacity)
                    for s in local["station_cfgs"]]
            elif parameter == "Electrolyte prep rate (early)":
                local["station_cfgs"] = [
                    AidStationConfig(mile=s.mile, servers=s.servers, porta_johns=s.porta_johns,
                        trash_capacity=s.trash_capacity, water_prep_per_min=s.water_prep_per_min,
                        electrolyte_prep_per_min=int(v) if s.mile<8 else s.electrolyte_prep_per_min,
                        snacks_prep_per_min=s.snacks_prep_per_min,
                        water_ready_init=s.water_ready_init,
                        water_buffer_capacity=s.water_buffer_capacity)
                    for s in local["station_cfgs"]]
            elif parameter == "Bike response speed (mph)":
                old = local["med_cfg"]
                local["med_cfg"] = MedicalConfig(
                    bike_teams=old.bike_teams,
                    bike_patrol_speed_mph=old.bike_patrol_speed_mph,
                    bike_response_speed_mph=float(v),
                    num_ambulances=old.num_ambulances,
                    num_bike_medics=old.num_bike_medics,
                    severity_prob=old.severity_prob)
            elif parameter.startswith("p(HIGH"):
                # Sweep p(HIGH) — keep p(MEDIUM)=0.10, distribute remainder to MINOR
                old = local["med_cfg"]
                p_high = float(v); p_med = 0.10
                p_min = max(0.01, 1.0 - p_high - p_med)
                local["med_cfg"] = MedicalConfig(
                    bike_teams=old.bike_teams,
                    bike_patrol_speed_mph=old.bike_patrol_speed_mph,
                    bike_response_speed_mph=old.bike_response_speed_mph,
                    num_ambulances=old.num_ambulances,
                    num_bike_medics=old.num_bike_medics,
                    severity_prob=(("minor", p_min), ("medium", p_med), ("high", p_high)))
            for r in range(n_reps):
                run_kwargs = {k: cc for k, cc in local.items() if k != "n_runners"}
                m, _ = run_marathon_full(seed=100+r, n_runners=cfg["n_runners"],
                                          **run_kwargs, monitor_interval_min=9999.0)
                rows.append({"value": v, "rep": r, **_extract_metrics(m)})
                done += 1; prog.progress(done/total, text=f"{done}/{total} runs")
        prog.empty()
        df = pd.DataFrame(rows)
        agg = df.groupby("value")[metric].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(agg["value"], agg["mean"], yerr=agg["std"], fmt="o-", lw=2,
                    capsize=5, color="#1565C0", markersize=8)
        ax.set_xlabel(parameter, fontsize=11)
        metric_label = DEFAULT_TARGETS.get(metric, type("T", (), {"name": metric})).name
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if metric in DEFAULT_TARGETS:
            ax.axhline(DEFAULT_TARGETS[metric].target_value, ls="--", color="green",
                       label=f"target {DEFAULT_TARGETS[metric].target_value}")
            ax.legend()
        ax.set_title(f"{parameter} vs. {metric_label}", fontweight="bold")
        st.pyplot(fig)
        st.dataframe(agg, use_container_width=True)
        st.download_button("Download (CSV)", df.to_csv(index=False), file_name="sweep.csv")

# =============================================================================
# Main
# =============================================================================
t1, t2, t3 = st.tabs(["🎯 Single Scenario", "⚖️ Compare Scenarios", "📊 Sensitivity Sweep"])
with t1: tab_single()
with t2: tab_compare()
with t3: tab_sweep()

st.markdown("---")
st.caption("Big Sur Marathon Simulator • Team Bixby Bridge Beer Bottleneck • "
           "Mandatory bus queue (no balking) • Bike teams patrol BSIM zones at 12 mph, sprint at 16 mph • "
           "Severe injuries → ambulance → DNF.")
