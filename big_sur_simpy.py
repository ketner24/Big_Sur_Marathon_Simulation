"""
Big Sur Marathon -- SimPy model v2
Team Bixby Bridge Beer Bottleneck

New in v2:
  - Elevation-aware pacing via Minetti energy-cost model + downhill cap
  - Three-tier injury severity (minor/medium/high) with different paths
  - Two-pool medical (bike medics first-responders + ambulances for transport)
  - Porta-johns as aid-station Resource
  - Trash capacity tracking with overflow events
  - server_sweep() helper to tune aid-station manning
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
import pandas as pd
import simpy
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Fitted finish-time distribution (Weibull from v1)
# ---------------------------------------------------------------------------
WEIBULL_C     = 3.481957
WEIBULL_LOC   = 137.042230
WEIBULL_SCALE = 154.822742
CUTOFF_MIN    = 6 * 60                # 6-hr permit cutoff in minutes

def sample_finish_time(rng: np.random.Generator) -> float:
    dist = stats.weibull_min(WEIBULL_C, loc=WEIBULL_LOC, scale=WEIBULL_SCALE)
    while True:
        t = dist.rvs(random_state=rng)
        if t <= CUTOFF_MIN:
            return float(t)


# ---------------------------------------------------------------------------
# 2. Course geometry: elevation profile + grade-adjusted pace
# ---------------------------------------------------------------------------
# Approximate elevation profile based on BSIM published course data:
#   Hurricane Point summit 560 ft at mile 12; climb begins mile 9.85
#   Bixby Bridge at mile 13.1, ~260 ft
#   Documented punchy climbs at miles 14.46, 15.59, 17.42, 18.55, 19.16, 19.78
#   Carmel Highlands rollers mile 22-25
#   Totals: ~2180 ft gain, ~2530 ft loss
# To swap in real GPX data: parse to (mile, elev_ft) tuples; nothing else changes.
ELEVATION_KNOTS_FT = [
    ( 0.00, 155), ( 1.00, 100), ( 3.00,  50), ( 5.00,  40), ( 7.00,  80),
    ( 8.00, 180), ( 9.00, 110), ( 9.85, 100), (11.00, 330), (12.00, 560),
    (13.10, 260), (14.00, 180), (14.46, 270), (15.18, 200), (15.59, 290),
    (16.50, 200), (17.42, 280), (18.55, 220), (19.16, 240), (19.78, 290),
    (21.00, 150), (22.50, 250), (24.00, 180), (25.00, 100), (26.20,  30),
]
COURSE_MILES = 26.2


def elevation_at_mile(m: float) -> float:
    if m <= ELEVATION_KNOTS_FT[0][0]:  return ELEVATION_KNOTS_FT[0][1]
    if m >= ELEVATION_KNOTS_FT[-1][0]: return ELEVATION_KNOTS_FT[-1][1]
    for (m0, e0), (m1, e1) in zip(ELEVATION_KNOTS_FT, ELEVATION_KNOTS_FT[1:]):
        if m0 <= m <= m1:
            return e0 + (e1 - e0) * (m - m0) / (m1 - m0)
    return ELEVATION_KNOTS_FT[-1][1]


def grade_at_mile(m: float) -> float:
    """Grade as a fraction (rise/run); piecewise constant within each segment."""
    if m <= ELEVATION_KNOTS_FT[0][0]:  return 0.0
    if m >= ELEVATION_KNOTS_FT[-1][0]: return 0.0
    for (m0, e0), (m1, e1) in zip(ELEVATION_KNOTS_FT, ELEVATION_KNOTS_FT[1:]):
        if m0 <= m <= m1:
            return (e1 - e0) / ((m1 - m0) * 5280.0)
    return 0.0


def pace_multiplier(grade: float) -> float:
    """Minetti energy-cost-of-locomotion ratio relative to flat.
    Source: Minetti et al. (2002) J. Appl. Physiol.

    Downhill speedup capped at 0.75x (no faster than ~33% relative to flat)
    because runners don't reap the full energetic benefit of steep descents
    (eccentric muscle damage, footing control).
    """
    g = grade
    c = (155.4 * g**5 - 30.4 * g**4 - 43.3 * g**3
         + 46.3 * g**2 + 19.5 * g + 3.6)
    return max(c / 3.6, 0.75)


def _course_integral(dx: float = 0.05) -> float:
    """Effective number of 'flat miles' the course represents."""
    total, m = 0.0, 0.0
    while m < COURSE_MILES:
        step = min(dx, COURSE_MILES - m)
        total += pace_multiplier(grade_at_mile(m + step/2)) * step
        m += step
    return total

# Computed once at import. Used to convert sampled finish time -> flat pace.
COURSE_FLAT_EQUIVALENT_MILES = _course_integral()


def leg_time(start_mi: float, end_mi: float, flat_pace_min_per_mi: float,
             dx: float = 0.05) -> float:
    """Integrate grade-adjusted pace over [start_mi, end_mi]."""
    total, m = 0.0, start_mi
    while m < end_mi:
        step = min(dx, end_mi - m)
        total += flat_pace_min_per_mi * pace_multiplier(grade_at_mile(m + step/2)) * step
        m += step
    return total


# ---------------------------------------------------------------------------
# 3. Aid stations and corrals
# ---------------------------------------------------------------------------
AID_STATION_MILES = [2.5, 4.8, 7.8, 10.4, 12.2, 14.7, 16.9, 19.0, 21.2, 23.0, 24.5]

@dataclass
class AidStationConfig:
    mile: float
    servers:             int   = 6
    porta_johns:         int   = 4
    trash_capacity:      int   = 400          # # discrete items before overflow
    water_cups:          int   = 4000
    electrolyte_cups:    int   = 2500
    snacks:              int   = 3500
    # Per-runner means
    p_uses_porta_john:           float = 0.10
    porta_john_service_mean_min: float = 1.8
    porta_john_service_std_min:  float = 0.7
    cups_water_per_runner:       float = 1.2
    cups_electrolyte_per_runner: float = 0.5
    snacks_per_runner:           float = 0.5
    trash_per_runner:            float = 1.5
    refill_water_per_min:        float = 100.0   # continuous; replace w/ truck arrivals later


# ---------------------------------------------------------------------------
# 3a. Corral plans (the *new* way to specify the start structure)
# ---------------------------------------------------------------------------
@dataclass
class CorralPlan:
    """Describes how the field is split into corrals and when each wave starts.

    Field is sorted by sampled finish time (fastest first) and binned into corrals
    using `shares`. Corral i starts at `start_offsets_min[i]` minutes after race start.
    """
    names:              tuple    # e.g., ("A", "B", "C")
    shares:             tuple    # population fractions per corral; must sum to ~1
    start_offsets_min:  tuple    # one offset per corral

    def __post_init__(self):
        assert len(self.names) == len(self.shares) == len(self.start_offsets_min), \
            "names, shares, start_offsets_min must all have the same length"
        s = sum(self.shares)
        assert abs(s - 1.0) < 1e-6, f"shares must sum to 1; got {s}"

    @classmethod
    def baseline_3x5(cls):
        """Project baseline: 3 corrals (A/B/C), equal thirds, 5-min intervals.
        Corral A = fastest third, expected sub-4hr; B = 4-5hr; C = 5-6hr+."""
        return cls(names=("A","B","C"),
                   shares=(1/3, 1/3, 1/3),
                   start_offsets_min=(0.0, 5.0, 10.0))

    @classmethod
    def five_corrals(cls, interval_min: float = 3.0):
        """5 corrals, equal fifths, constant interval."""
        return cls(names=tuple("ABCDE"),
                   shares=(0.2,)*5,
                   start_offsets_min=tuple(i*interval_min for i in range(5)))

    @classmethod
    def ten_corrals(cls, interval_min: float = 2.0):
        return cls(names=tuple("ABCDEFGHIJ"),
                   shares=(0.1,)*10,
                   start_offsets_min=tuple(i*interval_min for i in range(10)))

    @classmethod
    def custom(cls, n_corrals: int, interval_min: float, shares: tuple = None):
        names = tuple(chr(ord('A') + i) for i in range(n_corrals))
        if shares is None:
            shares = (1.0 / n_corrals,) * n_corrals
        return cls(names=names, shares=shares,
                   start_offsets_min=tuple(i*interval_min for i in range(n_corrals)))


def assign_corrals(finish_times: np.ndarray, plan: CorralPlan) -> np.ndarray:
    """Sort runners by finish time; bin into corrals per `plan.shares`.
    Returns: int array of corral indices (0-based), same length as finish_times."""
    n = len(finish_times)
    order = np.argsort(finish_times)
    sizes = [int(round(s * n)) for s in plan.shares]
    sizes[-1] += n - sum(sizes)          # absorb rounding into last corral
    out = np.empty(n, dtype=int)
    pos = 0
    for c, sz in enumerate(sizes):
        out[order[pos : pos + sz]] = c
        pos += sz
    return out


# ---------------------------------------------------------------------------
# 4. Medical model
# ---------------------------------------------------------------------------
@dataclass
class MedicalConfig:
    num_ambulances:        int   = 4
    num_bike_medics:       int   = 12
    ambulance_locations:   tuple = (3.0, 10.0, 17.0, 24.0)
    bike_locations:        tuple = (2.5, 7.0, 10.4, 13.1, 16.9, 19.0, 21.2, 24.5)
    bike_speed_mph:        float = 12.0
    ambulance_speed_mph:   float = 20.0
    # severity -> probability
    severity_prob:         tuple = (("minor", 0.65), ("medium", 0.25), ("high", 0.10))
    # On-scene treatment time (mean, std), lognormal
    treat_min_params:      tuple = (3.0, 1.5)
    treat_med_params:      tuple = (8.0, 4.0)
    stabilize_params:      tuple = (5.0, 2.0)    # bike stabilization before amb
    transport_params:      tuple = (15.0, 5.0)   # ambulance transport + offload

INJURY_RATE_PER_RUNNER = 0.05      # FRACTION OF RUNNERS REQUIRING ON-COURSE INTERVENTION
                                   # NOTE: the project doc's 15-30% number is "any reported injury
                                   # including post-race blisters/soreness". The fraction needing
                                   # active on-course medical contact is more like 3-8%. Tune this.


def _lognormal(rng, mean, std):
    sigma = math.sqrt(math.log(1 + (std/mean)**2))
    mu    = math.log(mean) - 0.5 * sigma**2
    return float(rng.lognormal(mu, sigma))

def sample_severity(rng, severity_prob):
    r, cum = rng.random(), 0.0
    for name, p in severity_prob:
        cum += p
        if r < cum:
            return name
    return severity_prob[-1][0]

def nearest_dist(mile, locations):
    return min(abs(mile - loc) for loc in locations)


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
@dataclass
class RaceMetrics:
    finishes:             list = field(default_factory=list)   # (bib, t_finish_min)
    injuries:             list = field(default_factory=list)   # list of dicts
    aid_wait_min:         list = field(default_factory=list)
    porta_wait_min:       list = field(default_factory=list)
    stockouts:            list = field(default_factory=list)   # (mile, item, t)
    trash_overflows:      list = field(default_factory=list)   # (mile, t)
    # Pre-race metrics:
    bus_wait_min:         list = field(default_factory=list)   # (stop_name, wait_min)
    bus_travel_min:       list = field(default_factory=list)   # (stop_name, travel_min)
    start_porta_wait_min: list = field(default_factory=list)
    late_to_corral:       list = field(default_factory=list)   # (bib, min_late)
    stranded_at_busstop:  list = field(default_factory=list)   # (bib, stop_name)
    # Timeline tracking (for event-timeline visualization):
    first_at_station:     dict = field(default_factory=dict)   # mile -> first arrival time
    last_at_station:      dict = field(default_factory=dict)   # mile -> last arrival time
    first_bus_at_start:   dict = field(default_factory=dict)   # stop -> arrival time
    last_bus_at_start:    dict = field(default_factory=dict)   # stop -> arrival time
    injury_events:        list = field(default_factory=list)   # (t, mile, severity)


# ---------------------------------------------------------------------------
# 6. AidStation
# ---------------------------------------------------------------------------
class AidStation:
    def __init__(self, env, cfg: AidStationConfig, metrics: RaceMetrics):
        self.env = env
        self.cfg = cfg
        self.metrics = metrics
        self.servers     = simpy.Resource(env, capacity=cfg.servers)
        self.porta_johns = simpy.Resource(env, capacity=cfg.porta_johns)
        self.water       = simpy.Container(env, init=cfg.water_cups,
                                           capacity=cfg.water_cups + 20000)
        self.electrolyte = simpy.Container(env, init=cfg.electrolyte_cups,
                                           capacity=cfg.electrolyte_cups + 20000)
        self.snacks      = simpy.Container(env, init=cfg.snacks,
                                           capacity=cfg.snacks + 20000)
        self._trash_level = 0
        env.process(self._refill_water())

    def _refill_water(self):
        while True:
            yield self.env.timeout(1.0)
            add = min(self.cfg.refill_water_per_min,
                      self.water.capacity - self.water.level)
            if add > 0:
                self.water.put(add)

    def _add_trash(self, amt: int):
        before, after = self._trash_level, self._trash_level + amt
        if after > self.cfg.trash_capacity and before <= self.cfg.trash_capacity:
            self.metrics.trash_overflows.append((self.cfg.mile, self.env.now))
        self._trash_level = after

    def visit(self, bib, rng):
        t_arrive = self.env.now
        m = self.cfg.mile
        if m not in self.metrics.first_at_station:
            self.metrics.first_at_station[m] = t_arrive
        self.metrics.last_at_station[m] = t_arrive

        # ----- aid line -----
        with self.servers.request() as req:
            yield req
            self.metrics.aid_wait_min.append(self.env.now - t_arrive)

            for cont, mean, label in [
                (self.water,       self.cfg.cups_water_per_runner,       "water"),
                (self.electrolyte, self.cfg.cups_electrolyte_per_runner, "electrolyte"),
                (self.snacks,      self.cfg.snacks_per_runner,           "snack"),
            ]:
                want = int(rng.poisson(mean))
                if want <= 0:
                    continue
                if cont.level >= want:
                    yield cont.get(want)
                else:
                    self.metrics.stockouts.append((self.cfg.mile, label, self.env.now))

            self._add_trash(int(rng.poisson(self.cfg.trash_per_runner)))

            # Realistic aid-station handoff: ~1-4 sec (volunteer hands cup, runner grabs and goes).
            # Was 8-15 sec previously which conflated handoff with consumption.
            yield self.env.timeout(rng.uniform(0.017, 0.067))

        # ----- porta-johns (stochastic) -----
        if rng.random() < self.cfg.p_uses_porta_john:
            t_pj = self.env.now
            with self.porta_johns.request() as req:
                yield req
                self.metrics.porta_wait_min.append(self.env.now - t_pj)
                yield self.env.timeout(_lognormal(rng,
                                                  self.cfg.porta_john_service_mean_min,
                                                  self.cfg.porta_john_service_std_min))


# ---------------------------------------------------------------------------
# 7. Medical dispatch (severity-aware)
# ---------------------------------------------------------------------------
def dispatch_medical(env, bib, mile, bikes, ambulances, med_cfg, metrics, rng):
    """Returns True if runner is removed from race (high-severity transport)."""
    severity = sample_severity(rng, med_cfg.severity_prob)
    t_call = env.now
    metrics.injury_events.append((t_call, mile, severity))

    # --- bike medic first responder ---
    bike_travel = (nearest_dist(mile, med_cfg.bike_locations)
                   / med_cfg.bike_speed_mph) * 60.0
    with bikes.request() as req:
        yield req
        yield env.timeout(bike_travel)
        bike_response = env.now - t_call

        if severity == "minor":
            treat = _lognormal(rng, *med_cfg.treat_min_params)
            yield env.timeout(treat)
            metrics.injuries.append({
                "bib": bib, "mile": mile, "severity": severity,
                "response_min": bike_response, "treat_min": treat,
                "total_off_course_min": bike_response + treat,
                "removed": False,
            })
            return False

        if severity == "medium":
            treat = _lognormal(rng, *med_cfg.treat_med_params)
            yield env.timeout(treat)
            metrics.injuries.append({
                "bib": bib, "mile": mile, "severity": severity,
                "response_min": bike_response, "treat_min": treat,
                "total_off_course_min": bike_response + treat,
                "removed": False,
            })
            return False

        # severity == "high": bike stabilizes; ambulance dispatched in parallel
        stabilize = _lognormal(rng, *med_cfg.stabilize_params)
        yield env.timeout(stabilize)

    # --- ambulance transport ---
    amb_travel = (nearest_dist(mile, med_cfg.ambulance_locations)
                  / med_cfg.ambulance_speed_mph) * 60.0
    with ambulances.request() as req:
        yield req
        yield env.timeout(amb_travel)
        amb_response_from_call = env.now - t_call
        transport = _lognormal(rng, *med_cfg.transport_params)
        yield env.timeout(transport)

    metrics.injuries.append({
        "bib": bib, "mile": mile, "severity": "high",
        "response_min": bike_response,
        "amb_response_min": amb_response_from_call,
        "treat_min": stabilize + transport,
        "total_off_course_min": amb_response_from_call + transport,
        "removed": True,
    })
    return True


# ---------------------------------------------------------------------------
# 8. Runner process
# ---------------------------------------------------------------------------
def runner_proc(env, bib, corral, flat_pace_factor, aid_stations,
                bikes, ambulances, med_cfg, metrics, rng):
    yield env.timeout(CORRAL_START_MIN[corral])

    finish_target = sample_finish_time(rng)
    flat_pace = finish_target / flat_pace_factor   # min per flat mile

    last_mile = 0.0
    for station in aid_stations:
        # travel leg, grade-adjusted
        yield env.timeout(leg_time(last_mile, station.cfg.mile, flat_pace))

        # per-leg injury roll, scaled by leg length
        leg_len = station.cfg.mile - last_mile
        if rng.random() < INJURY_RATE_PER_RUNNER * (leg_len / COURSE_MILES):
            removed = yield env.process(dispatch_medical(
                env, bib, last_mile + rng.random() * leg_len,
                bikes, ambulances, med_cfg, metrics, rng))
            if removed:
                return

        yield env.process(station.visit(bib, rng))
        last_mile = station.cfg.mile

    # final leg + final injury roll
    leg_len = COURSE_MILES - last_mile
    if rng.random() < INJURY_RATE_PER_RUNNER * (leg_len / COURSE_MILES):
        removed = yield env.process(dispatch_medical(
            env, bib, last_mile + rng.random() * leg_len,
            bikes, ambulances, med_cfg, metrics, rng))
        if removed:
            return
    yield env.timeout(leg_time(last_mile, COURSE_MILES, flat_pace))
    metrics.finishes.append((bib, env.now))


# ---------------------------------------------------------------------------
# 9. Top-level driver
# ---------------------------------------------------------------------------
def run_marathon(seed: int = 42, n_runners: int | None = None,
                 station_cfgs: list | None = None,
                 med_cfg: MedicalConfig | None = None,
                 sim_horizon_min: float = CUTOFF_MIN + 30) -> RaceMetrics:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    metrics = RaceMetrics()
    med_cfg = med_cfg or MedicalConfig()
    if station_cfgs is None:
        station_cfgs = [AidStationConfig(mile=m) for m in AID_STATION_MILES]

    aid_stations = [AidStation(env, c, metrics) for c in station_cfgs]
    bikes      = simpy.Resource(env, capacity=med_cfg.num_bike_medics)
    ambulances = simpy.Resource(env, capacity=med_cfg.num_ambulances)

    bib = 1
    for corral, count in FIELD_BY_CORRAL.items():
        if n_runners is not None and bib > n_runners: break
        for _ in range(count):
            if n_runners is not None and bib > n_runners: break
            env.process(runner_proc(env, bib, corral, COURSE_FLAT_EQUIVALENT_MILES,
                                    aid_stations, bikes, ambulances,
                                    med_cfg, metrics, rng))
            bib += 1
    env.run(until=sim_horizon_min)
    return metrics


def summarize(metrics: RaceMetrics) -> None:
    ft = np.array([t for _, t in metrics.finishes])
    print("=" * 70)
    print("Big Sur Marathon -- single replication summary")
    print("=" * 70)
    print(f"Finishers       : {len(ft):,}")
    if len(ft):
        print(f"Finish time mean: {ft.mean():.1f} min ({ft.mean()/60:.2f} hr)   "
              f"p10/p90: {np.percentile(ft,10):.1f} / {np.percentile(ft,90):.1f} min")

    print(f"\nInjuries: {len(metrics.injuries)} total")
    if metrics.injuries:
        by_sev = Counter(i["severity"] for i in metrics.injuries)
        for sev in ("minor", "medium", "high"):
            sub = [i for i in metrics.injuries if i["severity"] == sev]
            if not sub: continue
            r = np.array([i["response_min"] for i in sub])
            print(f"   {sev:6s} n={by_sev[sev]:4d}   "
                  f"first-responder resp mean={r.mean():.2f} min   "
                  f"p90={np.percentile(r,90):.2f}   max={r.max():.2f}")
        amb = [i for i in metrics.injuries if i["severity"] == "high"]
        if amb:
            ar = np.array([i["amb_response_min"] for i in amb])
            print(f"   high-severity ambulance from call: "
                  f"mean={ar.mean():.2f}   p90={np.percentile(ar,90):.2f}   max={ar.max():.2f}")
        print(f"   Removed from race: {sum(1 for i in metrics.injuries if i['removed'])}")

    if metrics.aid_wait_min:
        aw = np.array(metrics.aid_wait_min)
        print(f"\nAid wait        : mean {aw.mean():.2f} min   "
              f"p95 {np.percentile(aw,95):.2f}   max {aw.max():.2f}")
    if metrics.porta_wait_min:
        pw = np.array(metrics.porta_wait_min)
        print(f"Porta-john wait : mean {pw.mean():.2f} min   "
              f"p95 {np.percentile(pw,95):.2f}   max {pw.max():.2f}")
    print(f"Stockouts       : {len(metrics.stockouts)}")
    if metrics.stockouts:
        for (m, it), n in sorted(Counter((m, it) for m, it, _ in metrics.stockouts).items()):
            print(f"   mile {m:>4.1f}  {it:11s}  {n} events")
    print(f"Trash overflows : {len(metrics.trash_overflows)}")
    if metrics.trash_overflows:
        for m, n in sorted(Counter(m for m, _ in metrics.trash_overflows).items()):
            print(f"   mile {m:>4.1f}  {n} overflow events")

    # ---- Pre-race ----
    if metrics.bus_wait_min:
        print(f"\n--- Pre-race ---")
        print(f"Bus stop wait (by location):")
        by_stop = {}
        for name, w in metrics.bus_wait_min:
            by_stop.setdefault(name, []).append(w)
        for name, ws in by_stop.items():
            ws = np.array(ws)
            print(f"   {name:12s} n={len(ws):4d}  "
                  f"mean {ws.mean():.2f} min   "
                  f"p95 {np.percentile(ws,95):.2f}   max {ws.max():.2f}")
        if metrics.stranded_at_busstop:
            sc = Counter(s for _, s in metrics.stranded_at_busstop)
            print(f"   STRANDED runners (no bus before window closed): "
                  f"{sum(sc.values())} total: {dict(sc)}")
        if metrics.start_porta_wait_min:
            sw = np.array(metrics.start_porta_wait_min)
            print(f"Start porta-john wait : mean {sw.mean():.2f} min   "
                  f"p95 {np.percentile(sw,95):.2f}   max {sw.max():.2f}   n={len(sw)}")
        if metrics.late_to_corral:
            lm = np.array([m for _, m in metrics.late_to_corral])
            print(f"Late to corral        : {len(lm)} runners   "
                  f"mean {lm.mean():.2f} min late   max {lm.max():.2f}")


# ---------------------------------------------------------------------------
# 10. Optimization helpers
# ---------------------------------------------------------------------------
def server_sweep(server_counts, reps_per_count=3, seed_base=42,
                 n_runners: int | None = None) -> pd.DataFrame:
    """Sweep aid-station server counts; return one row per (server_count, rep)."""
    rows = []
    for ns in server_counts:
        for rep in range(reps_per_count):
            stations = [AidStationConfig(mile=m, servers=ns) for m in AID_STATION_MILES]
            metrics = run_marathon(seed=seed_base+rep, n_runners=n_runners,
                                   station_cfgs=stations)
            aw = np.array(metrics.aid_wait_min) if metrics.aid_wait_min else np.array([0.0])
            rows.append({
                "servers": ns, "rep": rep,
                "mean_aid_wait": float(aw.mean()),
                "p95_aid_wait":  float(np.percentile(aw, 95)),
                "max_aid_wait":  float(aw.max()),
                "n_finishers":   len(metrics.finishes),
            })
    return pd.DataFrame(rows)


def porta_john_sweep(pj_counts, reps_per_count=3, seed_base=42,
                     n_runners: int | None = None) -> pd.DataFrame:
    rows = []
    for npj in pj_counts:
        for rep in range(reps_per_count):
            stations = [AidStationConfig(mile=m, porta_johns=npj) for m in AID_STATION_MILES]
            metrics = run_marathon(seed=seed_base+rep, n_runners=n_runners,
                                   station_cfgs=stations)
            pw = np.array(metrics.porta_wait_min) if metrics.porta_wait_min else np.array([0.0])
            rows.append({
                "porta_johns": npj, "rep": rep,
                "mean_pj_wait": float(pw.mean()),
                "p95_pj_wait":  float(np.percentile(pw, 95)),
                "max_pj_wait":  float(pw.max()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass  # actual driver moved to end of file


# ---------------------------------------------------------------------------
# 11. Pre-race subsystem: bus pickup + start area
# ---------------------------------------------------------------------------
# Time origin = race start (t=0 minutes). Pre-race events happen at t<0.
# Big Sur reference: race at 6:45 AM; buses load 3:30-4:30 AM (~ -195 to -135).

@dataclass
class BusStopConfig:
    name: str
    n_runners:            int
    arrival_start_min:    float   # earliest runner arrival (mins before race)
    arrival_end_min:      float   # latest runner arrival (mins before race)
    travel_time_mean_min: float
    travel_time_std_min:  float
    bus_dispatch_times:   tuple   # mins-before-race at which buses depart (negative values)
    bus_capacity:         int = 50

DEFAULT_BUS_STOPS = (
    # All times in minutes-from-race-start; race start = 0 at 6:45 AM
    # Capacities and pickup windows from BSIM published transportation schedule.
    #
    # Bus dispatch windows extend ~10 min past the pickup window end to absorb
    # the tail of the queue. Cadence of 2 min per dispatch sized for capacity:
    #   15-min pickup, ~400 ppl: ~12 dispatches * 50 = 600 cap (50% buffer)
    #   30-min pickup, ~800 ppl: ~20 dispatches * 50 = 1000 cap (25% buffer)
    BusStopConfig("A_Marriott",       n_runners=400,
                  arrival_start_min=-195, arrival_end_min=-180,       # 3:45-4:00
                  travel_time_mean_min=52, travel_time_std_min=7,
                  bus_dispatch_times=tuple(range(-195, -170, 2))),    # 13 buses = 650 cap
    BusStopConfig("B_DowntownGarage", n_runners=800,
                  arrival_start_min=-210, arrival_end_min=-180,       # 3:30-4:00
                  travel_time_mean_min=50, travel_time_std_min=7,
                  bus_dispatch_times=tuple(range(-210, -170, 2))),    # 20 buses = 1000 cap
    BusStopConfig("C_CarmelMS",       n_runners=800,
                  arrival_start_min=-210, arrival_end_min=-180,       # 3:30-4:00
                  travel_time_mean_min=40, travel_time_std_min=6,     # ~1 mi from finish
                  bus_dispatch_times=tuple(range(-210, -170, 2))),    # 20 buses
    BusStopConfig("D_DelMonte",       n_runners=800,
                  arrival_start_min=-210, arrival_end_min=-180,       # 3:30-4:00
                  travel_time_mean_min=52, travel_time_std_min=7,
                  bus_dispatch_times=tuple(range(-210, -170, 2))),    # 20 buses
    BusStopConfig("E_MPC",            n_runners=400,
                  arrival_start_min=-195, arrival_end_min=-180,       # 3:45-4:00
                  travel_time_mean_min=50, travel_time_std_min=7,
                  bus_dispatch_times=tuple(range(-195, -170, 2))),    # 13 buses
    BusStopConfig("F_Embassy",        n_runners=400,
                  arrival_start_min=-195, arrival_end_min=-180,       # 3:45-4:00
                  travel_time_mean_min=54, travel_time_std_min=8,     # Seaside
                  bus_dispatch_times=tuple(range(-195, -170, 2))),    # 13 buses
    BusStopConfig("G_CarmelPlaza",    n_runners=400,
                  arrival_start_min=-195, arrival_end_min=-180,       # 3:45-4:00
                  travel_time_mean_min=42, travel_time_std_min=6,     # near finish
                  bus_dispatch_times=tuple(range(-195, -170, 2))),    # 13 buses
)
# Totals: 4 * 400 + 3 * 800 = 4000 base runners; proportional scaling applied if
# n_runners differs (see run_marathon_full).


@dataclass
class StartAreaConfig:
    porta_johns:            int   = 120
    p_use_before_race:      float = 0.85
    use_time_mean_min:      float = 1.5
    use_time_std_min:       float = 0.6
    use_window_start_min:   float = -60    # earliest a runner targets the porta-john
    use_window_end_min:     float = -5     # latest


class BusStop:
    """Runners join the queue; buses arrive on schedule and pick up up to capacity.
    Wait time = (time bus arrives) - (time runner joined queue)."""
    def __init__(self, env, cfg, metrics):
        self.env = env
        self.cfg = cfg
        self.metrics = metrics
        self.queue: list = []          # list of (event, t_arrive, bib)
        env.process(self._dispatch_loop())

    def _dispatch_loop(self):
        for t_dispatch in self.cfg.bus_dispatch_times:
            wait = t_dispatch - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            n_load = min(self.cfg.bus_capacity, len(self.queue))
            for _ in range(n_load):
                evt, t_arr, bib = self.queue.pop(0)
                self.metrics.bus_wait_min.append((self.cfg.name, self.env.now - t_arr))
                evt.succeed()
        # No more buses scheduled; anyone left in queue is stranded.
        for evt, t_arr, bib in self.queue:
            self.metrics.stranded_at_busstop.append((bib, self.cfg.name))

    def board_and_travel(self, bib, rng):
        evt = self.env.event()
        self.queue.append((evt, self.env.now, bib))
        yield evt
        travel = max(5.0, rng.normal(self.cfg.travel_time_mean_min,
                                     self.cfg.travel_time_std_min))
        self.metrics.bus_travel_min.append((self.cfg.name, travel))
        yield self.env.timeout(travel)
        name = self.cfg.name
        if name not in self.metrics.first_bus_at_start:
            self.metrics.first_bus_at_start[name] = self.env.now
        self.metrics.last_bus_at_start[name] = self.env.now


class StartArea:
    def __init__(self, env, cfg, metrics):
        self.env = env
        self.cfg = cfg
        self.metrics = metrics
        self.porta_johns = simpy.Resource(env, capacity=cfg.porta_johns)

    def maybe_use(self, bib, rng):
        if rng.random() > self.cfg.p_use_before_race:
            return
        t_target = rng.uniform(self.cfg.use_window_start_min,
                               self.cfg.use_window_end_min)
        wait = t_target - self.env.now
        if wait > 0:
            yield self.env.timeout(wait)
        t_q = self.env.now
        with self.porta_johns.request() as req:
            yield req
            self.metrics.start_porta_wait_min.append(self.env.now - t_q)
            yield self.env.timeout(_lognormal(rng,
                                              self.cfg.use_time_mean_min,
                                              self.cfg.use_time_std_min))


def _race_segment(env, bib, finish_target, flat_pace_factor, aid_stations,
                  bikes, ambulances, med_cfg, metrics, rng):
    """The race itself (assumes runner has already started -- env.now >= corral time).
    finish_target: pre-sampled finish time in minutes."""
    flat_pace = finish_target / flat_pace_factor

    last_mile = 0.0
    for station in aid_stations:
        yield env.timeout(leg_time(last_mile, station.cfg.mile, flat_pace))
        leg_len = station.cfg.mile - last_mile
        if rng.random() < INJURY_RATE_PER_RUNNER * (leg_len / COURSE_MILES):
            removed = yield env.process(dispatch_medical(
                env, bib, last_mile + rng.random() * leg_len,
                bikes, ambulances, med_cfg, metrics, rng))
            if removed:
                return
        yield env.process(station.visit(bib, rng))
        last_mile = station.cfg.mile

    leg_len = COURSE_MILES - last_mile
    if rng.random() < INJURY_RATE_PER_RUNNER * (leg_len / COURSE_MILES):
        removed = yield env.process(dispatch_medical(
            env, bib, last_mile + rng.random() * leg_len,
            bikes, ambulances, med_cfg, metrics, rng))
        if removed:
            return
    yield env.timeout(leg_time(last_mile, COURSE_MILES, flat_pace))
    metrics.finishes.append((bib, env.now))


def runner_lifecycle(env, bib, bus_stop, corral_name, corral_start_offset,
                     finish_time, flat_pace_factor, aid_stations, bikes,
                     ambulances, start_area, med_cfg, metrics, rng):
    """Full journey: arrive at bus stop -> bus -> start area -> corral -> race.
    finish_time: this runner's pre-sampled finish time (used inside _race_segment).
    corral_start_offset: minutes after race start when this runner's corral begins.
    """
    # 1. Arrive at bus stop at random time within window
    t_arr = rng.uniform(bus_stop.cfg.arrival_start_min,
                        bus_stop.cfg.arrival_end_min)
    wait = t_arr - env.now
    if wait > 0:
        yield env.timeout(wait)

    # 2. Bus stop queue, board, travel to start
    yield env.process(bus_stop.board_and_travel(bib, rng))

    # 3. Start area porta-john (probabilistic, with target use time)
    yield env.process(start_area.maybe_use(bib, rng))

    # 4. Wait for corral start
    wait = corral_start_offset - env.now
    if wait < 0:
        metrics.late_to_corral.append((bib, -wait))
        # Runner still starts (corral let them go), but is "late"
    else:
        yield env.timeout(wait)

    # 5. Race
    yield env.process(_race_segment(env, bib, finish_time, flat_pace_factor,
                                    aid_stations, bikes, ambulances,
                                    med_cfg, metrics, rng))


# ---------------------------------------------------------------------------
# 12. Queue monitoring for time-series visualization
# ---------------------------------------------------------------------------
@dataclass
class QueueMonitor:
    interval_min: float = 1.0
    times: list = field(default_factory=list)
    series: dict = field(default_factory=dict)     # name -> [queue length over time]

def monitor_process(env, monitor, resources_by_name):
    """Sample len(resource.queue) every `interval_min` minutes."""
    while True:
        monitor.times.append(env.now)
        for name, res in resources_by_name.items():
            monitor.series.setdefault(name, []).append(len(res.queue))
        yield env.timeout(monitor.interval_min)


# ---------------------------------------------------------------------------
# 13. Full driver: pre-race + race + monitoring
# ---------------------------------------------------------------------------
def run_marathon_full(seed: int = 42,
                      n_runners: int | None = None,
                      station_cfgs: list | None = None,
                      med_cfg: MedicalConfig | None = None,
                      bus_stop_cfgs: tuple = DEFAULT_BUS_STOPS,
                      start_area_cfg: StartAreaConfig | None = None,
                      corral_plan: CorralPlan | None = None,
                      monitor_interval_min: float = 1.0,
                      sim_horizon_min: float = CUTOFF_MIN + 30):
    """Full lifecycle simulation; returns (metrics, monitor).
    corral_plan: how to split the field into corrals (defaults to baseline_3x5)."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment(initial_time=-220.0)
    metrics = RaceMetrics()
    med_cfg = med_cfg or MedicalConfig()
    start_area_cfg = start_area_cfg or StartAreaConfig()
    corral_plan = corral_plan or CorralPlan.baseline_3x5()

    if station_cfgs is None:
        station_cfgs = [AidStationConfig(mile=m) for m in AID_STATION_MILES]
    aid_stations = [AidStation(env, c, metrics) for c in station_cfgs]
    bikes        = simpy.Resource(env, capacity=med_cfg.num_bike_medics)
    ambulances   = simpy.Resource(env, capacity=med_cfg.num_ambulances)
    bus_stops    = [BusStop(env, c, metrics) for c in bus_stop_cfgs]
    start_area   = StartArea(env, start_area_cfg, metrics)

    # Monitor key queues
    monitor = QueueMonitor(interval_min=monitor_interval_min)
    monitored = {}
    for stop in bus_stops:
        monitored[f"bus:{stop.cfg.name}"] = type("Q", (), {"queue": stop.queue})()
    monitored["start_porta"] = start_area.porta_johns
    monitored["bike_medics"] = bikes
    monitored["ambulances"]  = ambulances
    for s in aid_stations:
        monitored[f"aid_serv@{s.cfg.mile}"]  = s.servers
        monitored[f"aid_porta@{s.cfg.mile}"] = s.porta_johns
    env.process(monitor_process(env, monitor, monitored))

    # Decide field size
    total_cap = sum(s.cfg.n_runners for s in bus_stops)
    target_n  = n_runners if n_runners is not None else total_cap

    # Pre-sample finish times for the whole field, then assign corrals by rank
    finish_times  = np.array([sample_finish_time(rng) for _ in range(target_n)])
    corral_idx    = assign_corrals(finish_times, corral_plan)

    # Proportionally assign bus stops
    assignments = []
    for stop in bus_stops:
        share = round(target_n * stop.cfg.n_runners / total_cap)
        assignments += [stop] * share
    rng.shuffle(assignments)
    assignments = assignments[:target_n]
    while len(assignments) < target_n:
        assignments.append(bus_stops[0])

    # Spawn one process per runner
    for bib_idx, stop in enumerate(assignments):
        ci   = int(corral_idx[bib_idx])
        name = corral_plan.names[ci]
        off  = corral_plan.start_offsets_min[ci]
        ft   = float(finish_times[bib_idx])
        env.process(runner_lifecycle(env, bib_idx + 1, stop, name, off, ft,
                                     COURSE_FLAT_EQUIVALENT_MILES,
                                     aid_stations, bikes, ambulances,
                                     start_area, med_cfg, metrics, rng))

    env.run(until=sim_horizon_min)
    return metrics, monitor


# ---------------------------------------------------------------------------
# 14. Recommended configurations
# ---------------------------------------------------------------------------
def make_recommended_aid_configs(n_runners: int = 4500) -> list:
    """Per-station server counts that reflect pack-density physics:
    early stations get the bunched corral pack and need more hands;
    late stations see a spread-out field and need fewer."""
    configs = []
    for m in AID_STATION_MILES:
        if   m < 8:   servers = 16     # bunched pack, high arrival burst
        elif m < 18:  servers = 10     # pack spreading
        else:         servers = 6      # well-spread
        # Inventory scales with field size
        configs.append(AidStationConfig(
            mile=m,
            servers=servers,
            porta_johns=4,
            trash_capacity=int(1.6 * n_runners),       # ~one capacity per runner * 1.5x buffer
            water_cups=int(1.5 * n_runners),
            electrolyte_cups=int(1.0 * n_runners),
            snacks=int(1.2 * n_runners),
        ))
    return configs


# ---------------------------------------------------------------------------
# 15. Scenario analysis -- evaluate against targets, generate recommendations
# ---------------------------------------------------------------------------
@dataclass
class Target:
    name:        str
    target:      float   # value to be UNDER
    failure:     float   # value above this counts as a failure
    units:       str = ""

DEFAULT_TARGETS = {
    "aid_wait_p95_sec":           Target("Aid wait p95 (sec)",            10,  30, "sec"),
    "aid_wait_max_sec":           Target("Aid wait max (sec)",            10,  60, "sec"),
    "porta_wait_p95_min":         Target("Aid porta-john wait p95 (min)", 5,   15, "min"),
    "med_resp_p90_minor_min":     Target("Med response p90, minor (min)", 5,   10, "min"),
    "med_resp_p90_high_min":      Target("Med response p90, high (min)",  3,    6, "min"),
    "amb_resp_p90_min":           Target("Ambulance p90 from call (min)", 15,  25, "min"),
    "bus_wait_p95_min":           Target("Bus stop wait p95 (min)",       15,  30, "min"),
    "start_porta_p95_min":        Target("Start porta-john wait p95 (min)", 5, 15, "min"),
    "stranded_count":             Target("Runners stranded at bus stop",   0,   0,  ""),
    "late_to_corral_count":       Target("Runners late to corral",         0,  50,  ""),
    "stockout_count":             Target("Supply stockout events",         0,   0,  ""),
    "trash_overflow_count":       Target("Trash overflow events",          0,   5,  ""),
}


def _extract_metrics(metrics: RaceMetrics) -> dict:
    """Compute the dict of metric values that targets are compared against."""
    aw = np.array(metrics.aid_wait_min) * 60 if metrics.aid_wait_min else np.array([0.0])
    pw = np.array(metrics.porta_wait_min)    if metrics.porta_wait_min else np.array([0.0])
    bw = np.array([w for _, w in metrics.bus_wait_min]) if metrics.bus_wait_min else np.array([0.0])
    sw = np.array(metrics.start_porta_wait_min) if metrics.start_porta_wait_min else np.array([0.0])

    minor = [i["response_min"] for i in metrics.injuries if i["severity"] == "minor"]
    high  = [i["response_min"] for i in metrics.injuries if i["severity"] == "high"]
    amb_r = [i.get("amb_response_min") for i in metrics.injuries if i.get("amb_response_min") is not None]

    return {
        "aid_wait_p95_sec":      float(np.percentile(aw, 95)),
        "aid_wait_max_sec":      float(aw.max()),
        "porta_wait_p95_min":    float(np.percentile(pw, 95)),
        "med_resp_p90_minor_min":float(np.percentile(minor, 90)) if minor else 0.0,
        "med_resp_p90_high_min": float(np.percentile(high,  90)) if high  else 0.0,
        "amb_resp_p90_min":      float(np.percentile(amb_r, 90)) if amb_r else 0.0,
        "bus_wait_p95_min":      float(np.percentile(bw, 95)),
        "start_porta_p95_min":   float(np.percentile(sw, 95)),
        "stranded_count":        float(len(metrics.stranded_at_busstop)),
        "late_to_corral_count":  float(len(metrics.late_to_corral)),
        "stockout_count":        float(len(metrics.stockouts)),
        "trash_overflow_count":  float(len(metrics.trash_overflows)),
    }


def evaluate_targets(metrics: RaceMetrics, targets: dict = None) -> list:
    """Compare metrics to targets; return a list of dicts ready to print/save."""
    targets = targets or DEFAULT_TARGETS
    vals = _extract_metrics(metrics)
    rows = []
    for key, tgt in targets.items():
        v = vals[key]
        if   v <= tgt.target:  status = "PASS"
        elif v <= tgt.failure: status = "WARN"
        else:                   status = "FAIL"
        rows.append({
            "metric":  tgt.name,
            "value":   v,
            "target":  tgt.target,
            "failure": tgt.failure,
            "status":  status,
            "units":   tgt.units,
        })
    return rows


def print_scenario_report(rows, header: str = "Scenario report"):
    print(f"\n{'='*82}")
    print(f"{header:^82}")
    print(f"{'='*82}")
    print(f"{'Metric':<40}{'Value':>10}{'Target':>10}{'Fail':>10}{'Status':>10}")
    print("-" * 82)
    for r in rows:
        v = f"{r['value']:.2f}" if r['value'] < 1000 else f"{r['value']:.0f}"
        print(f"{r['metric']:<40}{v:>10}{r['target']:>10.1f}{r['failure']:>10.1f}{r['status']:>10}")
    n_fail = sum(1 for r in rows if r['status'] == 'FAIL')
    n_warn = sum(1 for r in rows if r['status'] == 'WARN')
    n_pass = sum(1 for r in rows if r['status'] == 'PASS')
    print("-" * 82)
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")


# ---------------------------------------------------------------------------
# 16. Monte Carlo wrapper + sweep functions
# ---------------------------------------------------------------------------
def run_monte_carlo(n_reps: int = 30,
                    n_runners: int = 2500,
                    base_seed: int = 0,
                    **kwargs) -> pd.DataFrame:
    """Run N independent replications; return DataFrame with one row per rep.
    All run_marathon_full kwargs (station_cfgs, med_cfg, corral_plan, ...) pass through."""
    rows = []
    for rep in range(n_reps):
        m, _ = run_marathon_full(seed=base_seed + rep, n_runners=n_runners, **kwargs)
        d = _extract_metrics(m)
        d["rep"] = rep
        d["n_finishers"] = len(m.finishes)
        rows.append(d)
    return pd.DataFrame(rows)


def monte_carlo_summary(df: pd.DataFrame, targets: dict | None = None) -> pd.DataFrame:
    """Summarize a Monte Carlo DataFrame: mean/std/quantiles/pass-rate per metric."""
    targets = targets or DEFAULT_TARGETS
    skip = {"rep", "n_finishers"}
    rows = []
    for col in df.columns:
        if col in skip:
            continue
        vals = df[col].values
        tgt = targets.get(col)
        row = {
            "metric":     tgt.name if tgt else col,
            "mean":       float(vals.mean()),
            "std":        float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "p5":         float(np.percentile(vals, 5)),
            "p50":        float(np.percentile(vals, 50)),
            "p95":        float(np.percentile(vals, 95)),
        }
        if tgt is not None:
            row["target"]       = tgt.target
            row["failure"]      = tgt.failure
            row["pass_rate"]    = float((vals <= tgt.target).mean())
            row["fail_rate"]    = float((vals >  tgt.failure).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bike_placement_sweep(placements: dict,
                         n_reps: int = 15,
                         n_runners: int = 2000,
                         **kwargs) -> pd.DataFrame:
    """Compare bike medic placements; n_bikes = number of locations supplied.
    placements: dict name -> tuple of mile locations.
    Returns DataFrame: one row per (placement, rep)."""
    rows = []
    for name, locs in placements.items():
        med_cfg = MedicalConfig(bike_locations=tuple(locs),
                                num_bike_medics=len(locs))
        for rep in range(n_reps):
            m, _ = run_marathon_full(seed=42 + rep, n_runners=n_runners,
                                     med_cfg=med_cfg, **kwargs)
            high = [i["response_min"] for i in m.injuries if i["severity"] == "high"]
            med_ = [i["response_min"] for i in m.injuries if i["severity"] == "medium"]
            mnor = [i["response_min"] for i in m.injuries if i["severity"] == "minor"]
            all_ = [i["response_min"] for i in m.injuries]
            rows.append({
                "placement":     name,
                "n_bikes":       len(locs),
                "rep":           rep,
                "p90_all":       float(np.percentile(all_, 90)) if all_ else 0.0,
                "p90_high":      float(np.percentile(high, 90)) if high else 0.0,
                "p90_medium":    float(np.percentile(med_, 90)) if med_ else 0.0,
                "p90_minor":     float(np.percentile(mnor, 90)) if mnor else 0.0,
                "max_response":  float(max(all_))               if all_ else 0.0,
                "n_injuries":    len(all_),
            })
    return pd.DataFrame(rows)


def corral_sensitivity_sweep(plans: dict,
                             n_reps: int = 15,
                             n_runners: int = 2000,
                             **kwargs) -> pd.DataFrame:
    """Compare different CorralPlans. plans: dict name -> CorralPlan.
    Returns DataFrame: one row per (plan, rep) with extracted metrics + plan info."""
    rows = []
    for plan_name, plan in plans.items():
        for rep in range(n_reps):
            m, _ = run_marathon_full(seed=42 + rep, n_runners=n_runners,
                                     corral_plan=plan, **kwargs)
            d = _extract_metrics(m)
            d["plan"]        = plan_name
            d["n_corrals"]   = len(plan.names)
            d["max_offset"]  = max(plan.start_offsets_min)
            d["rep"]         = rep
            d["n_finishers"] = len(m.finishes)
            rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"Course profile loaded: {len(ELEVATION_KNOTS_FT)} knots, "
          f"flat-equivalent = {COURSE_FLAT_EQUIVALENT_MILES:.2f} miles "
          f"(vs 26.2 nominal)\n")
    metrics, monitor = run_marathon_full(seed=42, n_runners=4500)
    summarize(metrics)
