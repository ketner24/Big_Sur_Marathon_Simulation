Big Sur Marathon Simulator
Discrete-event simulation of the Big Sur International Marathon, built with SimPy and exposed through a Streamlit UI for interactive exploration.
Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browser opens at http://localhost:8501.
Quick start (Docker)
```bash
docker build -t bigsur-sim .
docker run -p 8501:8501 bigsur-sim
```
Then open http://localhost:8501.
What it does
Three tabs:
Single Scenario -- pick all parameters, run a Monte Carlo of N reps, see the distribution of every metric. Color-coded PASS/WARN/FAIL targets at a glance.
Compare Scenarios -- paired-seed (common random numbers) comparison of two configurations. Uses `scipy.stats.ttest_rel` so claims like "Scenario B reduces porta-john wait significantly" come with a real p-value.
Sensitivity Sweep -- pick one parameter, pick values, see how outputs respond. Use for elbow-finding on resource sizing.
Parameters exposed
Field size (500-4500 runners)
Number of corrals & interval between waves
Per-segment server counts at aid stations (early/mid/late)
Porta-johns per aid station and at start
Water prep rate and initial buffer
Bike medic count & placement strategy
Ambulance count
Bus dispatch frequency
Performance notes
One replication at 1500 runners: ~3 sec
One replication at 4500 runners: ~8 sec
10-rep MC at 1500: ~30 sec (acceptable interactive)
Paired comparison runs 2x per rep
Sensitivity sweep: multiply by len(values) * n_reps
For deployment to a shared server, plan for ~10 sec average response per "Run" click. Consider gating the field-size slider's max if hosting publicly.
Files
`big_sur_simpy.py` -- the simulation model (~1200 lines)
`big_sur_visuals.py` -- static plot helpers (used by analysis scripts)
`app.py` -- the Streamlit UI
`Dockerfile`, `requirements.txt` -- deployment
