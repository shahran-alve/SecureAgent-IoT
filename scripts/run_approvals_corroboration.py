"""Experiments E30-E32: analyst approvals, filter false positives, manufactured corroboration.
Usage: python scripts/run_approvals_corroboration.py [--quick] [--only e30,e31,e32]"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secureagent_iot.config import load_config, set_global_seed
from secureagent_iot import extended4 as R

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", default="e30,e31,e32")
ap.add_argument("--config", default=None); a = ap.parse_args()
cfg = load_config(a.config); seeds = list(cfg["seeds"]); set_global_seed(seeds[0])
if a.quick: seeds = seeds[:1]
T = Path(__file__).resolve().parents[1] / "results" / "tables"; T.mkdir(parents=True, exist_ok=True)
only = set(a.only.split(","))
def save(df, name):
    df.to_csv(T / name, index=False); print(f"wrote {name} ({len(df)} rows)", flush=True)
t0 = time.time()
if "e30" in only:
    save(R.approvals_benchmarks(cfg, seeds), "approvals_benchmarks.csv")
    save(R.approvals_campaign(cfg, seeds), "approvals_campaign.csv")
if "e31" in only:
    save(R.filter_fpr_sweep(cfg, seeds), "filter_fpr.csv")
if "e32" in only:
    save(R.manufactured_corroboration(cfg, seeds), "manufactured_corroboration.csv")
    save(R.fleet_crowding(cfg, seeds), "fleet_crowding.csv")
print(f"done in {time.time() - t0:.0f} s")
