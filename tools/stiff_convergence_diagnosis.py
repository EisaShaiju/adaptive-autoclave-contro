"""
stiff_convergence_diagnosis.py
Diagnoses WHY stiff-window steps fail to formally converge, rather than
treating "converged=False" as one undifferentiated outcome. Uses only data
already recorded by tools/final_controlled_comparison.py's per_step_metrics.csv
-- no new simulation. The KKT certificate is a CONJUNCTION (see
docs/PHASE4_VALIDATION_REPORT.md section 15.1: converged = kkt(...) AND
obj_plateau(...)), so a step can fail
the overall flag while its KKT half already passes comfortably; this script
separates that case (solution is numerically exact, only the plateau half
blocks certification) from the case where the KKT half genuinely fails too
(solution is actually far from optimal).

Classification is PRIMARILY by the KKT half, computed directly from already-
logged fields (kkt_residual <= kkt_tolerance) -- an objective fact about the
run, not a threshold chosen to flatter the result. The independently-computed
objective_gap_snn_vs_reference column (from a reference OSQP solve on the
SNN's own scaled arrays, feasibility-gated, see final_controlled_comparison.py
reference_objective_gap()) is reported as supporting evidence within each
group, with the full distribution shown rather than a single cherry-picked
threshold.

Usage:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/stiff_convergence_diagnosis.py <run_dir> [<run_dir> ...]

Each <run_dir> is a results/final_comparison/<id>_<label>_<timestamp>/
directory containing per_step_metrics.csv and summary.json.
"""
from pathlib import Path
import sys
import csv
import json
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_stiff_window_rows(run_dir: Path):
    """Read nominal_heatup rows and slice to the run's own recorded stiff
    window, exactly as summary.json defines it for that run (peak rho(Ap)_cvx
    +/- margin) -- do not re-derive the window, trust what the run itself
    computed, since re-deriving it here risks a silent off-by-one vs. the
    window actually reported elsewhere for this run."""
    summary = json.load(open(run_dir / "summary.json"))
    label = summary["scenarios"]["stiff_exotherm_window"]["label"]
    # label format: "stiff_exotherm_window (steps {lo}-{hi-1} of nominal_heatup)"
    inside = label.split("steps")[1].split("of")[0].strip()
    lo_str, hi_minus_1_str = inside.split("-")
    lo, hi = int(lo_str), int(hi_minus_1_str) + 1

    rows = list(csv.DictReader(open(run_dir / "per_step_metrics.csv")))
    nominal_rows = [r for r in rows if r["scenario"] == "nominal_heatup"]
    stiff_rows = [r for r in nominal_rows if lo <= int(r["step"]) < hi]
    return stiff_rows, (lo, hi), summary["provenance"]["git_commit"]


def classify(rows):
    nonconv = [r for r in rows if r["snn_verified_converged"] == "False"]
    conv = [r for r in rows if r["snn_verified_converged"] == "True"]

    kkt_pass_group = []   # certificate's KKT half already satisfied, blocked only by plateau half
    kkt_fail_group = []   # KKT half itself not satisfied -- genuinely unresolved within budget
    no_kkt_data = []      # pre-KKT-certificate library (0.4.x) or field absent

    for r in nonconv:
        kres, ktol = r.get("snn_kkt_residual"), r.get("snn_kkt_tolerance")
        if not kres or not ktol or kres in ("", "None") or ktol in ("", "None"):
            no_kkt_data.append(r)
            continue
        kres, ktol = float(kres), float(ktol)
        (kkt_pass_group if kres <= ktol else kkt_fail_group).append(r)

    def gap_stats(group):
        gaps = []
        for r in group:
            g = r.get("objective_gap_snn_vs_reference")
            if g not in (None, "", "None"):
                gaps.append(abs(float(g)))
        if not gaps:
            return {"n_with_gap": 0, "n_total": len(group)}
        gaps.sort()
        return {
            "n_with_gap": len(gaps), "n_total": len(group),
            "min": gaps[0], "median": statistics.median(gaps), "max": gaps[-1],
            "mean": statistics.mean(gaps),
        }

    return {
        "n_stiff_steps": len(rows),
        "n_converged": len(conv),
        "n_nonconverged": len(nonconv),
        "kkt_half_passes_blocked_by_plateau": {
            "steps": [int(r["step"]) for r in kkt_pass_group],
            "n": len(kkt_pass_group),
            "objective_gap": gap_stats(kkt_pass_group),
        },
        "kkt_half_fails_genuinely_unresolved": {
            "steps": [int(r["step"]) for r in kkt_fail_group],
            "n": len(kkt_fail_group),
            "objective_gap": gap_stats(kkt_fail_group),
        },
        "no_kkt_certificate_data": {
            "steps": [int(r["step"]) for r in no_kkt_data],
            "n": len(no_kkt_data),
        },
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: stiff_convergence_diagnosis.py <run_dir> [<run_dir> ...]")
        return 1

    results = {}
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        label = run_dir.name
        rows, window, commit = load_stiff_window_rows(run_dir)
        diag = classify(rows)
        diag["window"] = {"lo": window[0], "hi": window[1]}
        diag["git_commit"] = commit
        results[label] = diag

        print(f"\n=== {label} (commit {commit[:8]}, window [{window[0]},{window[1]})) ===")
        print(f"  {diag['n_stiff_steps']} stiff steps: "
              f"{diag['n_converged']} formally converged, "
              f"{diag['n_nonconverged']} not.")
        a = diag["kkt_half_passes_blocked_by_plateau"]
        b = diag["kkt_half_fails_genuinely_unresolved"]
        print(f"  Of the {diag['n_nonconverged']} non-converged:")
        print(f"    {a['n']} pass the KKT half already (blocked only by the "
              f"objective-plateau half) -- steps {a['steps']}")
        if a["objective_gap"]["n_with_gap"]:
            g = a["objective_gap"]["n_with_gap"]
            gs = a["objective_gap"]
            print(f"      objective gap over these {g}: "
                  f"min={gs['min']:.2e} median={gs['median']:.2e} max={gs['max']:.2e}")
        print(f"    {b['n']} fail the KKT half itself (genuinely unresolved "
              f"within budget) -- steps {b['steps']}")
        if b["objective_gap"]["n_with_gap"]:
            g = b["objective_gap"]["n_with_gap"]
            gs = b["objective_gap"]
            print(f"      objective gap over these {g}: "
                  f"min={gs['min']:.2e} median={gs['median']:.2e} max={gs['max']:.2e}")
        n0 = diag["no_kkt_certificate_data"]["n"]
        if n0:
            print(f"    {n0} have no KKT certificate data recorded (library "
                  f"predates the certificate) -- steps {diag['no_kkt_certificate_data']['steps']}")

    out_path = PROJECT_ROOT / "results" / "stiff_convergence_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
