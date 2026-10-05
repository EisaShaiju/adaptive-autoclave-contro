FPGA QP export -- generated 2026-10-06T02:38:44
Commit: 2d28d20449ff81bc94dcd9abb913d5a1f384f043
Working tree dirty: False

Files: 2d28d204_nominal_heatup.npz, 2d28d204_disturbance_step60.npz

Each .npz, per scenario, stacks one QP per closed-loop step (160 steps,
recommended config N=10, soft, k0_scale=0.1, trust_region=False, LTI mode --
see module docstring of tools/export_fpga_qp_dump.py for why LTI and not LTV).

Arrays (all stacked along axis 0 = step):
  H_s, g_s, C_s, d_s   -- the CONDITIONED QP exactly as the SNN solver reads it.
                           Solved as: 0.5 z^T H_s z + g_s^T z  s.t.  C_s z + d_s <= 0.
  D_scale              -- the column-scaling diagonal; H_raw = D*H_s*D, etc.
                           (H_s = D^-1 H_raw D^-1, g_s = D^-1 f_raw, C_s = D^-1 A_raw
                           row-normalized; see _condition() in src/snn_mpc_controller.py).
  H_raw, f_raw         -- the raw canonical QP (pre-conditioning), for cross-check.
  A_box_lo, A_box_hi   -- actuator box bounds per step, length n_control (native
                           per-variable bounds: A_box_lo <= Ta_k <= A_box_hi).
  A_slew, b_slew       -- slew-rate rows (NOT a per-variable bound -- couples
                           Ta_k and Ta_{k-1}): A_slew @ z <= b_slew.
  A_out, b_out         -- output/gradient constraint rows (the only rows mixing
                           control and slack columns): A_out @ z <= b_out.
  slack_lo, slack_hi   -- slack variables' native bound (lo=0, hi=+inf), length
                           n_slack. Free addition beyond the literal ask, since
                           it's a genuine per-variable bound like the actuator
                           box, not a general row.

  IMPORTANT -- two different unit spaces are exported side by side:
  A_box_lo/A_box_hi, A_slew/b_slew, A_out/b_out, slack_lo/slack_hi, H_raw,
  f_raw are all in RAW PHYSICAL units (same space as TA_MIN/TA_MAX below).
  H_s, g_s, C_s, d_s, warm_start_scaled are in the CONDITIONED space the
  solver actually iterates in. D_scale is the per-variable column-scaling
  vector relating them (scaled_z_i = raw_z_i * D_scale_i) -- use it if you
  need the box/slew/output bounds in the conditioned space instead of raw.
  u0_cvx, u0_snn       -- applied first-move solutions, CVXPY and software SNN.
  snn_converged, snn_convergence_reason, snn_iterations,
  snn_kkt_residual, snn_kkt_tolerance, snn_constraint_residual
                       -- software-SNN solve diagnostics per step, for comparing
                          the FPGA kernel's output against the software run at
                          matched accuracy.

Scalars: n_control (=N), n_slack (=m), horizon, commit_hash, scenario,
sign_convention, z_ordering, TA_MIN, TA_MAX, TA_RATE_MAX.

z ordering: z = [Ta_0..Ta_{N-1}, s_0..s_{m-1}]  (n_control=10,
and n_slack is constant across the whole trajectory per the relative-degree-5
structural invariant -- see module docstring for why that's safe to assume).

snn_opt solver config / iteration budget (same field names as
tools/final_controlled_comparison.py's capture_provenance(), so this export
and that script's summary.json describe the same run without a translation
layer):
{'k0': None, 'k0_scale': 0.1, 'projection_method': 'adaptive', 'max_iterations': 8000, 'max_projection_iters': 5000, 'backend': 'c', 'convergence': {'enable_early_stopping': True, 'check_every': 50, 'min_iterations': 100, 'patience': 3, 'obj_rel_tol': 1e-07, 'proj_grad_tol': None, 'feasibility_tol': 0.01, 'optimality_test': 'kkt', 'kkt_abs_tol': 1e-09, 'kkt_rel_tol': 0.0001}}
