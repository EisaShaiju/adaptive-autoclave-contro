Hi, thank you for the detailed feedback, and no problem at all about the
turn-around. I have done the validation pass you asked for, and I want to give
you a proper answer back because a few things turned out differently from what
we both expected.

First, the three points you raised were all correct, and I have fixed them.
The prediction models were indeed different: `trust_region` was hardcoded on
in the SNN controller and off in the CVXPY one, and I measured the gap, it goes
up to about 1851 in max absolute difference of the A matrix through gelation,
so you were right that even a perfect solver could not have matched. It is now
a single explicit parameter that defaults to off on both controllers, so the
identical-model case is the default and any divergence has to be asked for on
purpose. I also found a second problem while doing this that we had not spotted:
the SNN was condensing its prediction with a one-step-shifted state window, so
it was costing and constraining x_1..x_N while CVXPY was costing x_0..x_{N-1}.
Both controllers now go through one shared QP builder, and I verified it
properly, given the same state, the two QPs are bit-identical, zero difference
on every array at every one of 160 steps including through the disturbance and
the exotherm. Your point about the clipping was right too, all the box, slew and
gradient constraints now live inside the QP for both controllers, and the clip
is only a reported safety filter with its activation rate published.

On the second point, the numbers. With the comparison made fair, the three you
asked for are: per-step difference in applied control 0.714 degC RMS, RMS
difference between the two closed-loop trajectories 0.249, and the SNN's max
constraint residual 1.55 with a formal convergence rate of 0 percent. For
comparison, before the fixes those same numbers were 16.005 degC RMS, 7.342, and
a residual of 1.85e5. So the control agreement improved by about 96 percent and
the residual by five orders of magnitude, but the convergence flag did not move
at all, which brings me to the interesting part.

The conditioning diagnosis turned out to be only partly right, and I think this
is the main finding of the pass. I tested it directly: I whitened the Hessian so
its condition number was exactly 1.0, which is as well-conditioned as a problem
can possibly be, and the solver still failed in essentially the same way, the
final feasibility changed by less than 3 percent. So conditioning was not the
dominant cause. When I swept it properly, 24 configurations across trust region,
constraint form, horizon and step size, the actual cause came out clearly: the
per-step QP is infeasible at the stiff steps. OSQP reports `infeasible` on all
12 of the hard-constrained configurations, at every horizon and under both
models. The gradient constraint at the end of the horizon has an offset that
gets amplified by rho(Ap)^(N-1), which is about 4200 times at the gelation peak,
so the constraint boundary sits astronomically far away and the feasible set is
empty. The solver was not failing to solve a stiff problem, it was being asked
to find a point inside a region that does not exist. That also explains why no
rescaling could ever have fixed it, the offending term does not depend on the
decision variable at all, so no change of variables can shrink it. Softening
just the predicted-state rows, with the actuator limits kept hard, makes the
problem feasible and the SNN then returns feasible points for the first time.

On whether it converges, this is where I have a specific answer rather than
just a negative result. The flag still never fires, but I traced exactly why.
The projected-gradient norm sits at about 1.66e10 and does not move even with a
six times larger iteration budget, and `snn_opt` tests that norm as an absolute
quantity against a tolerance of 5e-2. On a problem whose gradient scale is 1e10
that is asking for twelve orders of magnitude of reduction, so the test cannot
fire no matter how good the solver is, it is not scale-invariant. To check
whether the answer itself was right I went to a short horizon where the
amplification is mild and compared against OSQP on the identical problem: the
SNN matches the optimum to 0.0005 degC on the applied move and to -5.4e-8 on the
relative objective, and its relative projected-gradient norm is 0.0097, which is
comfortably inside the 5e-2 tolerance. So on a well-posed stiff QP the solver
does reach the optimum, and a scale-invariant version of the library's own
criterion would have fired. The persistent False is a property of the stopping
test, not evidence that the answer is wrong. I think that is the publishable
version of the finding you anticipated, and it is more specific than "it does
not converge".

One thing I want to flag because it nearly fooled me. At horizon 5 the
comparison looks perfect, 0.000 degC RMS difference, 98.8 percent formal
convergence, zero clipping, zero constraint violations. It is completely
worthless. With a 5-minute horizon the controller cannot see past the thermal
lag, so the energy term dominates, it drives the oven down to its 10 degC floor,
the part cools from 28 to 11 degC and the degree of cure never leaves zero. Both
controllers agree perfectly because both are doing nothing, saturated against the
same lower bound. I only caught it because I checked the final cure state. It is
the same trap as the heat-up agreement you warned me about, and I have written
it into the report as an explicit check. On that heat-up point, you were right,
I measured it, 41.7 percent of the first 60 steps have both controllers pinned
at the slew limit, and 77.4 percent inside the exotherm window, so that
agreement is excluded from the claim.

So where this leaves the claim. I am not calling it equivalence. The honest
statement is "same QP, but the SNN-QP does not reliably converge", and that is
what the report concludes. The controllers now provably receive the same
problem, the applied controls agree to 0.714 degC RMS, the objective gap is
about 1e-4 where it is measurable, but the formal criterion is met on 0 percent
of steps at any horizon that actually cures the part, and about 13 percent of
applied moves are still corrected by the safety clip. The settled configuration
is horizon 10 with soft state constraints and the identical model, which cures
the part fully with 13.24 degC overshoot against the baseline's 13.77.

I agree on holding the hardware phase. The remaining software item is narrow now
and I think worth doing before any FPGA work: wrap a scale-invariant convergence
test around the solver, or raise the absolute-tolerance issue upstream, and
re-measure the convergence rate at the working horizon. That is the one thing
standing between the current result and a clean convergence number.

Everything is pushed to the branch. The report is in
`docs/PHASE4_VALIDATION_REPORT.md`, the raw evidence for every number is under
`results/`, and the README has a table mapping each claim to the script that
produces it, so you can rerun any of it directly. Happy to talk through any part
of it, particularly the infeasibility finding, since that one changed how I
understand the whole problem.

Eisa
