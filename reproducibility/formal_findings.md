# Findings of the SMT proof (2026-11-20)

The first run of `secureagent_iot.formal.prove` against the engine as it then stood (transcription
differential-tested against the engine: 20,000 random cases, 0 mismatches) returned counterexamples
to theorem S1 (every authorization satisfies the written specification):

1. `action=restrict_egress, high=True, approved=True, p=0, dur=1, n_cited=0` was ALLOWED.
   With a valid approval the engine let a high-criticality device be restricted regardless of
   detector probability and evidence. The specification requires the thresholds AND the approval.
   Cause: the criticality branch was checked before the thresholds and returned early.
   None of the earlier checks found it: the proposal grid holds no approvals, and the decision table
   had no row combining an approval with failing thresholds.
   Fix: thresholds are checked first; approval is required in addition.

2. After fix 1: `action=release_restriction, resolved=True, restricted=False` was ALLOWED.
   A release with nothing to release; harmless (a no-op), but outside the specification.
   Fix: a release is denied when the device is not restricted.

After both fixes all eight theorems (S1-S5, L1-L3) are proved, and decision-table rows
`approved_high_crit_low_p`, `approved_high_crit_thin_evidence` and
`release_resolved_nothing_restricted` pin the corrected behaviour.
