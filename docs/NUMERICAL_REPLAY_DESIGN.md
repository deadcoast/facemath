# Numerical runtime identity and evidence transfer

This patch starts from the actual 0.11.1 archive, SHA-256
`417bb4e81b9725c2ede1dad6d020c3af4658c85f82f768044ce606652a77c6d2`.
The inherited archive inventory and extracted tests precede source edits.

## Evidence and decision

The retained Windows/Ubuntu pilot has identical source observations and all
twelve saved qualification arrays, but different derived joint scores. A new
read-only trace holds the covariance and all input rows fixed. Ocular bases,
residuals and marginal solves match. The 16-dimensional Cholesky factor differs
by up to 1.04e-17; the following solve also differs when given an identical
factor. Replacing only the final norm reduction does not eliminate the mismatch.
This is evidence against package-version equality as an exact replay boundary.
It is not evidence that the measured geometry or fitting protocol is wrong.

Keep the existing mathematics, stored arrays, calibration ranks, split policy
and exact qualification comparator. Bind new acquisitions and replays to a
versioned numerical runtime profile instead. Its necessary compatibility checks
include interpreter/build, OS/architecture/libc, actual compiled distribution
bytes, NumPy CPU features, OpenCV dispatch/optimization/thread state, loaded
BLAS kernel identities and thread limits, and numerical dispatch environment
variables. Absolute installation paths and machine names are not identity.

NumPy documents its CPU dispatch attributes in
[`show_runtime`](https://numpy.org/doc/2.3/reference/generated/numpy.show_runtime.html).
The loaded BLAS identities and active limits come from
[`threadpoolctl`](https://github.com/joblib/threadpoolctl). These are introspection
inputs, not settings that FACEMATH changes to make an old result pass.

Binary identities describe an immutable software installation for the life of
the process. Restart after installing or replacing dependencies. Dynamic thread
and dispatch settings are sampled on each request. A matching profile permits
recomputation; only the existing result comparison establishes an exact replay.
The profile does not attest every hidden hardware/process condition or promise
universal determinism. Unknown or old version-only profiles cannot authorize a
new strict replay. Their original archives remain available for historical use.

## Operator and pipeline behavior

Expose a JSON runtime profile and a comparison command with field-level
differences and an unsuccessful exit status for incompatibility. Compare before
creating replay output or executing an observation. Report the differing fields
in worker/resume/finalization/replay failures; never rewrite a source environment
record to make it match. Byte verification remains independent of this host's
runtime and remains usable after transfer.

The transferable unit is still the complete frozen dataset and evidence bundle.
Transfer, internal byte consistency, runtime compatibility, exact numerical
replay, and independent anatomical accuracy are separate claims. A different
runtime can verify the bytes and inspect observations. It must use a distinct
new study to acquire/recompute new evidence under a different contract.

## Validation and next data gate

Exercise fresh-process profile equality, changed BLAS thread limits, changed
dispatch settings, changed binary identity, legacy profile refusal, and rejection
before observer/replay work. Run the real pinned native pilot and source,
installed-wheel and extracted-archive gates. Compare new raw observations with
the parent. Preserve both old cross-host numerical failures and new preflight
refusals; neither is counted as cross-platform exact replay success.

The larger cohort remains a distinct prospective experiment. The primary iBUG
[annotation source](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
provides semi-automatic 68-point references for research use. Its
[coordinate convention](https://ibug.doc.ic.ac.uk/resources/300-W/) is one-based.
A future adapter must pin source archives, preserve source-photo grouping and
original labels, subtract the declared origin once, retain unknown visibility
and training overlap, and freeze enough calibration groups before inference.
Do not label those references as newly performed human testing or use the
existing evaluation results to choose alpha, splits or regularization.
