# Versioned landmark correspondence repair

## Review

`LANDMARK_DIAGNOSIS.json` retains 20 HELEN development enrollments, 19
observations and one failure. Source PTS conversion, crop pixels and native
normalized-to-pixel coordinates match exactly. The selected nose correspondence
changes from MediaPipe vertex 1 to vertex 4: development median error decreases
from 0.119660 to 0.041223 reference eye spans. This selection used development
only. Chin remains vertex 152; the diagnosis supplies no supported replacement.

The parent is `packages/facemath-0.16.0-enrollment-registry-20261003.zip`,
SHA-256 `2fc6d608b1af69d92178244f80af430eaa74f01bec50106f6a3c6d05eceac1aa`.
Its extracted tests and command record are retained under
`runs/landmark-repair-20261004/parent*`, including any inherited failure.
The initial checkout environment produced six failures (OpenCV 5 and stale
0.8.0 package metadata). The retained supported Python 3.11 environment reproduced
1,156 passes and one Windows directory-symlink skip, including native smoke.

The production defect extends beyond one index. `observer_recipe` omitted the
correspondence, so changing the global anchor table could silently reinterpret
historical observations and admit an incompatible calibration. The shared
anatomical grammar must govern reference prediction and pair registration alike.

## Design

Keep the 478 stored detector coordinates and samples unchanged. New native
observations explicitly declare `mediapipe478.reference8.v2` in a new observation
format; historical v3 observations retain the original correspondence. Old
readers must reject the new format. The new mapping changes only `nose_tip`.
The new nose path includes vertex 4 between 5 and 1 so region measurements retain
the selected apex; historical contour paths remain byte-identical.
The reference annotation protocol and source PTS coordinates do not change.

Use one immutable correspondence definition for anatomy, reference prediction,
observer declarations and compatibility hashes. The observer recipe binds its
identifier and full anchor assignment. Existing error models already bind that
recipe: new/old calibration, mixed studies and mixed pair grammars must refuse.
Frozen worker requests pass the declaration into the provider, so historical
studies do not inherit a new default. Preserve legacy serialization and digests.

New PTS plans and recipes explicitly select correspondence in a v17 format.
Older prepared datasets keep their frozen recipe. Do not reinterpret a saved
model or reseal a historical result to claim a repair.

## Plan

1. Retain the extracted-parent baseline; reproduce the omitted-identity defect.
2. Implement observation/declaration serialization, shared grammar and recipe
   binding, native worker propagation, dataset selection and pair rejection.
3. Test actual selected coordinates/covariance/missingness, legacy round trips,
   unknown/downgraded identities, model incompatibility and full-mesh invariance.
4. Freeze a new HELEN photograph-root window `[132,192)` with 20 development,
   20 calibration and 20 evaluation groups. Keep seed, crop policy, shrinkage,
   noise floor, alpha and thresholds fixed. Bind the retained 255-case registry
   before inference. Fit a distinct joint model and report all enrolled outcomes.
5. Execute full tests, milestone-lab, verify/replay, installed wheel outside the
   checkout and extracted archive validation using the final source fingerprint.
   Retain raw logs and compare all raw observations across qualification.
6. On the two prespecified evaluation cases `case_00040` and `case_00041`, run the
   full 478-point pair instrument with default flow, edges and 96 sensitivity
   draws. Compare every raw geometry array before/after applying the research
   model. Preserve a failed execution if either case cannot support that check.

The [official MediaPipe canonical mesh](https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj)
provides vertex geometry; its relative nose positions support the diagnosis but
do not establish equivalence to the [300-W annotations](https://ibug.doc.ic.ac.uk/resources/300-W/).
The mapping choice is based on the retained development comparison, not a claim
that the mesh document validates the reference convention.

## Acceptance and limits

Final review found that qualification labeled native execution against any
non-machine reference as `live_human_observer_validation`. That included published
PTS with unknown exposure/visibility. The repair records native execution,
declared reference kinds and the absence of external provenance/anatomical
verification separately. The legacy human flag requires declared blind-human
references and a recorded native observation; it remains a declaration, not an
attestation. Published PTS sets it false. Original outputs and source are retained.
Final qualification repeats the identical frozen plan under a new fingerprint;
outcomes are already known, and no mapping, model setting or threshold is retuned.

Implementation acceptance requires that compatibility cannot cross mappings and
that the new native study executes and replays without modifying raw arrays.
Scientific results report source-coordinate error, group delivery/coverage and
widths; no post-evaluation mapping, radius, center or scatter tuning is allowed.
Published PTS agreement is not independent anatomical equivalence, visibility
validation, person independence or empirical query admission. Improvement in the
nose does not close the chin discrepancy or validate the full mesh.

The inherited mathematics inventory and one-CanonicalState architecture remain
authoritative. This repair does not claim completion of the historical research
inventory or add a second alignment.
