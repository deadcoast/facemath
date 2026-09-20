# Wave 08 operator guide

## Complete reproducible control

```shell
python -m pip install -e ".[dev]"
python -m facemath stratified-lab --out runs/wave8
python -m facemath verify runs/wave8/qualification
python -m facemath verify runs/wave8/pair_02_mouth_8px
python -m facemath replay runs/wave8/qualification --out runs/study-replay
python -m facemath replay runs/wave8/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Outputs are create-only. Use a new directory for another experiment. `study-run` is the explicit resumable operation; it never retries a completed case silently.

Open `runs/wave8/qualification/report.html` and `runs/wave8/pair_02_mouth_8px/report.html` from the extracted directory so relative source/evidence links resolve. The demonstration is synthetic. Read `laboratory.json`, `inputs/fixture_contract.json`, and the retained partial journals for refused queries.

## Freeze acquisition categories in a native study

Continue using the independent annotation workbench, `observer-pin`, and the same prospective-study manifest. The optional MediaPipe runtime and a compatible local model remain prerequisites; this release does not download weights automatically or bundle them.

Select the following fields inside a COMPLETE manifest:

```json
{
  "bias_mode": "development_joint_ocular_stratified",
  "joint_error": {"shrinkage": 0.02, "floor_eye_spans": 0.001},
  "strata": ["protocol_clear", "protocol_degraded"],
  "stratified_error": {
    "definition": "State the acquisition protocol that assigns whole groups BEFORE predictions or errors are viewed.",
    "minimum_calibration_groups": 9
  }
}
```

This fragment is not a complete study and these values are engineering examples, not validated human-study recommendations. Nine is only the finite-order-statistic algebraic minimum at alpha0.1 without selected infinite scores; it is not a practical sample-size justification. Change the policy only in a new study. No dependent group or declared lineage may cross routing categories. Images of a known dependent source must remain correctly grouped, not renamed to increase counts.

```shell
python -m facemath study-freeze study/study.json --out studies/acquisition-v1
python -m facemath study-run studies/acquisition-v1 --model models/face_landmarker.task --out runs/acquisition-v1
python -m facemath study-finalize runs/acquisition-v1 --out qualification/acquisition-v1
```

The observer worker receives neither categories nor reference labels. This argument separation is not a filesystem/network sandbox, and it does not prove human blinding or category correctness. The native inference boundary remains unvalidated in the release environment.

## Apply a category-conditioned model

Supply two explicit query categories, along with the existing source-bound observations, groups, lineages and exact domain:

```shell
python -m facemath analyze a.png b.png --observation-a a.json --observation-b b.json --localization-calibration qualification/acquisition-v1/localization_calibration.json --query-groups fresh_a fresh_b --lineage-a new_source_a --lineage-b new_source_b --query-strata protocol_clear protocol_degraded --domain "exact frozen domain" --out pairs/acquisition-qualified
```

The same interface can use the optional local observer through `--model`. Do not change category based on which q produces a preferred resemblance result. The instrument binds the declarations to the actual resulting observations, but cannot establish whether an operator's declared acquisition metadata is true.

Unknown, absent, insufficient and unbounded categories do not receive a pooled fallback. A calibrated request fails with a retained partial journal; ordinary nominal analysis is still available by explicitly omitting calibration. Replaying the bound pair needs no new category choice: its original immutable routes are loaded and revalidated.

Older v4/v5/v7 models retain their previous scalar-radius paths. Do not attach `--query-strata` to a non-stratified model. Historical bundles remain inspectable, but exact numerical replay requires their matching code and numerical runtime.

## Raw quantities

`q_A` and `q_B` are full-shape localization-error multipliers under the same scoring recipe, not similarity percentages. The original perturbation covariance, fitted model covariance and admitted set radius remain separate. In a mixed-radius pair, the old single `quantile` is null while `source_quantiles` contains two finite values; explicit bound-state/availability fields disambiguate that from an unbounded result.

`arrays.npz` retains the complete observed geometry and derived fields. `localization_calibration.json` contains the full shared model, category inventories and source routes. `events.jsonl` includes the actual routing stage before geometry construction. The study arrays include category/radius indices and failure masks; JSON null/array false masks do not mean a zero error.
