# Wave 07 operator guide

## Run the integrated control

From the extracted project directory (PowerShell, zsh or bash):

```shell
python -m pip install -e ".[dev]"
python -m facemath joint-lab --out runs/wave7
python -m facemath verify runs/wave7/qualification
python -m facemath verify runs/wave7/pair_02_mouth_8px
python -m facemath replay runs/wave7/qualification --out runs/study-replay
python -m facemath replay runs/wave7/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

The laboratory uses a scripted synthetic observer, not the native extra. Output directories are create-only. Open `runs/wave7/pair_02_mouth_8px/report.html` **from the extracted folder** so relative source/evidence links resolve.

The study qualification report retains the deliberately shifted held-out category. The shape workbench's smaller interval is not evidence that this failed acquisition category became supported.

## Select joint errors in a real prospective study

Use the existing source/reference workbench and native model pinning from `WAVE5_OPERATIONS.md`. Before freezing, select these fields in the study manifest:

```json
{
  "bias_mode": "development_joint_ocular",
  "joint_error": {
    "shrinkage": 0.02,
    "floor_eye_spans": 0.001
  }
}
```

These are fields within a complete prospective study, not a runnable standalone manifest. The values are prespecified engineering choices, not empirically recommended human settings. Provide adequate group-separated development/calibration/evaluation data under a frozen acquisition domain and independent reference protocol. The algebraic minimum number of groups is not an accuracy or representativeness recommendation.

Then use the existing commands:

```shell
python -m facemath study-freeze study/study.json --out studies/joint-v1
python -m facemath study-run studies/joint-v1 --model models/face_landmarker.task --out runs/joint-v1
python -m facemath study-finalize runs/joint-v1 --out qualification/joint-v1
```

Study-run resumes only never-started eligible cases; failures are not silently retried. Native setup still needs a compatible installed MediaPipe runtime and a locally supplied model. No model is bundled or downloaded implicitly. This build environment did not execute real MediaPipe inference.

## Attach the qualified model to a pair

The existing `analyze` command accepts the v7 calibration without a second analysis pipeline:

```shell
python -m facemath analyze a.png b.png --observation-a a.json --observation-b b.json --localization-calibration qualification/joint-v1/localization_calibration.json --query-groups new_group_a new_group_b --lineage-a new_source_a --lineage-b new_source_b --domain "exact-declared-domain" --out pairs/joint
```

Use matching image-bound observations from the frozen observer recipe. The same model/runtime/domain/perturbation contracts and development/calibration leakage checks apply. The calibration cannot qualify all 478 landmarks from eight reference labels. It neither corrects raw coordinates nor estimates identity.

Without calibration, nominal observed geometry remains available; the instrument does not invent joint covariance. Existing v4/v5 calibrations retain their existing product-marginal path. Existing historical bundles remain readable, but exact replay requires their own implementation and numerical runtime; source changes must not be relabeled as identical replay.

## Read the output

`intrinsic.json` and the same profile embedded in `evidence.json` include `joint_error_geometry`. `arrays.npz` holds all old observed arrays plus the transported joint covariance, marginal blocks, same-set relaxed bounds, and eigenmodes. The original jitter covariance stays under `intrinsic_source_covariance`.

The mode slider shows an admissible error vector, not another observed face. The heatmap can be exported with exact numeric values. Every covariance entry and mode is available in NPZ. The browser computes only display locations/correlation normalization, not new bounds or decisions.

Interpret interval tightening against the same-set marginal relaxation as reduced mathematical over-approximation. Both cross-point correlation and the single joint norm budget are lost in the marginal-product relaxation. Compare empirical coverage separately. Never compare q values alone across different score definitions.
