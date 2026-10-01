# Wave 10 · One operator workflow

## Installation and first result

Python 3.11 or later. From the extracted project directory:

```shell
python -m pip install -e ".[dev]"
python -m facemath --version
python -m facemath demo --size 160 --draws 8 --no-flow --out runs/demo
python -m facemath overview runs/demo
```

The demo needs no model. It is synthetic. Its uncalibrated reference-bound fields are **unavailable**, not zero. The detailed report still contains observed geometry and exact arrays. Open reports from their extracted directory so relative links work.

All output directories are create-only except the explicit resumable `study-run` and annotation intake. Use a new output name for another analysis. Do not erase evidence to make a rerun succeed.

## 1. Prepare native prerequisites explicitly

Install the optional observer extra on the intended machine and obtain the compatible task model from the official Google FaceLandmarker documentation. FACEMATH does not silently download model bytes.

```shell
python -m pip install -e ".[observer,dev]"
python -m facemath observer-pin --model models/face_landmarker.task --out study/observer.json
python -m facemath observer-smoke original_portrait.png --model models/face_landmarker.task --out runs/smoke
python -m facemath verify runs/smoke
```

An absent model/runtime yields `not_attempted` with no detector call. Valid preflight leads to the existing isolated worker; raw request/response, process output and observation are retained. `observed` means the worker produced a source-bound observation, **not** that anatomy has been independently validated. Native runtime privacy/network behavior is not sandboxed by this wrapper.

## 2. Prepare sources and blank annotations

Fill `examples/workspace.spec.template.json` with a complete source inventory. The template is deliberately not a sufficient study. Choose the source apertures, grouping, acquisition predicates, estimator and empirical criteria using development material, before observing calibration/evaluation outcomes.

A workspace spec has exactly `format`, `study`, `rater_ids`, `sources` and `source_description`. `study` contains the existing prospective settings but no cases or observations. `observer` can reference your local `observer.json`. Each source declares case ID, group, split, source image and all known lineages. Related images cannot cross groups or splits.

```shell
python -m facemath workspace prepare study/workspace.spec.json --out work/portraits
python -m facemath workspace status work/portraits
```

The output holds immutable source PNGs and one blank workbench per included source/rater. The assigned rater must use their declared ID on export. No observer predictions are shown by this preparation command. The software cannot prove that a person was previously unexposed to them.

Mark unobservable points rather than guessing anatomy. The current reference normalization needs all four eye corners. Native empirical admission accepts `human_blind` or reviewed `external_dataset` records with `prediction_exposure="none_declared"`, not known-exposed annotations.

## 3. Import labels and preserve revisions

```shell
python -m facemath workspace import work/portraits labels/source001-rater-a.json --case-id source001
python -m facemath workspace import work/portraits labels/source001-rater-b.json --case-id source001
python -m facemath workspace status work/portraits
```

Repeating the identical import is idempotent. Importing a corrected label creates a new revision and supersession receipt; prior coordinates remain retained. Source/rater mismatches are rejected. A renamed rater does not establish a new independent human.

A crash can leave an explicitly uncommitted label file. Status refuses it rather than selecting it. Preserve the workspace and investigate the partial write; this release does not provide automatic forensic repair. A new separately named workspace is safer than deleting evidence without a record.

## 4. Freeze acceptance with the study

An example policy inside `study`:

```json
{
  "admission_policy": {
    "format": "facemath.empirical_admission_policy.v10",
    "required_categories": ["protocol_clear", "protocol_degraded"],
    "minimum_groups": 30,
    "minimum_delivery": 0.8,
    "minimum_coverage": 0.8,
    "family_error": 0.05,
    "evidence_kind": "native_reference",
    "independent_groups_declared": true,
    "sampling_design": "Replace with the actual group sampling protocol and justification."
  }
}
```

These are engineering examples, not empirically validated human thresholds or a sample-size recommendation. Required categories must match the measured policy. The existing estimator must be `development_joint_ocular_stratified` with an embedded source measurement policy. Merely changing a domain/category name does not establish its validity.

```shell
python -m facemath workspace freeze work/portraits --out studies/portraits-v1
python -m facemath verify studies/portraits-v1
```

The lock preserves every reference revision and the actual selected labels. It reports whether the evaluation inventory could pass even with all successes. Label readiness is not statistical feasibility; feasibility is not expected success. Underpowered studies remain possible research but cannot pass their frozen acceptance rule.

## 5. Collect and qualify

```shell
python -m facemath study-run studies/portraits-v1 --model models/face_landmarker.task --max-cases 20 --out runs/portraits-v1
python -m facemath study-status runs/portraits-v1
python -m facemath study-run studies/portraits-v1 --model models/face_landmarker.task --out runs/portraits-v1
python -m facemath study-finalize runs/portraits-v1 --out qualification/portraits-v1
python -m facemath overview qualification/portraits-v1
```

The existing runner never silently reinvokes committed cases. Started-but-uncommitted attempts remain interrupted outcomes. Eligible failures stay in complete group denominators. Finalization fits only development, calibrates only calibration and evaluates the frozen held-out criteria. Passing only one of several requested categories does not admit the study.

Do not refit q, change source rules, lower thresholds, drop failures or select a favorable category using evaluation results. A newly designed study must be separately recorded; repeated adaptive study selection is not statistically controlled by this release.

## 6. Prepare complete measured query groups

Follow `WAVE9_OPERATIONS.md` for the source-group JSON. Every related context source participates, not only the portrait selected for the pair.

```shell
python -m facemath acquisition-group query/a.group.json --policy qualification/portraits-v1/localization_calibration.json --out query/a-source
python -m facemath acquisition-group query/b.group.json --policy qualification/portraits-v1/localization_calibration.json --out query/b-source
```

For the empirically gated path, query source/context pixels, groups and lineages must be disjoint from all development, calibration, evaluation and excluded enrollment. Re-encoded duplicates and undisclosed subject dependence still require honest curator grouping; this is not identity recognition.

## 7. Compare using the same instrument

```shell
python -m facemath analyze query/a.png query/b.png --model models/face_landmarker.task --qualification qualification/portraits-v1 --acquisition-a query/a-source --acquisition-b query/b-source --out pairs/qualified
python -m facemath overview pairs/qualified
python -m facemath verify pairs/qualified
python -m facemath replay pairs/qualified --out pairs/replay
```

`--qualification` carries the complete study, replays it, requires acceptance, and derives the domain. It is mutually exclusive with `--localization-calibration`. Imported image-bound observations can replace `--model` through the existing `--observation-a` and `--observation-b` flags.

`--localization-calibration` remains a research path, not an assertion of successful held-out acceptance. A failure of the empirical route never silently falls back to research. The exact partial instrument journal is written to `<out>.failure.events.jsonl`, without replacing a previous failure journal.

## 8. Inspect and transfer

`overview` prints a compact tree; `overview --raw` preserves the complete JSON result. `summary.json` shows existing shape measurements, interval availability and dependent relation-fraction bounds. The detailed workbench retains acquisition grids, covariance modes, exact arrays and all relation rows.

The admitted pair contains `qualification_snapshot.zip`. It includes source/reference/observation evidence; treat it as potentially sensitive data. Moving that pair transfers the complete retained study. No externally attested timestamps or truth claims are supplied by its hash manifest.

Exact replay requires the corresponding implementation and numerical library versions. Install an older release alongside its historical bundle rather than relabeling a new computation as the historical one. New CI portability configurations are not proof that those platforms ran during this release.
