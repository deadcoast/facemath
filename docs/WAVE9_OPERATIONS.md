# Wave 09 operator guide

## Reproduce the included synthetic experiment

From the extracted project (PowerShell, zsh or bash):

```shell
python -m pip install -e ".[dev]"
python -m facemath measured-lab --out runs/wave9
python -m facemath verify runs/wave9/qualification
python -m facemath verify runs/wave9/query_group_a
python -m facemath verify runs/wave9/pair_02_mouth_8px
python -m facemath replay runs/wave9/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Open qualification/report.html and pair_02_mouth_8px/report.html from the extracted run. Data are synthetic renderer coordinates and scripted observer errors. `laboratory.json` retains all prespecified queries, hidden-shift failures, negative controls and actual observer-call accounting. Directories are create-only; reruns need a new name. `study-run` is the explicit resume command, not retry-until-success.

## Freeze a measured source protocol

Continue the independent annotation and pinned observer setup in WAVE5_OPERATIONS.md. Choose apertures, predicates and group definitions on development material before calibration and evaluation outcomes are inspected. `examples/measured_acquisition_policy.example.json` is an engineering SCHEMATIC example, not suitable unmodified as a claim of validated portrait quality.

Inside the complete prospective study manifest:

- keep bias_mode `development_joint_ocular_stratified` and the existing joint_error fields;
- put the entire policy object in `stratified_error.measurement_policy` (not a filename);
- use the policy's categories including `__unmatched__` and `__ambiguous__` in strata;
- make maximum_cases_per_group equal the policy maximum_group_members;
- each case can declare acquisition_aperture `[x0,y0,x1,y1]` in stored pixels; omission means full raster;
- omit authored stratum. A supplied conflicting category is rejected, never preferred over pixels.

`examples/measured.study.template.json` is a structural template with placeholder files and an illustrative policy. It is NOT a sufficient sample-size recommendation. Fill all required source, reference, model, grouping and domain data. Known repeats stay in one group/lineage. Every enrolled member, including observation-excluded members, contributes to that group's pixel extrema.

```shell
python -m facemath study-freeze study/study.json --out studies/measured-v1
python -m facemath study-run studies/measured-v1 --model models/face_landmarker.task --out runs/measured-v1
python -m facemath study-finalize runs/measured-v1 --out qualification/measured-v1
```

Only `study-run` invokes the optional native observer. Unsupported pixel categories are still observed during the prescribed study, so their errors remain in accounting. The optional MediaPipe integration still needs independently acquired compatible package/model assets and real annotation validation.

## Measure complete query groups, not just category strings

Create a JSON next to your original portraits, with all related source members:

```json
{
  "format": "facemath.acquisition_group_spec.v9",
  "group_id": "new_source_group_b",
  "members": [
    {"image": "portrait_b.png", "lineage_ids": ["session_b"], "aperture": [20, 20, 220, 250]},
    {"image": "portrait_b_context.png", "lineage_ids": ["session_b_context"], "aperture": [20, 20, 220, 250]}
  ]
}
```

Aperture coordinates above are illustrative; they must be valid for your actual raster and chosen according to your frozen protocol. No inference from existing landmark files is performed.

```shell
python -m facemath acquisition-group query_a.group.json --policy qualification/measured-v1/localization_calibration.json --out query/a-source
python -m facemath acquisition-group query_b.group.json --policy qualification/measured-v1/localization_calibration.json --out query/b-source
python -m facemath verify query/b-source
```

The policy argument accepts the exact calibration model or its standalone policy JSON. Output contains original PNGs, the exact numerical grayscale grids, predicates, assignments and a report. Handwritten feature vectors/categories are not accepted by this command.

## Use those measured sources in the existing analyzer

```shell
python -m facemath analyze portrait_a.png portrait_b.png --observation-a a.observation.json --observation-b b.observation.json --localization-calibration qualification/measured-v1/localization_calibration.json --domain "exact frozen domain" --acquisition-a query/a-source --acquisition-b query/b-source --out pairs/measured
```

Groups and complete lineage lists are derived from the measured source bundles. Existing explicit --query-groups/--lineage-a/--lineage-b/--query-strata arguments must agree exactly when supplied. A selected portrait must match one of its original decoded member rasters. All context members are checked for known development/calibration leakage.

Use --model instead of observation files for actual local inference. The CLI delays native initialization until the measured acquisition gate passes. A rejected calibrated request records its actual partial journal and does not silently retry nominal analysis. Deliberate nominal uncalibrated geometry is a different invocation without calibration or acquisition groups; standalone source diagnostics remain available.

Pair replay loads the saved original group pixels and remeasures them. It does not ask the operator to choose categories again. The calibrated model's serialized group-digest claims alone do not authorize deployment.

## Interpretation and display

The measured category is a reproducible protocol result. Rule permission, sufficient category calibration, localization coverage and geometric resemblance remain different questions. Noise can pass a gradient threshold; a real face can fail it. Neither result identifies a person or establishes human localization accuracy.

The source workbench lets you change the displayed group/member and toggle original raster versus saved analysis grid. These are display choices only. The table shows that source's values and the unchanged whole-group min/max. Exact CSV exports all group members, not just the selected row. The category, aperture and thresholds are not editable in a result report.
