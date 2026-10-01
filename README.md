# FACEMATH · Wave 10 milestone

A local Python instrument for **projected facial geometry, observation error, and auditable source evidence**. It does not identify people or turn a visually chosen percentage into a biometric measurement.

Wave 10 consolidates source intake → independent annotation → frozen acquisition → error-model qualification → measured query sources → geometric comparison into one operational workflow. The existing `Observation`, `CanonicalState`, error models, relation graph and evidence bundles remain authoritative.

**New at this milestone:** an append-only reference workspace; a prespecified held-out acceptance policy; full-study numerical replay before the optional empirically gated query path; evaluation/context leakage checks; an actual native-execution smoke command; a compact result readout. Ordinary research analysis remains available and explicitly labeled.

## First executable result

```shell
python -m pip install -e ".[dev]"
python -m facemath demo --size 160 --draws 8 --no-flow --out runs/research-demo
python -m facemath overview runs/research-demo
python -m facemath verify runs/research-demo
```

Open `runs/research-demo/report.html`. This is a known-coordinate **synthetic** demonstration. The fraction of a declared relation inventory within a stated tolerance is a computable percentage; it is not a probability of resemblance. Unresolved/missing relations remain in the denominator.

## Complete milestone control

```shell
python -m facemath milestone-lab --out runs/wave10
python -m facemath overview runs/wave10/positive/qualification
python -m facemath overview runs/wave10/negative/qualification
python -m facemath overview runs/wave10/admitted_pair
python -m facemath replay runs/wave10/admitted_pair --out runs/wave10-pair-replay
python -m pytest -q
```

The positive and negative workflows are frozen independently before their scripted outcomes run. The positive branch is an **engineered containment/acceptance control**, not a representative or exchangeable human study. The negative branch keeps the same policy but shifts the observer's errors in one required category. All requested categories must pass; no successful subset is selected after seeing outcomes.

## Start an actual reference study

```shell
python -m facemath workspace prepare study/workspace.spec.json --out work/portraits
python -m facemath workspace status work/portraits
python -m facemath workspace import work/portraits labels/portrait-rater-a.json --case-id portrait_0001
python -m facemath workspace freeze work/portraits --out studies/portraits-v1
python -m facemath observer-smoke portrait.png --model models/face_landmarker.task --out runs/native-smoke
python -m facemath study-run studies/portraits-v1 --model models/face_landmarker.task --out runs/portraits-v1
python -m facemath study-finalize runs/portraits-v1 --out qualification/portraits-v1
```

The spec needs actual images, independently produced reference annotations, declared dependent groups/lineages, a pinned compatible runtime/model and a source-measurement policy designed on development data. The example template is not a sufficient dataset. See [the consolidated operator guide](docs/WAVE10_OPERATIONS.md).

`observer-smoke` never downloads a model or installs a runtime. Missing prerequisites produce **not_attempted**, not a fictional detector failure or accuracy result. A successfully running detector would still need independent reference evaluation.

## Research versus empirically gated queries

`--localization-calibration model.json` applies a research calibration as before. The report does not infer empirical approval merely because its radius is finite.

`--qualification qualification/portraits-v1` loads the **complete retained study**, replays its numerical qualification, requires all frozen acceptance criteria to pass, and rejects query source/context overlap with development, calibration, evaluation and excluded enrollment. Original source-group pixels remain necessary. The portable pair stores the qualification snapshot, not just an approval string.

```shell
python -m facemath analyze query/a.png query/b.png \
  --observation-a query/a.observation.json --observation-b query/b.observation.json \
  --qualification qualification/portraits-v1 \
  --acquisition-a query/a-source --acquisition-b query/b-source --out pairs/qualified
```

Use `--model` instead of imported observation files for a pinned native observer. Source and acceptance gates run before its deferred construction. Failed instrument invocations preserve an actual partial journal at `<out>.failure.events.jsonl`, without overwriting previous evidence.

## Read the evidence

- `summary.json`: compact view of existing measurements, bounds, relation fractions and acceptance status.
- `evidence.json`, `intrinsic.json`, `arrays.npz`: original measurements and complete numerical arrays.
- `events.jsonl`: actual instrument-stage events, not private neural telemetry.
- `qualification_snapshot.zip`: retained raw study, observations, labels, receipts and qualification behind an admitted pair.
- Workspace `labels/`: original annotations and explicit superseding revisions. Freeze records the complete chain and selected latest labels.

Hashes establish byte consistency; replay establishes consistency with the retained numerical inputs and implementation. Neither proves external chronology, true subject independence, human blinding, accurate source metadata, or resistance to an attacker recreating an entire consistent dataset.

## Scientific and release boundaries

The release's `evidence/WAVE10_VALIDATION.json` records what was executed. Synthetic controls and API fixtures are not live detector validation. The actual native smoke outcome is separate. No new similarity score for the original video's two people is manufactured.

Eight-point reference calibration does not validate all 478 vertices, 3D anatomy or identity. Source gradient rules are not face-quality classifiers. A passed held-out policy is conditional on its sampling assumptions and does not guarantee performance under hidden shift or control repeated adaptive studies. Exact replay requires the matching package code and numerical runtime.

## Development map

[Design and audit](docs/WAVE10_DESIGN.md) · [Mathematics](docs/WAVE10_MATHEMATICS.md) · [Operations](docs/WAVE10_OPERATIONS.md) · [Milestone architecture map](docs/MILESTONE_MAP.md) · [Next development gates](docs/WAVE10_ROADMAP.md) · [Contributor contract](AGENTS.md)

The inherited section-number registry is a research inventory, **not proof of implementation of every historical mathematical analogy**. The active implementation and its non-equivalent/missing research methods are separately documented by `facemath capabilities` and the retained Wave 3 audit.
