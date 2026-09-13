# FACEMATH · Wave 05

An integrated, auditable instrument for **facial geometry and observation error**, not identity recognition.

Wave 05 links prospectively frozen acquisition, one committed outcome per source, development-only bias estimation, independent residual calibration and held-out slices into the existing canonical geometry engine. Raw measured anatomy never moves to improve the error model.

## Run the complete development control

```shell
python -m pip install -e ".[dev]"
python -m facemath study-lab --out runs/wave5
python -m facemath verify runs/wave5/qualification
python -m facemath replay runs/wave5/qualification --out runs/replay
python -m pytest -q
```

Open `runs/wave5/qualification/report.html`. The lab has 91 enrolled schematic cases, 90 eligible attempts, a frozen development/calibration/evaluation split, two prescribed nondetections, one excluded case and a deliberately shifted held-out stratum. It is a falsification experiment, not human observer validation. A smaller error multiplier is **not** automatically successful generalization.

## Native acquisition

```shell
python -m pip install -e ".[observer,dev]"
python -m facemath observer-pin --model models/face_landmarker.task --out study/observer.json
python -m facemath study-freeze study/study.json --out studies/portrait-v1
python -m facemath study-run studies/portrait-v1 --model models/face_landmarker.task --out runs/portrait-v1
python -m facemath study-status runs/portrait-v1
python -m facemath study-finalize runs/portrait-v1 --out qualification/portrait-v1
```

Provide a local model and actual independently annotated source images. `examples/prospective.study.template.json` is a structural template, not a dataset. `docs/WAVE5_OPERATIONS.md` describes the complete protocol and how to use calibration with ordinary pair analysis.

**Executed boundary:** MediaPipe/model acquisition failed in the build environment. The live integration test remains skipped. API fixtures, a native subprocess failure-path test, and synthetic controls are not represented as successful live inference. See `evidence/wave5/native_environment.json` and the release validation record for actual executions.

## Integrated state

```text
Frozen source inventory + independent references
                    │
     pinned native observer / explicit fixture
                    │
       one receipt per included source
                    │
    existing source-coordinate validation
                    │
       development-only error model
                    │
       calibration + held-out slices
                    │
     existing CanonicalState and geometry
                    │
   shifted error-set centers, unchanged anatomy
                    │
      raw arrays / reports / exact replay
```

The same `Observation`, source-coordinate policy, reference grammar, covariance, canonical transforms, masks, geometric field and evidence contracts are used throughout. v4 zero-centered and v5 bias-centered calibrations share `ErrorCalibration`; there is no second registration or separate geometry calculator for the new model.

## Inspect the implementation

- [Mathematics, architecture and limits](docs/WAVE5_DESIGN.md)
- [Operational workflow and native study protocol](docs/WAVE5_OPERATIONS.md)
- [Raw array definitions](docs/RAW_ARRAYS.md)
- [Prior Wave 04 architecture](docs/WAVE4_DESIGN.md)
- [Earlier mathematical-completeness audit](docs/WAVE3_AUDIT.md)
- [Wave 04 README preserved](docs/README_WAVE4.md)

No human cohort or model weights are bundled. Group identity and blinding are curator declarations. Checksums establish consistency, not external attestation. The report makes no remote requests; the separate native worker is not a network sandbox. Review the upstream MediaPipe privacy notice before deployment.
