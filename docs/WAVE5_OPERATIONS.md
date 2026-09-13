# Wave 05 operator guide

## Run the complete controlled laboratory

From the extracted project directory, in PowerShell, zsh or bash:

```shell
python -m pip install -e ".[dev]"
python -m facemath study-lab --out runs/wave5
python -m facemath verify runs/wave5/frozen
python -m facemath study-status runs/wave5/acquisition
python -m facemath verify runs/wave5/qualification
python -m facemath replay runs/wave5/qualification --out runs/qualification-replay
python -m facemath replay runs/wave5/qualified_pair --out runs/pair-replay
python -m pytest -q
```

Open `runs/wave5/qualification/report.html` in a browser. The generated schematic sources are conspicuously labeled synthetic. Every failed fixture outcome remains in the report. The fixed pair is chosen by prespecified indices, not by reviewing its result.

Use new output directories for another run. `study-run` alone supports resuming the SAME output session. `study-finalize`, `study-freeze`, `study-lab` and `replay` are create-only operations. Do not overwrite forensic evidence to make a command work.

## Native study on a machine with the runtime and model

1. Install the observer extra, obtain the compatible MediaPipe model from the official task documentation, and keep it locally. No model is downloaded implicitly or bundled here.
2. Collect original portraits and prediction-blind reference8 annotations with the existing `annotate` workbench. Review the domain, image resolution, pose, expression, visibility, rater policy and lineage grouping BEFORE inspecting predictions.
3. Create the model/runtime declaration:

```shell
python -m pip install -e ".[observer,dev]"
python -m facemath observer-pin --model models/face_landmarker.task --out study/observer.json
python -m facemath doctor --model models/face_landmarker.task --out study/runtime.json
```

`observer-pin` reads the model digest and installed runtime version. A digest is not evidence that arbitrary file bytes constitute a valid model. When preparing a study on another machine, `--runtime-version` explicitly declares the intended version; acquisition still requires an exact installed match.

4. Prepare `study/study.json`, following `examples/prospective.study.template.json`. Observer can be the relative filename `observer.json`. Case paths are explicit relative POSIX paths (including on Windows). Use separate groups/lineages for development, calibration and evaluation. Paths cannot escape the study root or be symlinks. An `excluded` case needs a reason frozen before collection; its enrolled image remains in accounting.
5. Freeze, collect in bounded batches, then resume:

```shell
python -m facemath study-freeze study/study.json --out studies/portrait-v1
python -m facemath study-run studies/portrait-v1 --model models/face_landmarker.task --max-cases 20 --out runs/portrait-v1
python -m facemath study-status runs/portrait-v1
python -m facemath study-run studies/portrait-v1 --model models/face_landmarker.task --out runs/portrait-v1
python -m facemath study-finalize runs/portrait-v1 --out qualification/portrait-v1
python -m facemath verify qualification/portrait-v1
```

The timeout comes from the frozen study, not a mutable resume flag. A changed model, runtime, code or protocol needs a new explicitly distinct study. A successful resume invokes only never-started cases. A started-but-uncommitted case remains an interrupted failed attempt, even when a partial prediction was left behind. A deliberate retry belongs in a separately prespecified study; do not erase the old case directory.

`study-status --raw` prints all retained receipt contents. Default console output omits verbose receipts but points to their complete on-disk directory. Native stdout/stderr and response files reside beside each receipt. A contract error stops collection and prevents finalization. A missing runtime/model preflight does not count as a detection attempt.

## Apply the selected localization model to geometry

The ordinary `analyze` command accepts both v4 zero-centered calibration and v5 development-centered calibration:

```shell
python -m facemath analyze query_a.png query_b.png --model models/face_landmarker.task --localization-calibration qualification/portrait-v1/localization_calibration.json --query-groups query_a_group query_b_group --lineage-a query_a_session --lineage-b query_b_session --domain "exact-frozen-domain" --out pairs/query
```

For fully pinned observation inputs, export them with the same model/noise-floor/augmentation recipe and use `--observation-a`/`--observation-b` instead. Query images/groups/lineages cannot overlap development or calibration. Matching a string domain does not prove actual domain membership: inspect study slices and acquisition conditions before interpreting bounds. This release does not provide automated domain-membership recognition or deployment certification.

## What the outputs mean

- Raw geometry: the unmodified observed coordinates in the shared registered frame.
- Raw localization error: distance from the observed source coordinates to independent labels.
- Centered residual: raw error minus the development-fit error model; NOT a remeasured coordinate.
- q: simultaneous error multiplier from calibration groups, NOT resemblance or probability.
- Held-out coverage: fraction of eligible evaluation groups covered under that error model, including unavailable outputs as misses when the envelope is finite.
- Unseen stratum: missing calibration representation, NOT proof that nominal coverage transfers.

The native worker receives no label/split/stratum arguments. It still executes in the caller's OS account; this is not a security sandbox. Offline HTML and local numerical replay do not guarantee upstream runtime network silence.
