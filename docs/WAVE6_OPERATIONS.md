# Wave 06 operator guide

## New analysis runs

Install the package, then use the existing `analyze` command. Intrinsic reference8 analysis is on by default. `--no-intrinsic` disables it without changing existing geometric measurements. Supply either two source-bound observation JSON files or the optional real observer and its local model.

```shell
python -m facemath analyze a.png b.png --observation-a a.json --observation-b b.json --relation-length-tolerance 0.05 --relation-angle-tolerance-deg 5 --relation-area-tolerance 0.05 --relation-asymmetry-tolerance 0.05 --out pairs/run01
```

Lengths use each source's own reference eye spans, angles use radians internally (CLI accepts degrees), signed double areas use squared reference eye spans, and asymmetry uses the signed difference/sum of two lengths. The defaults are engineering tolerances, not learned cutoffs for human resemblance. The configuration is frozen in the measurement recipe.

To attach a localization model, add the same `--localization-calibration`, `--query-groups`, `--domain`, `--lineage-a` and `--lineage-b` arguments documented in Wave 05. Bias and zero-centered models use the same validated observation recipe. Development/calibration overlap is rejected. An unbounded calibration is rejected upstream; absence of calibration produces nominal metrics without invented bounds.

## Read the outputs

The ordinary report contains the new "Shape relations without a frozen alignment" section. It is not a replacement for the registered-point table. Source-eye diagrams show the raw points or error-set centers and full covariance ellipses; chart zoom is a display operation only. Select a relation to see anatomy names, units, frozen tolerance and exact array columns. Family/status/name filters do not alter the measurement.

`intrinsic.json` contains all 256 relations and the normalizer/shape ranges. `intrinsic.csv` contains every row with full numeric values. Selected-row CSV export uses those same values, not rounded strings. `arrays.npz` retains coordinates, coefficient vectors, covariances, masks, bounds, directions and candidate rotation. `events.jsonl` records the actual intrinsic stage.

A source bound may be unavailable even though its observed nominal value exists. Read its reason. If the eye-span outer range can reach zero, normalized reference geometry cannot be bounded by this implementation. It does not freeze the denominator to force a number. Missing points stay in the prespecified family denominator as unknowns.

"Within tolerance" means every admitted scalar gap is under that threshold, conditional on source sets being valid. "Outside" means every admitted gap is over it. "Unresolved" means the interval crosses it. None of these is a percentage probability of resemblance. A deterministic fraction over the relation inventory is also not a population percentile.

The refitted shape range encloses the normalized centered shape discrepancy after independently optimizing a proper rotation for each candidate reference pair. Its raw observed RMS and set-center RMS are different values. Error-model centers never replace raw coordinates in the earlier geometry.

## Execute the complete control

```shell
python -m facemath intrinsic-lab --out runs/wave6
python -m facemath verify runs/wave6/qualification
python -m facemath verify runs/wave6/pair_02_mouth_8px
python -m facemath replay runs/wave6/qualification --out runs/study-replay
python -m facemath replay runs/wave6/pair_02_mouth_8px --out runs/pair-replay
```

The fixed experiment retains 49 enrolled schematic sources, 48 scripted attempts, development/calibration/evaluation membership and failures. Four fresh queries apply mouth expansions at declared values. `query_ground_truth.npz` retains their source coordinates; `set_controls/controls.npz` retains every sampled admissible shape and scalar value. The samples are checks, not an estimator for the analytic ranges.

Outputs are create-only. Numerical replay uses captured observations and the matching code/runtime; it does not run the observer again. Source changes, including numerical implementation changes, need a new run rather than relabeling it an exact replay.

## Empirical gate

No human score is furnished by these controls. The optional real model still requires installation and an explicit local compatible model file. Follow the preserved annotation/reference protocol and prospective study instructions to obtain genuine held-out localization results. Independently annotated real portraits, supported acquisition conditions and visibility review are still necessary; new geometric formulas do not supply them.
