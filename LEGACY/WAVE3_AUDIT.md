# Audit of the supplied Wave 2 package

The starting point was the actual attached `facemath_wave_02.zip`, extracted and tested before changes. The inherited baseline produced 15 passing tests. Those tests established their assertions, not the much broader claims in earlier prose. `evidence/inherited-baseline.txt` preserves the run output. The final release metadata records the parent archive's actual SHA-256.

## Corrected active faults

| Finding | Consequence | Wave 3 action |
|---|---|---|
| Low-rank crossfade interpreted as shared facial structure | Any unrelated pair can satisfy it | Exact rank-null proof, unrelated-texture controls, sequence inadmissibility marker |
| First/last screen-recording frames treated as clean portrait endpoints | UI, scaling, multiple faces and blending contaminate measurements | Separate video acquisition inspector; no automatic endpoint face score |
| One-way least-squares alignment used without measuring gauge dependence | Similarity could depend on which face was target and which region anchored the fit | Symmetric-scale registration, symmetric normalizer, exchange-order tests, regional holdouts |
| Proper-rotation weighted fit omitted signed singular values from scale | Reflection-constrained scale could be mathematically wrong | Corrected Umeyama primitive with analytic reflected-rectangle optimum |
| Infinitesimal strain treated as generic structural deformation | Finite rigid rotations can look deformed | Exact per-triangle Jacobian, Green–Lagrange strain, log singular stretch |
| Dense flow confidence could be read as anatomical confidence | Flat or misleading image matches could look certain | Independent texture/cycle diagnostics, no score fusion, flat-image negative controls |
| Observation mutability and coordinate provenance were too weak | Array edits or image substitution could invalidate prior measurements silently | Immutable bytes-backed observation arrays, full image/schema/provider binding |
| `linear_cka` centered the caller's arrays in place | A diagnostic could silently mutate later computations | Non-mutating implementation and regression test |
| Procrustes allowed reflection implicitly | Mirrored geometry could be normalized away | Proper rotation by default; explicit reflection opt-in on inherited primitive |
| Undirected spectral routines accepted incompatible graph data | Directed/negative graphs and isolated-node conventions could give misleading spectra | Validate undirected nonnegative input; explicit zero isolated-node Laplacian |
| Temporal mask did not govern the low-rank downsample consistently | Excluded screen areas could still dominate the decomposition | Apply selected mask to the inherited sequence sample and mark interpretation limits |
| Calibration accepted values without sufficiently bound context | Scores from different instruments/protocols could be pooled | Exact recipe/runtime/model/grammar matching; verified-bundle ingestion |
| Repeated images counted as independent comparison rows | Naïve bootstrap overstated independent sample size | Connected image/declared-subject component bootstrap and query leakage checks |

## Earlier mathematical-completeness claim is not upheld

The inherited `coverage.py` has an entry for each section 20–180. Its coverage assertion checks those integer keys. It does not test the mathematics described by every section. Several entries refer to interfaces; others point to different or weaker algorithms. Sections 0–19 were not in that registry.

Concrete non-equivalences are now recorded in `observation/capabilities.py`: CKA is not InfoNCE; a Gabor summary is not a complete scattering transform; a fixed-coupling distortion is not a GW optimizer; vertex normals do not fit a 3D morphable model; 0D merge deaths are not all-dimensional persistence; empirical ranks are not Bayesian posteriors; compression proxies are not exact Kolmogorov complexity. Several late temporal mappings were also misindexed.

The inventory reports these distinctions. It does not delete the inherited research, convert every analogy into executable science, or declare all untouched functions wrong. The active instrument is tested separately from the inherited research inventory.

## Percentages need definitions, not blanket prohibition

The previous narrative implied that every percentage needs a population dataset. That is too broad. The fraction of supported landmarks below an explicitly declared error threshold is a directly computable empirical percentage. It is different from a reference-cohort percentile and different again from a probability of resemblance or identity. Wave 3 keeps all three meanings separate.

## Validation boundary

Analytic/synthetic tests establish the tested numerical and software invariants. They do not establish real-landmark accuracy. The MediaPipe adapter is executable API code with fixture-based contracts, but live weights could not be obtained in this environment. The release names that skip and contains no fabricated live-detector output.

A real-frame coarse ROI locator and video sampler were run. Neither supplies full anatomical observations. The actual recording was inspected, but no new human-face similarity score is claimed from those mixed screen-recording frames.
