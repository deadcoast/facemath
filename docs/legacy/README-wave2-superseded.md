# FACEMATH — Development Wave 1

FACEMATH is an integrated evidence engine for comparing **facial geometry and structural correspondence**, not identity.

The system is deliberately designed against a failure mode exposed during development: **a qualitative model judgment must never masquerade as a measured percentage**. Every computation writes a typed `MetricResult` into one `EvidenceLedger`. A result only receives a percentile-like interpretation after calibration against an explicit baseline population.

## Architecture

```text
input images / video / optional landmarks / optional 3D mesh / optional model telemetry
                                  │
                                  ▼
                         canonical coordinate frame
                                  │
          ┌───────────────────────┼──────────────────────────┐
          │                       │                          │
          ▼                       ▼                          ▼
      signal field          landmark geometry          model telemetry
  gradients / phase /      Procrustes / graphs /      activations / attention /
  multiscale / residual    symmetry / topology         logits / interventions
          │                       │                          │
          ├──────────────┬────────┴───────────┬─────────────┤
          ▼              ▼                    ▼             ▼
      deformation      statistics          temporal      3D surface
      flow / strain    population model     trajectories  HKS/WKS/LB
      TPS / bending    calibration          SFA/RPCA/DMD  spectra
          │              │                    │             │
          └──────────────┴──────────────┬─────┴─────────────┘
                                       ▼
                              typed evidence ledger
                                       │
                              calibrated domain fusion
```

### Why this is one system

All analyzers consume the same `CanonicalPair` and write to the same evidence ledger. Registration, provenance, units, similarity direction, calibration, and aggregation are centralized. Modules do not pass ad-hoc dictionaries between one another.

The `coverage.py` registry maps **every conversation mathematics section 20 through 180** to an executable primitive or, where the mathematics requires inaccessible model internals, to the explicit `ModelAdapter` boundary. The coverage test fails if any section is absent.

## Implemented mathematics

- reflection symmetry, paired asymmetry topology, midline deviation
- intrinsic eye-centered coordinates, normalized distance matrices, angle signatures
- Procrustes analysis, rigid/similarity transforms, epsilon-isometry distortion
- landmark graphs, graph Laplacians and spectra
- dense optical flow, flow Jacobians, strain tensors, smooth deformation energy
- thin-plate-spline warp and bending-energy proxy
- contour curvature and Fourier shape descriptors
- 0D Vietoris–Rips persistence and GW-style relational distortion
- gradient fields, structure tensors, phase correlation and Fourier phase distance
- Laplacian pyramids, Gabor scattering-like deformation-stable descriptors
- residual entropy and compression/MDL proxies
- covariance whitening, Mahalanobis distance, PCA statistical shape models, population percentile calibration
- trajectory velocity/acceleration, path energy/curvature, Slow Feature Analysis, Robust PCA, Dynamic Mode Decomposition, CCA
- representational cosine similarity, CKA, RSA, SVCCA, attention-graph spectra, logit contrasts
- finite-difference gradients/Hessians, Shapley values, interaction terms, Fisher information
- PyTorch input gradients and Integrated Gradients
- 3D cotangent Laplace–Beltrami operator, spectra, HKS, WKS, functional maps
- evidence provenance and domain-balanced calibrated fusion

## Explicit boundary

Internal neural activations are **never fabricated**. `facemath.representation.ModelAdapter` is the contract for supplying real telemetry from a model implementation. The rest of FACEMATH can analyze those tensors once an adapter captures them.

## Run tests

```bash
cd facemath_wave
python -m pytest -q
```

## Compare two aligned images

```bash
facemath compare face_a.png face_b.png --out evidence.json
```

If corresponding landmark JSON is available:

```bash
facemath compare face_a.png face_b.png \
  --landmarks-a a.json \
  --landmarks-b b.json \
  --out evidence.json
```

Landmark files:

```json
{
  "names": ["left_eye", "right_eye", "nose", "mouth", "chin"],
  "xy": [[120, 100], [220, 101], [170, 160], [171, 215], [170, 270]]
}
```

## Development direction

Wave 1 establishes the mathematical substrate and provenance architecture. The next implementation wave should attach a real landmark/segmentation frontend, add mask-aware regional analysis, produce deformation and distortion field artifacts, and build population calibration datasets. The engine is intentionally ready for those additions without changing its evidence contract.


## Wave 2 — Shared Facial State

Wave 2 changes the unit of computation. Metrics no longer independently decide registration,
masking, or correspondence. `FacialGeometryEngine` constructs one immutable `FacialState`:

- robust landmark registration with RANSAC + confidence-weighted refinement;
- registration diagnostics and explicit support states (`valid`, `degraded`, `insufficient`);
- detector-agnostic anatomical region hierarchy built from named landmarks;
- face support mask shared by all analyzers;
- dense A→B correspondence with forward/backward cycle-consistency confidence;
- region-level displacement and confidence evidence;
- quality gates that can refuse substantive analysis when geometric support is insufficient;
- temporal sequence analysis for morph/path smoothness, SFA, RPCA and structural change;
- cohort-fingerprinted calibration with optional bootstrap uncertainty;
- report layer that explicitly refuses percentage-style consensus until baselines exist.

### Design invariant

No analyzer may silently establish its own coordinate system. All Wave-2 geometry consumes the
same `FacialState`, and every calibration result is traceable to a named/fingerprinted baseline cohort.

### CLI

```bash
facemath compare a.png b.png --landmarks-a a.json --landmarks-b b.json --out evidence.json
facemath sequence transition.mp4 --frames 48 --out sequence.json
```

Landmark JSON accepts `xy`, optional `names`, and optional per-landmark `confidence`.
