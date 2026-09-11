# 05_image-analyze: Python development wave 01

> mathematics into a **single integrated Python system**, rather than translating the equations into a directory of unrelated utility functions.

[Download FACEMATH Wave 01](sandbox:/mnt/data/facemath_wave_01.zip)

The complete source tree: [FACEMATH source folder](../facemath_wave)

## Architectural decision

The central object is not a similarity score. It is an **evidence ledger**:

```text
INPUT
│
├── raster frames
├── landmarks
├── optional 3D surface
├── optional temporal sequence
└── optional neural telemetry
        │
        ▼
CANONICAL COORDINATE SYSTEM
        │
        ├───────────────────────────────────────────┐
        │                                           │
        ▼                                           ▼
SIGNAL GEOMETRY                              LANDMARK GEOMETRY
gradients                                    Procrustes
phase                                        intrinsic coordinates
structure tensors                            distance matrices
multiscale pyramids                          angles
scattering                                   symmetry topology
residual fields                              metric distortion
        │                                           │
        ├──────────────┐                ┌────────────┤
        ▼              ▼                ▼            ▼
DEFORMATION        TOPOLOGY         RELATIONAL     STATISTICS
optical flow       persistence      graphs         covariance
Jacobian           GW distortion    spectra        Mahalanobis
strain                                             PCA shape model
TPS                                                population baseline
bending
        │              │                │            │
        └──────────────┴────────────────┴────────────┘
                              │
                              ▼
                    EVIDENCE LEDGER
                              │
                  population calibration
                              │
                              ▼
                 DOMAIN-BALANCED FUSION
```

Every result carries:

```python
MetricResult(
    name=...,
    value=...,
    unit=...,
    direction=...,
    domain=...,
    description=...,
    provenance=...,
    calibrated_percentile=...,
    uncertainty=...,
)
```

This means we **cannot accidentally convert “the model thinks this looks highly similar” into “87% measured similarity.”**

A percentage only becomes legitimate when a metric is calibrated against an explicit comparison population.

---

## Mathematical coverage

I also made the conversation mathematics itself an enforceable development requirement.

`coverage.py` maps **every section from 20 through 180** to its Python implementation or, where actual model internals are required, to the explicit telemetry adapter boundary.

```python
SECTION_TO_IMPLEMENTATION = {
    20: "symmetry.image_reflection_error",
    21: "symmetry.paired_symmetry_signature",
    ...
    90: "deformation.fit_thin_plate_spline",
    91: "deformation.tps_bending_proxy",
    ...
    108: "surface3d.heat_kernel_signature",
    109: "surface3d.wave_kernel_signature",
    ...
    120: "perturbation.shapley_values",
    ...
    130: "representation.linear_cka",
    131: "representation.rsa",
    ...
    150: "temporal.slow_feature_analysis",
    ...
    169: "perturbation.fisher_information",
    ...
    180: "fusion.calibrated_consensus",
}
```

And the test suite explicitly requires:

```python
assert set(SECTION_TO_IMPLEMENTATION) == set(range(20, 181))
```

So mathematical concepts cannot silently disappear during later development.

---

## Major implemented subsystems

### Geometry

Implemented directly:

```python
generalized_procrustes()
similarity_transform()
intrinsic_coordinates()
pairwise_distance_matrix()
distance_matrix_error()
angle_signature()
epsilon_isometry_distortion()
rigid_residuals()
relational_graph()
graph_laplacian()
graph_spectrum()
spectral_graph_distance()
```

This gives us the foundation for:

$$
D_P
=
\|\hat X-\hat YR^*\|_F
$$

and:

$$
\delta_{ij}
=
d_A(i,j)-d_B(i,j)
$$

instead of pretending landmark correspondence is one scalar.

---

### Symmetry topology

Implemented:

```python
image_reflection_error()
paired_symmetry_signature()
symmetry_topology_distance()
midline_deviation_signature()
midline_signature_distance()
```

This specifically preserves the distinction we uncovered between:

> both faces being individually symmetrical

and:

> both faces having **similar patterns of asymmetry**.

---

### Deformation geometry

Implemented:

```python
optical_flow_farneback()
flow_jacobian()
strain_tensor()
strain_energy()
smooth_deformation_energy()
fit_thin_plate_spline()
tps_grid()
tps_bending_proxy()
local_affine_strain()
```

The engine therefore distinguishes:

$$
\text{translation}
$$

from:

$$
\text{actual deformation}.
$$

That was a critical design requirement.

A five-pixel rigid shift should not be treated as equivalent to:

```text
eye inward
nose stretched
mouth rotated
jaw widened
```

The strain tensor catches that distinction.

---

### Shape morphology

Implemented:

```python
resample_closed_contour()
contour_curvature()
curvature_distance()
fourier_descriptors()
fourier_descriptor_distance()
```

This gives us separate channels for:

```text
POSITION AGREEMENT

versus

SHAPE AGREEMENT
```

So an eye can occupy nearly the exact same coordinates while still having a distinctly different contour.

---

### Signal-level geometry

Implemented:

```python
difference_field()
gradient_field()
structure_tensor()
phase_correlation()
fourier_phase_distance()
laplacian_pyramid_distance()
gabor_scattering_signature()
scattering_distance()
residual_entropy()
```

This goes underneath landmarks.

It gives us:

$$
\nabla I
$$

structure tensors:

$$
J=
\begin{bmatrix}
I_x^2&I_xI_y\\
I_xI_y&I_y^2
\end{bmatrix}
$$

and multiscale geometry.

That means FACEMATH can observe resemblance even when no landmark detector has explicitly said:

> this is the nose.

---

### Topology and relational geometry

Implemented:

```python
zero_dim_persistence()
persistence_distance_0d()
relational_assignment()
gromov_relational_distortion()
```

The Gromov-style path compares:

$$
d_A(i,k)
$$

against:

$$
d_B(j,l)
$$

rather than merely comparing raw coordinates.

This is important because the claim we're ultimately examining is strongly **relational**:

> Do the internal structures of these two faces obey nearly the same geometry?

---

### Population/statistical shape analysis

Implemented:

```python
fit_shape_model()
mahalanobis_distance()
empirical_similarity_percentile()
nearest_neighbor_tail()
covariance_whiten()
```

This creates the eventual path from:

> these measurements are close

to the far more meaningful:

> these measurements are unusually close relative to unrelated faces.

That requires an actual baseline dataset.

No baseline:

```text
NO percentile.
```

That rule is encoded into the architecture.

---

### Temporal/morph mathematics

Implemented:

```python
trajectory_velocity()
trajectory_acceleration()
path_energy()
path_curvature_energy()
slow_feature_analysis()
robust_pca_pcp()
dynamic_mode_decomposition()
canonical_correlation()
```

So the crossfade itself becomes analytical data.

We can separate:

$$
X=L+S
$$

with Robust PCA, search for slowly varying structural components with SFA, and measure whether the A→B trajectory has unusually low structural curvature.

This is one of the most important parts of the system because **your video is not merely two images**.

The transition contains information.

---

### Neural/model representation analysis

Implemented the analytical side:

```python
cosine_similarity()
linear_cka()
rdm()
rsa()
svcca()
attention_laplacian()
attention_spectral_distance()
logit_contrast()
```

And critically:

```python
class ModelAdapter:
    def capture(...):
        ...

    def intervene(...):
        ...
```

I deliberately **did not invent an internal-model implementation**.

Actual activations, attention, logits, residual streams, etc. must enter through this boundary from real telemetry.

Once supplied, the rest of FACEMATH already knows how to analyze it.

---

### Causal perturbation

Implemented:

```python
occlusion_effect()
finite_difference_gradient()
finite_difference_hessian()
shapley_values()
pair_interaction()
fisher_information()
```

Plus real PyTorch attribution:

```python
input_gradient()
integrated_gradients()
```

This gives us the transition from:

> “I think eyes mattered”

to:

$$
\Delta S_{\text{eyes}}
=
S(I)-S(I\setminus\text{eyes})
$$

and eventually:

$$
\frac{\partial^2R}
{\partial \delta_E\,\partial \delta_N}
$$

for eye–nose interaction.

---

### 3D shape mathematics

Implemented:

```python
vertex_normals()
cotangent_laplacian()
laplace_beltrami_eigensystem()
heat_kernel_signature()
wave_kernel_signature()
spectral_shape_distance()
functional_map_from_descriptors()
```

So the architecture does not dead-end at 2D landmarks.

It has a proper route into:

$$
\Delta\phi_i=\lambda_i\phi_i
$$

HKS:

$$
HKS_t(x)
=
\sum_i
e^{-\lambda_i t}\phi_i(x)^2
$$

and functional maps.

---

## Validation

The initial suite passes:

```text
7 passed in 1.22s
```

Including explicit tests for:

* similarity-transform recovery,
* Procrustes invariance,
* epsilon-isometry,
* translation-invariant strain,
* Slow Feature Analysis,
* Robust PCA reconstruction,
* complete 20–180 mathematics coverage.

---

## I also ran the engine against your actual MP4

Without inventing landmarks.

I used the actual first and final frames:

```text
0.0 s
11.7 s
```

and ran the signal/deformation layers.

The resulting raw evidence is here:

[Video endpoint evidence JSON](sandbox:/mnt/data/facemath_wave/examples/video_endpoint_evidence.json)

Current raw values include:

```text
residual_rms
    0.5313844229573208

residual_entropy
    4.558912960237616 bits

structure_tensor_distance
    0.32154113592869277

phase_distance
    1.6583405004249796

multiscale_laplacian
    [
      0.014504530763932409,
      0.019398588484512615,
      0.027302443122796883,
      0.03620358791734379,
      0.05004349265329073
    ]

scattering_distance
    0.0024205977996883724

flow_mean_magnitude
    43.94156948599634 px

strain_energy
    41.56711833560088

smooth_deformation_energy
    5671.269998241147
```

**These are deliberately not interpreted as facial-similarity scores yet.**

The current endpoint run lacks anatomical masks, landmarks, region weighting, and population calibration. Therefore these are **raw signal/deformation measurements**, nothing more.

That distinction is enforced rather than left to my discretion.

---

## The development standard going forward

Wave 01 establishes the substrate.

The next wave should not be “add more formulas.” The formulas already exist.

The next serious engineering work is to build the **facial correspondence field** that causes these systems to operate as one instrument:

```text
FACE REGION MODEL
        │
        ├── canonical anchor graph
        ├── dense correspondence field
        ├── anatomical masks
        ├── contour families
        ├── symmetry pairs
        └── central/peripheral hierarchy
                 │
                 ▼
        SHARED GEOMETRIC STATE
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
      signal   shape   deformation
        │        │        │
        └────────┼────────┘
                 ▼
          relational topology
                 │
                 ▼
        population calibration
```

That is the difference between **a mathematics repository** and the beginnings of a genuinely novel **facial structural analysis instrument**.

[The complete Wave 01 project](../packages/facemath_wave_01.zip)
