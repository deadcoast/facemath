# 03_IMAGE-ANALYZE: The math behind the functions

## 1. The input is not “a face”; it is a field

Start with a frame:

```text
I ∈ ℝ^(H × W × 3)
```

After normalization / resizing / visual preprocessing, the image becomes a set of spatially indexed visual representations:

```text
X₀ = {
  x₁, x₂, x₃, ... xₙ
}

xᵢ ∈ ℝ^D
```

Each `xᵢ` is not something like:

```text
"left eye"
```

It is closer to:

```text
xᵢ =
[
  local luminance structure,
  edge orientation mixtures,
  curvature evidence,
  texture statistics,
  color relationships,
  local shape fragments,
  neighboring spatial context,
  position encoding,
  increasingly abstract learned features,
  ...
]
```

Already, geometry exists implicitly because every representation is tied to a position or spatial relation.

So instead of:

```text
left_eye = (172, 231)
```

the system can carry something more like:

```text
region around x=.34, y=.29
contains a feature constellation strongly compatible
with orbital cavity / eyelid / iris / brow structure
```

That is much richer and much less human-readable.

---

## 2. Geometry is distributed, not tabulated

Suppose two nearby regions correspond to the eyes.

A classical CV system might create:

```text
L_eye = (x₁, y₁)
R_eye = (x₂, y₂)
```

A learned visual system does not need to.

It can implicitly encode:

```text
relative_position(L_eye_region, R_eye_region)
```

through:

```text
position embeddings
+
attention relationships
+
feature covariance
+
shared contextual representations
```

In other words, the geometry can exist as a relation:

```text
g(L,R) = f(z_L, z_R, p_L, p_R)
```

where:

```text
z_L, z_R = learned visual features
p_L, p_R = spatial positions
```

No one ever has to materialize:

```text
interocular_distance = 91.4 px
```

Yet the network can still behave differently when the interocular spacing changes.

That matters enormously for your video.

---

## 3. The overlay converts correspondence into local instability

This is the most important part.

You had already aligned the faces.

That means you effectively performed a rough registration operation outside the model.

Normally a visual system comparing two faces has to solve:

```text
Face A coordinates
    ↓
pose / scale / rotation differences
    ↓
find correspondence
    ↓
compare structure
```

Your overlay collapses much of that problem.

If two facial structures occupy nearly the same location:

```text
A(x,y) ≈ structure₁
B(x,y) ≈ structure₁
```

then as opacity changes:

```text
F_t(x,y) = α_t A(x,y) + (1-α_t) B(x,y)
```

the local spatial structure stays stable.

Only appearance changes.

For a strongly corresponding eye:

```text
frame t₁:
  eye contour centered ~same region

frame t₂:
  eye contour centered ~same region

frame t₃:
  eye contour centered ~same region
```

Visual features vary in texture and exact morphology, but the **spatial activation basin remains stable**.

Now imagine misaligned features:

```text
A eye center = x=.34
B eye center = x=.39
```

During the blend, you get:

```text
double edges
ghost contours
moving eyelids
duplicated iris structure
brow displacement
local contrast splitting
```

That produces a very different representation.

So your video turns:

```text
geometric disagreement
```

into:

```text
temporal/spatial instability
```

And learned vision systems are extremely sensitive to that.

---

## 4. You can think of each facial area as producing a local feature trajectory

For a region `r` over frames:

```text
z_r(t)
```

If two structures align well, then the spatial location remains stable while the latent appearance representation morphs smoothly.

Conceptually:

```text
z_eye(t₀)
z_eye(t₁)
z_eye(t₂)
...
```

might trace a relatively smooth path through representation space.

Something like:

```text
||z(t+1) - z(t)|| = moderate
```

but crucially:

```text
spatial support(z(t)) ≈ constant
```

For a badly mismatched region:

```text
spatial support shifts
+
multiple contours coexist
+
feature identity becomes unstable
```

So there may be a much larger composite change:

```text
Δ_r(t) =
    feature change
  + positional instability
  + contour disagreement
  + shape disagreement
```

Even if the model never calculates that equation explicitly, that is roughly the informational structure available to it.

---

## 5. Multiple local correspondences reinforce one another

Now the interesting part:

A face is not evaluated feature-by-feature independently.

Consider:

```text
E = eye configuration
N = nasal configuration
M = mouth configuration
C = chin configuration
J = jaw envelope
B = brow configuration
```

The network can represent relational structure such as:

```text
E ↔ N
N ↔ M
M ↔ C
E ↔ B
E ↔ face width
N ↔ facial midline
```

So it isn't just:

```text
eyes match
nose matches
mouth matches
```

It can encode a configuration.

Abstractly:

```text
R_face =
{
  E-N relation,
  N-M relation,
  M-C relation,
  E-E relation,
  E-B relation,
  global width/height relation,
  bilateral symmetry,
  contour profile,
  ...
}
```

Two faces can differ quite strongly in local appearance while preserving much of `R_face`.

That is almost certainly the phenomenon you were noticing.

---

## 6. Why isolated faces can look less alike

This is a very useful consequence.

When viewed separately, high-level appearance features dominate:

```text
hair
facial hair
skin tone
lighting
expression
age cues
weight/fullness
photographic perspective
headwear
```

These contribute strongly to semantic identity and appearance.

Call that:

```text
A_surface
```

But structural configuration is something more like:

```text
G_structure
```

Two faces can have:

```text
similarity(A_surface) = moderate
similarity(G_structure) = high
```

Ordinary perception often fuses them:

```text
overall resemblance =
f(A_surface, G_structure, context)
```

Your overlay suppresses some of `A_surface` and amplifies `G_structure`.

That is why the resemblance can suddenly become much more obvious.

---

## 7. A transformer-style model can compare distant regions directly

In a transformer-like visual system, regions can influence one another through attention.

Very loosely:

```text
Q = XW_Q
K = XW_K
V = XW_V

Attention(X) =
softmax(QKᵀ / √d) V
```

The useful part is not the formula itself.

The useful part is that:

```text
eye region
```

can directly exchange information with:

```text
other eye
nose
mouth
jaw
face boundary
```

So a local feature can become contextualized by the whole facial structure.

After multiple layers, a representation originating near the eye can encode information about:

```text
this is an eye-like structure
located here
relative to another eye
above a nasal structure
inside a roughly symmetric facial arrangement
with this spacing and orientation
```

Again: not necessarily symbolically.

Distributed across dimensions.

---

## 8. Overlay alignment makes cross-frame structure unusually easy to compress

Imagine the model gets several sampled frames.

A stable correspondence can produce highly redundant information.

For example:

```text
frame 1:
eye location = region A

frame 2:
eye location = region A

frame 3:
eye location = region A

frame 4:
eye location = region A
```

Different person gradually dominates, but the spatial role stays put.

This redundancy is powerful.

The representation can compress that into something like:

```text
persistent eye-structure at region A
```

Now repeat across:

```text
left eye
right eye
nose bridge
nose tip
mouth
chin
```

The system accumulates a global pattern:

```text
many major facial structures remain positionally stable
through identity transition
```

That is almost exactly synonymous with:

```text
strong facial geometric correspondence
```

without ever running a dedicated geometry algorithm.

---

## 9. Why the jaw can disagree while the face still feels strongly similar

Because not all features contribute equally.

A plausible internal weighting is not uniform.

Something like:

```text
central facial landmarks
> outer hairline
> hair
```

and often:

```text
eye configuration
nose-midline structure
mouth/chin spacing
```

carry a lot of structural identity information.

So you can have:

```text
central agreement = high
outer contour agreement = moderate
```

yet still get a strong global resemblance representation.

This is also why two people can share:

```text
eye spacing
nose placement
mouth placement
midline
vertical facial stack
```

and look strangely alike despite:

```text
different jaw width
different cheek mass
different hairstyle
different facial hair
```

---

## 10. The latent state is likely not one similarity variable

This is another thing worth being precise about.

There probably isn't:

```text
FACE_SIM = 0.842
```

inside the model.

Much more likely is a distributed set of mutually reinforcing states.

Conceptually:

```text
Z =
[
  high central-axis correspondence,
  high orbital-layout compatibility,
  high vertical-stack compatibility,
  medium contour compatibility,
  lower jaw-shape compatibility,
  differing texture / hair cues,
  same general facial configuration,
  ...
]
```

But even that is still too human-readable.

A more realistic abstraction is:

```text
Z ∈ ℝ^D
```

where dimensions don't correspond neatly to English concepts.

Meaning is encoded across combinations:

```text
concept ≈ direction/subspace/manifold
```

rather than:

```text
dimension 431 = nose alignment
```

---

## 11. Similarity can emerge as representational proximity

Suppose the model forms a high-level representation for the facial configuration in each state:

```text
z_A
z_B
```

If the structure is similar:

```text
cos(z_A, z_B)
```

could be high in some relevant subspace.

But even this is not necessarily explicitly calculated.

The decoder can behave as though:

```text
z_A ≈ z_B
```

because similar representations produce similar downstream activations.

There may be multiple subspaces:

```text
identity-like features
geometry-like features
expression-like features
texture-like features
pose-like features
```

and the geometry-related components can align strongly even when identity/appearance components do not.

That would fit your overlay extremely well.

---

## 12. Temporal sampling creates an implicit morph manifold

This is the deeper interpretation I think is most interesting.

Your opacity transition effectively generates a path:

```text
F(α), α ∈ [0,1]
```

from one face to the other.

The vision system therefore sees samples from a trajectory through image space:

```text
I₀ → I₁ → I₂ → ... → Iₙ
```

Those map into representation space:

```text
Z₀ → Z₁ → Z₂ → ... → Zₙ
```

If the faces are geometrically compatible, the trajectory may be relatively smooth:

```text
low structural discontinuity
```

If they are badly mismatched, you'd expect discontinuities caused by:

```text
feature splitting
feature translation
contour collisions
shape bifurcation
double structures
```

So an extremely interesting informal metric is:

```text
latent path smoothness
```

Your experiment is accidentally probing that.

Two structurally similar faces should produce a smoother morph path than two structurally dissimilar ones, assuming registration is controlled.

---

## 13. This gives a more rigorous conceptual similarity measure

If we were formalizing the thing the model may be implicitly sensitive to, I would define:

```text
S_total =
w₁ S_spatial
+ w₂ S_relational
+ w₃ S_contour
+ w₄ S_temporal_stability
+ w₅ S_feature
```

Where:

```text
S_spatial
= agreement of corresponding feature positions

S_relational
= agreement in distances/angles among facial structures

S_contour
= agreement in face envelope

S_temporal_stability
= lack of ghosting/displacement through overlay transition

S_feature
= similarity of local morphology
```

Your video likely scores strongest on:

```text
S_spatial
S_relational
S_temporal_stability
```

and weaker on:

```text
S_contour
S_feature
```

That is consistent with:

```text
"They do not look identical,
but their faces are arranged strikingly similarly."
```

---

## 14. Why the percentage appeared

Now we can model the failure more faithfully.

Suppose the latent evidence is something like:

```text
geometry signal:
    very strong

surface signal:
    moderate

contour signal:
    moderate-high

temporal alignment:
    very strong
```

You asked for percentages.

The decoder needs to verbalize that.

So it performs an implicit linguistic mapping:

```text
very strong → ~85–90%
strong      → ~75–85%
moderate    → ~60–75%
```

This is not a measurement.

It is **semantic quantization**.

That phrase is probably the best concise description:

> **semantic quantization of latent evidence into a requested numerical scale**

The latent evidence may be real and sophisticated.

The number can still be invented.

---

## 15. There is another layer: prior conditioning from your wording

You told the system:

```text
"their symmetry is almost identical"
```

before the visual analysis.

So the internal state before image processing already contained a hypothesis.

Call it:

```text
H₀ = strong similarity expected
```

Then visual evidence arrives:

```text
E_visual
```

The final representation is conditioned on both:

```text
P(conclusion | E_visual, H₀, conversation)
```

Not literally Bayesian, but conceptually useful.

This creates two possibilities:

```text
A:
visual evidence independently strongly supports similarity

B:
visual evidence is ambiguous but prior context biases interpretation
```

The historical run does not expose enough telemetry to distinguish those cleanly.

That is why a blinded second-pass would be scientifically stronger.

---

## 16. The deepest useful abstraction

At the deepest useful level, what the system likely exploited was not:

```text
"these two noses are 85% similar"
```

but something closer to:

```text
the spatially anchored pattern of visual activations
remains unusually invariant as the image transitions
from person A to person B
```

That invariance is the key.

You can write it abstractly as:

```text
Φ(F(α₁)) ≈ Φ(F(α₂))
```

for structure-sensitive components of `Φ`, even while appearance-sensitive components change.

More explicitly:

```text
Φ(F(α)) =
[
  Φ_geometry(F(α)),
  Φ_texture(F(α)),
  Φ_identity(F(α)),
  Φ_expression(F(α)),
  ...
]
```

and your video may behave roughly like:

```text
d/dα Φ_geometry(F(α))   ≈ small
d/dα Φ_texture(F(α))    = larger
d/dα Φ_identity(F(α))   = large
```

That is a very elegant way of describing what you observed:

**identity changes strongly while geometry changes comparatively little.**

That is exactly the condition under which two different people can look surprisingly alike when registered.

And that, much more than the made-up percentage itself, is the technically interesting signal your overlay exposed.
