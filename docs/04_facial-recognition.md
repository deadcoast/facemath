# Image Analyze Workflow & Logic

> This document describes the **computational anatomy that could turn your particular registered face-transition into a resemblance judgment**

## 0. Forget “faces” entirely

At the bottom, neither person exists.

There is only a sampled signal:

$$
I_t(x,y,c)
$$

where \(t\) is frame/time, \(x,y\) are spatial coordinates, and \(c\) is channel.

Your overlay creates a particularly interesting signal because approximately:

$$
I_t = \alpha_t A +(1-\alpha_t)B
$$

Thus:

$$
\frac{\partial I}{\partial \alpha}=A-B
$$

That difference field is already enormously informative.

If corresponding anatomy occupies the same coordinates, much of \(A-B\) consists of **appearance residuals**.

If anatomy is displaced, \(A-B\) contains characteristic paired positive/negative structures around edges.

Conceptually:

```text
GOOD REGISTRATION

A eye:       ───╮ ╭───
B eye:       ───╮ ╭───
difference:     small


BAD REGISTRATION

A eye:       ───╮ ╭───
B eye:          ───╮ ╭───

difference:  ++++----++++----
```

That second pattern matters enormously.

You have accidentally constructed something resembling a **visual residual experiment**.

---

# 1. Displacement leaves a mathematical fingerprint

Suppose feature \(B\) is just feature \(A\) translated by a small vector \(\delta\):

$$
B(x)\approx A(x-\delta)
$$

Taylor-expand:

$$
A(x-\delta)\approx A(x)-\nabla A(x)\cdot\delta
$$

Therefore:

$$
A(x)-B(x)\approx\nabla A(x)\cdot\delta
$$

This is fucking important for your experiment.

Small geometric displacement doesn't merely produce arbitrary visual difference.

It produces difference proportional to:

$$
\boxed{\nabla I\cdot\delta}
$$

Meaning **edges amplify geometric misalignment**.

Eyes, nostrils, lips, jaw boundaries, brows and nose edges contain strong gradients.

Therefore even tiny displacement produces conspicuous residual structure.

When your overlay *doesn't* explode into those moving/doubled edge structures, that's meaningful evidence that \(\delta\) is relatively small for those structures.

This is considerably more fundamental than “the eyes seem lined up.”

---

# 2. Your eyes are effectively performing optical-flow detection

The classical brightness-constancy assumption is:

$$
I(x,y,t)=I(x+\Delta x,y+\Delta y,t+\Delta t)
$$

Differentiating gives the optical-flow constraint:

$$
I_xu+I_yv+I_t=0
$$

Your video isn't genuine physical motion.

But perceptually, mismatched facial geometry **creates pseudo-motion**.

If A's mouth is here:

```text
────────╮  ╭────────
```

and B's is five pixels lower:

```text


────────╮  ╭────────
```

crossfading makes the mouth appear to migrate.

Thus the transition converts:

$$
\text{static geometric difference}
$$

into:

$$
\text{apparent temporal displacement}
$$

Your visual system—and a machine vision representation sensitive to temporal/spatial change—gets a much easier problem.

You're essentially turning **geometry into motion energy**.

---

# 3. The real primitive isn't “eye.”

Go lower.

A learned visual representation can initially discriminate structures like:

$$
\text{oriented contrast}
$$

$$
\text{curvature}
$$

$$
\text{junctions}
$$

$$
\text{frequency}
$$

$$
\text{texture}
$$

Imagine a bank of filters:

$$
r_k(x,y)=\sigma((K_k * I)(x,y))
$$

Different \(K_k\) respond preferentially to:

```text
──── horizontal edge

│
│ vertical edge

╱ diagonal edge

) curvature

>< converging structure

• compact contrast feature
```

An eye isn't initially:

> eye.

It's a **constellation of responses**.

---

# 4. Constel­lations become topology

Now something deeper happens.

Individual features have relationships:

```text
brow arc
   ↓

eye contour ─ iris contrast
   ↓
cheek transition
```

The topology matters more than individual pixels.

Represent a local region as a graph:

$$
G=(V,E)
$$

where nodes are feature structures and edges encode relative relationships.

For example:

```text
          BROW
       /        \
 LEFT-EYE —— RIGHT-EYE
      \          /
       \        /
         NOSE
        /    \
   NOSTRIL  NOSTRIL
        \    /
         MOUTH
           |
          CHIN
```

Two photographs can have wildly different pixels while preserving approximately the same graph.

That graph-like relational stability is much closer to **facial geometry** than raw appearance.

---

# 5. Now remove absolute scale

Suppose:

$$
d_{ee}=\|E_L-E_R\|
$$

Use interocular distance as a normalization unit.

Then geometry becomes dimensionless:

$$
\tilde d_{nm}=\frac{\|N-M\|}{d_{ee}}
$$

$$
\tilde d_{mc}=\frac{\|M-C\|}{d_{ee}}
$$

$$
\tilde w_{mouth}=\frac{w_{mouth}}{d_{ee}}
$$

etc.

Now two faces photographed at different resolutions can still share:

```text
eye spacing     = 1.000
eye→nose        ≈ 0.71
nose→mouth      ≈ 0.39
mouth→chin      ≈ 0.56
```

Again, I'm **not claiming those numbers for your faces**.

I'm showing the mathematical object underneath the phenomenon you're noticing.

The important entity is:

$$
\boxed{\text{ratios of relationships}}
$$

not pixel coordinates.

---

# 6. Now remove translation, rotation and scale altogether

This takes us into **shape analysis**.

Let facial landmarks form:

$$
X=
\begin{bmatrix}
x_1&y_1\\
x_2&y_2\\
\vdots&\vdots\\
x_n&y_n
\end{bmatrix}
$$

and another face:

$$
Y
$$

Center both:

$$
X_c=X-\bar X
$$

$$
Y_c=Y-\bar Y
$$

Normalize scale:

$$
\hat X=\frac{X_c}{\|X_c\|_F}
$$

$$
\hat Y=\frac{Y_c}{\|Y_c\|_F}
$$

Find the optimal rotation:

$$
R^*=\arg\min_R\|\hat X-\hat YR\|_F
$$

Then:

$$
D_P=\|\hat X-\hat YR^*\|_F
$$

is a **Procrustes distance**.

Now we're getting very close to formalizing your claim.

You manually performed part of this operation when you lined up the faces.

Your overlay is essentially a **human-created approximate registration** before comparison.

---

# 7. But even landmarks throw away information

Go deeper.

Faces are deformable surfaces.

Instead of:

$$
\{p_1,\ldots,p_n\}
$$

consider a deformation field:

$$
\phi:\mathbb{R}^2\rightarrow\mathbb{R}^2
$$

mapping face A onto face B:

$$
B(x)\approx A(\phi(x))
$$

Write:

$$
\phi(x)=x+u(x)
$$

where \(u(x)\) is the displacement field.

Now the question becomes:

> **How much deformation is necessary to turn A's geometry into B's?**

If they're structurally similar:

$$
\|u(x)\|\approx small
$$

over important facial regions.

Not everywhere.

Maybe:

```text
eyes       ██████████  tiny deformation
nose       █████████   small
mouth      ████████    small
chin       ███████     moderate
cheeks     █████       larger
jaw        ████        larger
```

Again: conceptual, not measured from your video.

---

# 8. Better still: measure deformation energy

Instead of asking only how far things move, ask how violently the mapping must distort space.

The Jacobian:

$$
J_\phi(x)=
\frac{\partial\phi}{\partial x}
$$

describes local deformation.

If:

$$
J_\phi\approx I
$$

then local geometry barely changes.

Large deviations imply:

```text
stretching
compression
shearing
rotation
local warping
```

One could define:

$$
E_{\text{deform}}
=
\int_\Omega
\|J_\phi(x)-I\|_F^2dx
$$

Now:

$$
\boxed{\text{low deformation energy}}
$$

means something considerably stronger than:

> “They kinda look alike.”

It means **one registered facial structure can be mapped onto the other without much geometric work.**

---

# 9. Your crossfade exposes deformation without actually calculating it

This is why your primitive experiment is clever.

Suppose the registered faces are:

$$
A(x),B(x)
$$

If geometry differs strongly, crossfade produces spatial interference:

```text
A contour ────────
B contour     ────────

blend:
────────░░────────
```

If geometry agrees:

```text
A contour ────────
B contour ────────

blend:
────────
```

The human visual system gets extremely sensitive evidence from the **ghosting width**.

Ghosting width is approximately related to displacement magnitude.

So conceptually:

$$
W_{\text{ghost}}\propto\|\delta\|
$$

for sufficiently simple local edges.

You can literally see geometry disagreement as thickness/doubling.

---

# 10. Frequency space gives another view

A face can be decomposed into spatial frequencies.

Low frequencies encode broad structure:

```text
head mass
cheek mass
orbital placement
large shadows
facial proportions
```

High frequencies encode:

```text
hair
wrinkles
pores
fine edges
facial hair
texture
```

Write:

$$
I=I_L+I_H
$$

Two people can have:

$$
I_H^A \not\approx I_H^B
$$

while:

$$
I_L^A\approx I_L^B
$$

This produces exactly the subjective phenomenon:

> separately, not obviously identical;

but after alignment:

> holy shit, the underlying faces occupy almost the same structure.

Your overlay helps perceptually discount some high-frequency identity noise.

---

# 11. Now enter invariance

A useful representation tries not to radically change because of nuisance transformations:

$$
T=\{
\text{lighting},
\text{minor pose},
\text{scale},
\text{texture variation},
\ldots
\}
$$

Ideally:

$$
\Phi(I)\approx\Phi(T(I))
$$

for irrelevant transformations.

But:

$$
\Phi(I)\neq\Phi(G(I))
$$

when meaningful geometry changes substantially.

Thus a geometry-sensitive representation can discard:

```text
different beard
different brightness
different hair
```

while retaining:

```text
eyes occupy this relationship
nose terminates here
mouth sits this far below
jaw expands this way
```

This is how machine representation can see correspondence that raw pixel similarity would completely miss.

---

# 12. Now stop thinking in Euclidean feature space

High-dimensional learned representations frequently organize information along **manifolds**.

Imagine all plausible human facial configurations occupying some lower-dimensional structure:

$$
\mathcal M_{\text{face}}\subset\mathbb{R}^D
$$

A particular face corresponds approximately to:

$$
z_A\in\mathcal M
$$

another:

$$
z_B\in\mathcal M
$$

The interesting question isn't necessarily:

$$
\|z_A-z_B\|
$$

but potentially the geodesic distance:

$$
d_\mathcal{M}(z_A,z_B)
$$

Two people may be far apart in texture/identity directions but close along geometry directions.

Locally:

$$
T_z\mathcal M
$$

can contain directions corresponding abstractly to variations like:

```text
jaw width
facial length
orbital spacing
nose projection
cheek fullness
expression
pose
illumination
```

Not necessarily as clean disentangled axes—but as distributed directions.

---

# 13. Decompose identity from geometry conceptually

Imagine:

$$
z=z_G+z_A+z_E+z_P+z_L+\epsilon
$$

where:

$$
z_G = geometry
$$

$$
z_A = appearance
$$

$$
z_E = expression
$$

$$
z_P = pose
$$

$$
z_L = lighting
$$

This isn't necessarily how the actual architecture explicitly factorizes things.

But it is a useful local approximation.

Your experiment is effectively asking:

$$
\boxed{d(z_G^A,z_G^B)\ ?}
$$

while casual face recognition asks something closer to:

$$
d(z^A,z^B)
$$

Those are **different questions**.

That explains your original disagreement with the other person surprisingly well.

They may be weighting:

$$
z_{\text{total}}
$$

while you noticed:

$$
z_G
$$

very strongly.

---

# 14. Now the overlay becomes a perturbation experiment

Your transition parameter \(\alpha\) defines:

$$
I(\alpha)=\alpha A+(1-\alpha)B
$$

Feed that through representation \(\Phi\):

$$
z(\alpha)=\Phi(I(\alpha))
$$

Now differentiate:

$$
\frac{dz}{d\alpha}
=
J_\Phi(I)
\frac{dI}{d\alpha}
$$

and since:

$$
\frac{dI}{d\alpha}=A-B
$$

we obtain:

$$
\boxed{
\frac{dz}{d\alpha}
=
J_\Phi(I)(A-B)
}
$$

This is much deeper.

The difference between the faces is being projected through the **Jacobian of the visual representation**.

Some pixel differences produce enormous representational changes.

Others barely matter.

So:

```text
different hair pixels
```

may project strongly into appearance dimensions but weakly into geometry dimensions.

Meanwhile:

```text
move eye 12 pixels
```

could strongly perturb geometry-sensitive dimensions.

Your video probes the representation's sensitivity to the direction:

$$
A-B
$$

through image space.

---

# 15. Curvature of that trajectory is even more interesting

First derivative:

$$
z'(\alpha)
$$

tells us how rapidly representation changes.

Second derivative:

$$
z''(\alpha)
$$

tells us whether the transition bends sharply through representation space.

A geometrically compatible morph could plausibly have:

$$
\|z''_G(\alpha)\|
$$

relatively small.

A grotesquely mismatched morph might create abrupt representational changes:

```text
one eye becomes two
nose edge bifurcates
mouth splits
jaw suddenly relocates
```

producing higher curvature.

So another theoretical resemblance statistic is:

$$
E_{\text{path}}
=
\int_0^1
\|z_G''(\alpha)\|^2d\alpha
$$

Lower:

> smoother structural transformation.

Higher:

> substantial structural incompatibility.

---

# 16. Your video contains an accidental counterfactual

This is perhaps the coolest part.

At intermediate alpha:

$$
I_{0.5}=0.5A+0.5B
$$

the image asks:

> **Can these two facial configurations coexist spatially without producing contradiction?**

If yes, intermediate images remain coherent.

If no:

```text
four nostril edges
double pupils
two lip boundaries
split jaw
duplicated eyebrows
```

appear.

So the intermediate frames act like **counterfactual compatibility probes**.

You aren't merely comparing endpoints.

You're observing whether the path *between them remains anatomically coherent*.

---

# 17. Now information theory

Let:

$$
G
$$

represent facial geometry and:

$$
V
$$

the observed visual signal.

A good representation retains information relevant to geometry:

$$
I(Z;G)
$$

while discarding nuisance information \(N\):

$$
I(Z;N)
$$

The desirable representation maximizes something conceptually like:

$$
I(Z;G)-\beta I(Z;N)
$$

Your aligned overlay makes this easier because alignment reduces nuisance variance.

In other words, you externally increased the **signal-to-nuisance ratio for geometry**.

That's why the experiment feels so revealing.

---

# 18. Compression itself becomes evidence

Suppose two facial configurations require separate descriptions:

```text
Face A geometry:
████████████████████████

Face B geometry:
████████████████████████
```

But if B can be represented as:

```text
Face A
+
slightly wider jaw
+
slightly different cheek contour
+
minor nose deformation
```

then the second geometry has a shorter conditional description.

Conceptually:

$$
K(B_G|A_G)
$$

is small.

This is Kolmogorov-complexity territory.

The stronger the structural resemblance, the less information required to describe B's geometry given A's.

That gives an extraordinarily deep definition of resemblance:

$$
\boxed{
\text{Faces resemble each other geometrically when one requires little additional structural information given the other.}
}
$$

Your overlay makes that compression perceptually obvious.

---

# 19. Predictive coding interpretation

Now suppose the system predicts the next overlay frame.

Given:

$$
I_t
$$

predict:

$$
\hat I_{t+1}
$$

Residual:

$$
e_{t+1}=I_{t+1}-\hat I_{t+1}
$$

For strongly aligned structures, prediction is easy:

```text
eye stays here
nose stays here
mouth stays here
appearance gradually changes
```

For poorly aligned structures:

```text
where does the eye migrate?
when does this contour disappear?
why are two mouth edges appearing?
```

Prediction residual rises.

So structural resemblance can manifest as:

$$
\boxed{\text{low unexpected spatial change}}
$$

during identity substitution.

Again, that's incredibly close to what your eyes noticed immediately.

---

## 20. Symmetry is not one variable

This is where your wording—**“their symmetry is almost identical”**—becomes more interesting than simple bilateral symmetry.

For a face \(I(x,y)\), define a candidate midline \(m\). Reflection across it is:

$$
R_m(x,y)=(2m-x,y)
$$

A primitive bilateral-symmetry error would be:

$$
E_{\text{sym}}(I)
=
\int_\Omega
\left|
I(x,y)-I(R_m(x,y))
\right|^2dxdy
$$

But that answers:

> How symmetrical is **this individual face**?

Your observation is subtly different:

> Do **the two faces organize their departures from symmetry in similar ways?**

Let each person's asymmetry field be:

$$
A_I(x,y)=I(x,y)-I(R_m(x,y))
$$

Then compare:

$$
A_A(x,y)
\quad\text{against}\quad
A_B(x,y)
$$

Two people could have similar global symmetry scores:

$$
E_{\text{sym}}(A)\approx E_{\text{sym}}(B)
$$

while possessing completely different asymmetry patterns.

Conversely, these two faces could exhibit:

$$
A_A(x,y)\approx A_B(x,y)
$$

meaning that not only their nominal landmarks but their **specific departures from an idealized symmetric template correspond**.

That is a much stronger resemblance signal.

---

# 21. Symmetry has a hierarchy

Break the face into paired structures:

$$
P=
\{
(E_L,E_R),
(B_L,B_R),
(C_L,C_R),
(J_L,J_R),
(N_L,N_R),
...
\}
$$

Each pair has several properties:

$$
s_i=
[
\Delta x,
\Delta y,
\Delta\theta,
\Delta\text{scale},
\Delta\text{curvature},
\Delta\text{appearance}
]
$$

So the facial symmetry signature becomes:

$$
S_{\text{face}}
=
[s_1,s_2,\ldots,s_n]
$$

Now two faces can be compared by:

$$
D_S
=
d(S_A,S_B)
$$

This captures something ordinary statements such as “both have symmetrical faces” completely miss.

For example:

```text
Person A                 Person B

left eye +1.5° high      left eye +1.7° high
right brow lower         right brow lower
nose deviates left       nose deviates left
mouth tilts +2°          mouth tilts +1.8°
jaw fuller on right      jaw fuller on right
```

If those relationships correspond, then the faces don't merely share **symmetry magnitude**.

They share a **symmetry topology**.

---

# 22. Midline stability is its own signal

Consider the vertical facial axis:

$$
M=
\{
\text{glabella},
\text{nasal bridge},
\text{nose tip},
\text{philtrum},
\text{mouth center},
\text{chin}
\}
$$

Fit a curve:

$$
\gamma(s)
$$

through these structures.

An idealized face might produce nearly:

$$
x(s)=c
$$

A real face has deviations:

$$
x(s)=c+\epsilon(s)
$$

The function:

$$
\epsilon(s)
$$

is effectively a **midline deviation signature**.

Comparing two people:

$$
D_M
=
\int
|\epsilon_A(s)-\epsilon_B(s)|^2ds
$$

asks whether their central facial structures wander away from the nominal axis similarly.

Your registered crossfade is unusually good at exposing this because mismatched centerlines produce:

```text
nose bridge drift
      ↓
nose-tip drift
      ↓
philtrum drift
      ↓
mouth-center drift
      ↓
chin drift
```

A stable centerline during the transition means several relationships are agreeing simultaneously.

---

# 23. Ratios create a facial coordinate system

Instead of treating landmarks independently, establish intrinsic coordinates.

Take:

$$
d_{io}=\|E_L-E_R\|
$$

as unit scale.

Set eye midpoint:

$$
O=\frac{E_L+E_R}{2}
$$

as origin.

Define:

$$
\hat x=
\frac{E_R-E_L}
{\|E_R-E_L\|}
$$

and perpendicular facial axis:

$$
\hat y
$$

Now every landmark can be represented intrinsically:

$$
p_i=
u_i\hat x+v_i\hat y
$$

with:

$$
(u_i,v_i)
$$

normalized by interocular distance.

The face becomes something like:

```text
                 (0, -a)
                   brow

(-.5, 0)  LEFT EYE ───── RIGHT EYE  (.5, 0)

                  (0,b)
                   nose

                  (0,c)
                  mouth

                  (0,d)
                   chin
```

Now photographic translation and much of scale cease to matter.

What remains is **configuration**.

---

# 24. Distances alone are insufficient

Given \(n\) landmarks, construct:

$$
D_{ij}=\|p_i-p_j\|
$$

This produces a facial distance matrix:

$$
D=
\begin{bmatrix}
0&d_{12}&d_{13}&\cdots\\
d_{21}&0&d_{23}&\cdots\\
d_{31}&d_{32}&0&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{bmatrix}
$$

Normalize it:

$$
\tilde D=\frac{D}{d_{io}}
$$

Now compare:

$$
E_D=
\|\tilde D_A-\tilde D_B\|_F
$$

This is interesting because it doesn't merely ask whether the nose occupies the same pixel.

It asks whether the nose has the same relationship to:

```text
left eye
right eye
mouth
chin
brows
jaw anchors
```

simultaneously.

A single landmark participates in many constraints.

Therefore matching one point accidentally is easy.

Matching an entire relational matrix becomes progressively less accidental.

---

# 25. Angles add another relational layer

Construct triples:

$$
\theta_{ijk}
=
\angle(p_i,p_j,p_k)
$$

Examples:

```text
left-eye → nose → right-eye

left-jaw → chin → right-jaw

eye-midpoint → nose → mouth

nose → mouth-corners
```

Now the structural signature contains:

$$
G=
\{D,\Theta\}
$$

where:

$$
\Theta=\{\theta_{ijk}\}
$$

Two faces can have slightly different absolute dimensions yet preserve remarkably similar angles and ratios.

This is why **shape correspondence can survive size differences**.

---

# 26. Now construct a facial graph

Let landmarks/regions be vertices:

$$
V=\{v_1,\ldots,v_n\}
$$

and meaningful anatomical relationships be edges:

$$
E=\{e_{ij}\}
$$

Each edge carries attributes:

$$
e_{ij}=
[
d_{ij},
\theta_{ij},
\text{curvature},
\text{relative orientation},
\text{appearance relation}
]
$$

The face becomes:

$$
G=(V,E)
$$

Comparison becomes **graph matching**.

The interesting quantity isn't merely:

$$
v_i^A\approx v_i^B
$$

but:

$$
G_A\approx G_B
$$

This gives a better formal interpretation of your observation:

> Many mutually constrained relationships survive the transition simultaneously.

---

# 27. Rigidity versus deformation

Suppose the two facial graphs are registered.

A rigid transformation has:

$$
p'_i=sRp_i+t
$$

Scale \(s\), rotation \(R\), translation \(t\).

If that maps A very close to B, then:

$$
p_i^B\approx sRp_i^A+t
$$

with little residual:

$$
r_i=p_i^B-(sRp_i^A+t)
$$

Then:

$$
E_R=\sum_i\|r_i\|^2
$$

is small.

That means the resemblance isn't dependent upon individually dragging:

```text
eye here
nose there
mouth elsewhere
jaw separately
```

The **whole structure travels together**.

That's much stronger evidence of common geometry than matching a few isolated features.

---

# 28. Non-rigid residuals reveal exactly where resemblance ends

After optimal rigid alignment, define:

$$
u_i=p_i^B-\hat p_i^A
$$

These displacement vectors produce a residual field:

```text
brow       ↘ 1px

eye        → 2px

nose       ↓ 1px

mouth      ↙ 3px

cheek      ← 8px

jaw        ← 11px
```

Again, illustrative numbers only.

The resulting vector field distinguishes:

$$
\text{shared global geometry}
$$

from:

$$
\text{local morphological differences}
$$

That distinction maps almost perfectly onto the original visual impression we were discussing.

---

# 29. Dense optical flow turns every pixel into a candidate landmark

Forget selected landmarks.

Estimate:

$$
u(x,y),v(x,y)
$$

for every facial pixel.

Now you obtain:

$$
F(x,y)=
\begin{bmatrix}
u(x,y)\\
v(x,y)
\end{bmatrix}
$$

A strongly geometrically corresponding face pair should contain large coherent regions where:

$$
\|F(x,y)\|
$$

is small after registration.

Regions requiring deformation show larger vectors.

Visualized:

```text
            FOREHEAD
          · · · · · ·
       · · · · · · · ·

      → ·   EYES   · ←
          · · · ·
            NOSE
           · ↓ ·
           MOUTH
          ↙ · · ↘

       ← ← CHEEKS → →

      ← ← ← JAW → → →
```

That gives you an actual **deformation anatomy** of resemblance.

---

# 30. But flow magnitude still misses strain

Suppose every point translates five pixels right.

Large flow:

$$
\|F\|=5
$$

but zero structural deformation.

So examine the gradient:

$$
\nabla F
$$

The symmetric component:

$$
\epsilon=
\frac{1}{2}
(\nabla F+\nabla F^T)
$$

is analogous to a strain tensor.

It captures:

```text
stretch
compression
shear
```

rather than mere translation.

Thus:

$$
E_{\text{strain}}
=
\int_\Omega
\|\epsilon(x,y)\|_F^2dxdy
$$

is arguably a better facial-geometry disagreement metric than raw displacement.

Two faces can move globally while requiring almost no strain.

That means:

> same shape, different registration.

Whereas high strain means:

> actual structural disagreement.

---

# 31. Local affine transformations expose anatomical compatibility

Around each region:

$$
x'\approx A_rx+b_r
$$

with:

$$
A_r=
\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

Decompose \(A_r\) into:

$$
A_r=R_rS_r
$$

via polar decomposition.

Here:

$$
R_r
$$

captures local rotation and:

$$
S_r
$$

captures local stretching.

If:

$$
S_r\approx I
$$

then that anatomical region maps almost rigidly.

If:

$$
S_r
$$

has eigenvalues far from 1, the region requires substantial deformation.

So the comparison can become:

```text
eyes       S ≈ I
nose       S ≈ I
mouth      S ≈ I
cheeks     S deviates
jaw        S deviates more
```

That is a mathematically meaningful version of:

> “central geometry matches much better than the outer morphology.”

---

# 32. Curvature gives us morphology independent of position

Take a contour:

$$
C(s)=(x(s),y(s))
$$

Its curvature is:

$$
\kappa(s)
=
\frac{x'y''-y'x''}
{(x'^2+y'^2)^{3/2}}
$$

Now compare:

$$
\kappa_A(s)
\quad\text{and}\quad
\kappa_B(s)
$$

for:

```text
upper eyelid
lower eyelid
brow
nasal wing
upper lip
lower lip
chin
jaw
```

This answers something position cannot:

> Even when the structures occupy the same place, **are they actually shaped similarly?**

Thus we can separate:

$$
S_{\text{position}}
$$

from:

$$
S_{\text{morphology}}
$$

That distinction was missing from my original percentage bullshit.

---

# 33. Shape descriptors go further

A contour can be represented using Fourier descriptors.

Represent:

$$
z(s)=x(s)+iy(s)
$$

Take Fourier transform:

$$
Z_k=\mathcal F\{z(s)\}
$$

Low-frequency coefficients capture broad contour shape.

High-frequency coefficients capture fine irregularity.

After normalizing translation, scale and rotation:

$$
\tilde Z_A
$$

and:

$$
\tilde Z_B
$$

can be compared directly.

This lets you say:

> Jaw placement agrees, but jaw **shape** doesn't.

or:

> Eye placement and eye contour both agree strongly.

Those are different claims.

---

# 34. Topology survives even more deformation

Go deeper again.

Instead of exact shape, consider topology:

```text
two eye basins
one nasal ridge
two nostril regions
one mouth aperture
outer facial envelope
```

Persistent homology can characterize structures across thresholds.

Construct a filtration:

$$
X_\tau
$$

and track birth/death of features.

Persistence diagram:

$$
PD(I)=\{(b_i,d_i)\}
$$

Then compare:

$$
d_B(PD_A,PD_B)
$$

using bottleneck distance, or Wasserstein distance.

This would be overkill for ordinary face comparison—but we're below Atlantis now.

The point is that **structural similarity exists at several mathematical levels simultaneously**:

```text
pixels
edges
contours
landmarks
relations
graphs
deformation fields
curvature
topology
latent representations
```

---

# 35. Phase contains geometry

Fourier-transform an image:

$$
F(u,v)=|F(u,v)|e^{i\phi(u,v)}
$$

Magnitude:

$$
|F|
$$

contains frequency strength.

Phase:

$$
\phi
$$

contains enormous amounts of spatial structural information.

Famously, image structure is strongly dependent on phase.

Two faces with similar spatial organization can exhibit stronger **phase correspondence** than raw RGB similarity suggests.

Phase correlation computes:

$$
R=
\frac{F_A\overline{F_B}}
{|F_A\overline{F_B}|}
$$

then:

$$
r=\mathcal F^{-1}(R)
$$

A sharp peak indicates translation correspondence.

With localized/windowed versions, you can estimate regional structural alignment.

So another way of describing your overlay is:

> You manually placed two images into approximately compatible phase geometry before visually comparing them.

---

# 36. Gradient orientation may matter more than intensity

For each pixel:

$$
\nabla I=
(I_x,I_y)
$$

Orientation:

$$
\theta=\operatorname{atan2}(I_y,I_x)
$$

Magnitude:

$$
m=\sqrt{I_x^2+I_y^2}
$$

Faces with different lighting can have quite different intensities but similar edge orientations.

Construct orientation histograms:

$$
H_r(\theta)
$$

for facial regions.

Then:

$$
d(H_A,H_B)
$$

captures structural similarity more robustly than RGB.

Your overlay visually exposes this because matching edges remain crisp while mismatched edges bifurcate.

---

# 37. Structural tensors expose local geometry

The image structure tensor is:

$$
J=
\begin{bmatrix}
I_x^2&I_xI_y\\
I_xI_y&I_y^2
\end{bmatrix}
$$

usually locally averaged.

Its eigenvalues/eigenvectors characterize:

```text
flat region
edge
corner
dominant orientation
anisotropy
```

Now compare:

$$
J_A(x,y)
$$

against:

$$
J_B(x,y)
$$

after registration.

If major facial structures correspond, many regions should have similar dominant orientations and anisotropy.

Again:

**geometry without ever saying “nose.”**

---

# 38. The face can become a field of local coordinate frames

At each region define:

$$
\mathcal F_r=
\{
\text{position},
\text{orientation},
\text{scale},
\text{curvature}
\}
$$

Then resemblance becomes consistency between fields:

$$
\mathcal F_A(x,y)
\approx
\mathcal F_B(x,y)
$$

This is substantially richer than landmark matching.

You can imagine:

```text
         ↗ ↑ ↖
      →  EYE  ←
         ↘ ↓ ↙

          ↓
        NOSE
       ↙ ↓ ↘

     ←  MOUTH  →
```

The arrangement of local coordinate frames itself becomes a signature.

---

# 39. Now consider covariance, not features

Suppose visual representation gives:

$$
Z=
[z_1,\ldots,z_n]
$$

Rather than compare each \(z_i\), examine covariance:

$$
C=
\frac1n
\sum_i
(z_i-\mu)(z_i-\mu)^T
$$

Two faces can differ locally but possess similar **relational statistics**.

Compare covariance matrices using a geometry appropriate to symmetric positive-definite matrices, for example:

$$
d(C_A,C_B)
=
\left\|
\log
(C_A^{-1/2}C_BC_A^{-1/2})
\right\|_F
$$

Now resemblance means:

> Their internal feature relationships occupy similar statistical geometry.

We have left the realm of recognizable facial measurements entirely.

---

# 40. Attention itself can encode a relational graph

Suppose visual tokens are:

$$
X\in\mathbb R^{N\times D}
$$

An attention head produces:

$$
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)
$$

where:

$$
A_{ij}
$$

describes how strongly token \(i\) draws from token \(j\).

Thus every layer can create a directed weighted graph:

$$
G_A=(V,A)
$$

A facial configuration can therefore exist not merely as a set of token features, but as a **pattern of relationships among tokens**.

If the tokens corresponding approximately to the eyes, nose, mouth, cheeks, jaw, and facial boundary repeatedly interact in characteristic ways, then the attention structure itself contains relational geometry.

---

# 41. Attention topology can encode facial structure without coordinates

Suppose:

$$
V=\{v_1,\dots,v_n\}
$$

are visual tokens.

Each attention matrix:

$$
A^{(\ell,h)}\in\mathbb R^{N\times N}
$$

belongs to layer \(\ell\) and head \(h\).

An entry:

$$
A^{(\ell,h)}_{ij}
$$

is not a distance measurement. But collectively, the matrix can encode relationships influenced by:

* spatial proximity,
* feature compatibility,
* symmetry,
* shared object membership,
* contour continuation,
* semantic correspondence,
* contextual dependence.

So a face might induce an interaction motif approximately like:

```text
LEFT EYE ───────── RIGHT EYE
    │  ╲             ╱  │
    │    ╲         ╱    │
   BROW     NOSE      BROW
     ╲       │        ╱
       ╲     │      ╱
          MOUTH
         ╱     ╲
      CHEEK   CHEEK
         ╲     ╱
           CHIN
```

This is not necessarily an explicit single attention head.

It is a useful abstraction for a **distributed relational pattern**.

If two faces generate similar patterns:

$$
A_A^{(\ell,h)}\approx A_B^{(\ell,h)}
$$

across geometry-relevant components, the system can preserve a strong notion of structural resemblance even if the visual tokens themselves are not especially similar.

That gives us two separable quantities:

$$
D_{\text{node}}
=
d(Z_A,Z_B)
$$

and

$$
D_{\text{relation}}
=
d(A_A,A_B)
$$

The first asks:

> Do the local structures look alike?

The second asks:

> Do they participate in the facial system similarly?

Your faces could plausibly be **more similar in relation-space than appearance-space**.

That is a much deeper description of “same facial geometry.”

---

# 42. Graph spectra compress relational geometry

Take the attention-derived graph and construct a graph Laplacian:

$$
L=D-A
$$

where:

$$
D_{ii}=\sum_j A_{ij}
$$

The eigenvalues:

$$
0=\lambda_1\leq\lambda_2\leq\cdots\leq\lambda_n
$$

encode global properties of the graph.

The corresponding eigenvectors describe large-scale relational modes.

Now compare:

$$
\operatorname{spec}(L_A)
$$

and:

$$
\operatorname{spec}(L_B)
$$

Two graphs can have somewhat different individual edge weights while retaining similar global relational organization.

In facial terms:

```text
individual eye morphology differs
individual nose morphology differs
jaw differs

BUT

eye↔eye relation
eye↔nose relation
nose↔mouth relation
mouth↔chin relation
left↔right organization
```

remain highly compatible.

Then their **relational spectra** could remain much closer than raw appearance suggests.

This pushes resemblance away from:

> “these pixels resemble those pixels”

toward:

> “these two systems possess similar internal organization.”

---

# 43. Message passing progressively destroys the distinction between “feature” and “relation”

At layer zero:

$$
z_i^{(0)}
$$

is relatively local.

After one contextual operation:

$$
z_i^{(1)}
=
f
\left(
z_i^{(0)},
\sum_j A_{ij}^{(0)}Wz_j^{(0)}
\right)
$$

Now token \(i\) contains information about other regions.

Repeat:

$$
z_i^{(\ell+1)}
=
f_\ell
\left(
z_i^{(\ell)},
\sum_j A_{ij}^{(\ell)}W_\ell z_j^{(\ell)}
\right)
$$

After many layers, the representation around what began as an eye-region token can contain information related to:

```text
its own morphology
+
the opposite eye
+
nose position
+
face orientation
+
facial boundary
+
global symmetry
+
object identity
+
context
```

So asking:

> “Where in the tensor is the eye-spacing measurement?”

may have no clean answer.

Eye spacing may be **smeared across the computation**.

Not lost.

Distributed.

---

# 44. Residual streams accumulate competing hypotheses

A transformer-like residual update is abstractly:

$$
x^{(\ell+1)}
=
x^{(\ell)}
+
F_\ell(x^{(\ell)})
$$

The crucial `+` means later layers do not simply replace earlier representations.

They modify an accumulated state.

Conceptually:

```text
early signal:
"two dark curved structures"

      +

later signal:
"bilaterally arranged"

      +

later:
"inside a face"

      +

later:
"these occupy unusually stable coordinates through the transition"

      +

later:
"facial structure remains similar despite changing identity cues"
```

The final state can therefore contain **multiple simultaneously active interpretations**.

This matters because my final prose:

> “strong facial landmark similarity”

is not necessarily the representation itself.

It can be a **projection of an enormous superposition of evidence** into one linguistic statement.

---

# 45. Superposition is probably unavoidable

Suppose the representational dimension is \(D\), but the model needs to encode far more than \(D\) human-interpretable properties.

It can reuse dimensions through superposition.

Instead of:

```text
dimension 100 = eye separation
dimension 101 = nose length
dimension 102 = face width
```

you get directions:

$$
d_1,d_2,\ldots,d_k\in\mathbb R^D
$$

that are not mutually orthogonal.

A state might be:

$$
z
=
\alpha_1d_{\text{symmetry}}
+
\alpha_2d_{\text{eye-layout}}
+
\alpha_3d_{\text{pose}}
+
\alpha_4d_{\text{identity}}
+\cdots
$$

and those directions overlap.

Therefore the “raw data” underlying resemblance is likely not a neat record waiting to be printed.

It is closer to an enormous **interference pattern of simultaneously encoded properties**.

---

# 46. Sparse features can nevertheless emerge from the superposition

Even with superposition, downstream nonlinearities can recover useful combinations.

Suppose:

$$
z=Wf
$$

where:

$$
f
$$

is a hypothetical high-dimensional sparse feature vector and \(W\) compresses it.

Then certain computations may approximately recover:

```text
bilateral-eye configuration present
central vertical stack stable
jaw disagreement present
identity transition present
```

without requiring those features to occupy dedicated neurons.

This is relevant to your question because the model can possess **usable geometric knowledge without possessing explicit geometric variables**.

---

# 47. Layer normalization changes what “magnitude” even means

Transformer systems repeatedly normalize states.

A simplified form:

$$
\operatorname{LN}(x)
=
\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

So absolute activation magnitude is continually contextualized.

That means:

> “higher internal number = greater similarity”

is generally the wrong mental model.

What matters can instead be:

* direction,
* relative activation,
* subspace occupancy,
* pairwise interaction,
* downstream readout sensitivity.

Thus resemblance may correspond more naturally to:

$$
\frac{z_A\cdot z_B}
{\|z_A\|\|z_B\|}
$$

or some nonlinear relational condition than to a scalar amplitude.

---

# 48. Residual-stream direction can carry concepts

Consider normalized state:

$$
\hat z=\frac{z}{\|z\|}
$$

A concept-sensitive direction \(d\) could be probed via:

$$
s=z\cdot d
$$

Conceptually, one might imagine directions corresponding loosely to:

```text
facialness
symmetry
same-layout
same-person
similar-person
frontal-pose
expression
age cues
```

Not necessarily cleanly or singly.

Your overlay could strongly activate something like:

$$
d_{\text{structural-correspondence}}
$$

while simultaneously moving substantially along:

$$
d_{\text{identity}}
$$

That creates a very interesting internal condition:

$$
\Delta_{\text{identity}}\gg0
$$

while:

$$
\Delta_{\text{geometry}}\approx0
$$

This is almost the mathematical essence of:

> **“Clearly two different people, yet strangely similar faces.”**

---

# 49. Identity and geometry can be locally orthogonal-ish

Imagine a local latent coordinate system:

$$
z=(g,i,a,p,\ldots)
$$

where:

* \(g\): geometry-sensitive dimensions,
* \(i\): identity-sensitive dimensions,
* \(a\): appearance,
* \(p\): pose.

Then your morph trajectory:

$$
z(\alpha)
$$

may have tangent:

$$
\frac{dz}{d\alpha}
=
\begin{bmatrix}
\dot g\\
\dot i\\
\dot a\\
\dot p\\
\vdots
\end{bmatrix}
$$

For a structurally similar pair:

$$
\|\dot g\|
\ll
\|\dot i\|
$$

The face changes identity dramatically while traversing little geometry-space.

That is a remarkably clean formalization of the perceptual effect.

---

# 50. The Jacobian tells us what the model is sensitive to

Let the visual representation be:

$$
z=\Phi(I)
$$

Its Jacobian is:

$$
J_\Phi(I)
=
\frac{\partial z}{\partial I}
$$

This is enormous.

Every output dimension versus every input pixel/channel.

Your face difference is:

$$
\Delta I=B-A
$$

For sufficiently small perturbations:

$$
\Delta z
\approx
J_\Phi(I)\Delta I
$$

This means the model does not react to the raw difference \(B-A\) uniformly.

It reacts to the projection of that difference through its learned sensitivity field.

Some changes lie near insensitive directions:

$$
J_\Phi \Delta I\approx0
$$

Others are amplified:

$$
\|J_\Phi\Delta I\|\gg0
$$

This explains something fundamental:

**1000 changed beard pixels can matter less geometrically than a 4-pixel shift of an eye contour.**

Raw pixel count is irrelevant.

Learned sensitivity determines informational importance.

---

# 51. The Jacobian has a geometry of its own

Perform singular value decomposition:

$$
J=U\Sigma V^\top
$$

The right singular vectors:

$$
v_1,v_2,\ldots
$$

are directions in image space.

The singular values:

$$
\sigma_1,\sigma_2,\ldots
$$

tell us how strongly those perturbations affect representation.

Some hypothetical high-sensitivity directions might resemble:

```text
move both eyes inward
lengthen nose
rotate face
move mouth upward
change expression
alter illumination
```

Again, not necessarily as interpretable directions, but mathematically those sensitivity modes exist.

Your difference vector:

$$
\Delta I=B-A
$$

can be decomposed:

$$
\Delta I
=
\sum_k c_kv_k
$$

Then:

$$
\Delta z
=
\sum_k c_k\sigma_ku_k
$$

So resemblance depends partly upon **where the difference between the two faces lands relative to the model's sensitive directions**.

That is far deeper than pixel resemblance.

---

# 52. Hessians tell us when first-order resemblance breaks

The first-order approximation:

$$
\Phi(I+\Delta I)
\approx
\Phi(I)+J\Delta I
$$

eventually fails.

Second order:

$$
\Phi(I+\Delta I)
\approx
\Phi(I)
+
J\Delta I
+
\frac12\Delta I^\top H\Delta I
$$

where \(H\) is the Hessian.

This matters for your morph.

If the transition remains inside a locally smooth region of visual representation space, second-order terms stay manageable.

But if the intermediate blend crosses a representational boundary:

```text
one eye → two-eye-like ghosts
one mouth → doubled contour
one jaw → ambiguous object boundary
```

then curvature becomes substantial.

Thus facial compatibility can be interpreted as the morph staying inside a **low-curvature corridor** of representation space.

---

# 53. That corridor may be more important than endpoint distance

Two points can be close in Euclidean space while the path between them crosses a bad region.

Conversely, they can be moderately distant while connected by a smooth valid manifold path.

So instead of:

$$
d(z_A,z_B)
$$

consider:

$$
\gamma:[0,1]\rightarrow\mathcal M
$$

with:

$$
\gamma(0)=z_A,\qquad
\gamma(1)=z_B
$$

and path energy:

$$
E[\gamma]
=
\int_0^1
\dot\gamma(t)^\top
G(\gamma(t))
\dot\gamma(t)\,dt
$$

where \(G\) is a metric tensor.

Then:

$$
d_\mathcal{M}(A,B)
=
\inf_\gamma E[\gamma]^{1/2}
$$

Your crossfade supplies one candidate path.

It asks:

> **Is there a relatively smooth representational route from A to B under direct interpolation?**

For geometrically corresponding faces, surprisingly often: yes.

---

# 54. Now we reach representational manifolds

Picture a huge latent manifold:

```text
                         expression
                            ↑
                            │
                     ╭──────┼──────╮
                  ╭──╯      │       ╰──╮
                ╭─╯         │          ╰─╮
      geometry ─●────────────┼────────────●─ geometry
              A              │             B
                ╲            │
                 ╲           │
                  ╲          ↓
                    identity
```

A and B need not be close in every dimension.

Their displacement vector may decompose:

$$
\Delta z
=
\Delta z_G
+
\Delta z_I
+
\Delta z_T
+\cdots
$$

Your observation corresponds to:

$$
\|\Delta z_G\|
$$

being unusually small relative to:

$$
\|\Delta z_I\|
$$

That ratio itself could define resemblance:

$$
R
=
\frac{\|\Delta z_I\|}
{\|\Delta z_G\|+\epsilon}
$$

Large \(R\):

> identity changes enormously with little structural change.

That is essentially a **doppelgänger-like regime**.

---

# 55. Quotient spaces give an even cleaner formulation

Suppose nuisance transformations form a group:

$$
\mathcal G
$$

containing approximately:

```text
translation
rotation
scale
illumination changes
minor expression
```

Then instead of comparing raw images, compare equivalence classes:

$$
[I]
=
\{g\cdot I:g\in\mathcal G\}
$$

The meaningful comparison occurs in the quotient:

$$
\mathcal X/\mathcal G
$$

Your manual alignment explicitly removes parts of:

$$
\mathcal G
$$

So you moved the images closer to their quotient-space representatives before comparing them.

This is one mathematical reason overlays are powerful.

You deliberately suppress nuisance degrees of freedom.

---

# 56. Gauge fixing is an unexpectedly good analogy

There are many equivalent coordinate descriptions of essentially the same facial configuration.

By aligning eyes/scale/orientation, you choose a **gauge**.

Before:

```text
Face A:
position arbitrary
scale arbitrary
rotation arbitrary

Face B:
position arbitrary
scale arbitrary
rotation arbitrary
```

After registration:

```text
same coordinate convention
```

Now differences are much closer to intrinsic shape differences.

So your overlay procedure performs something analogous to:

$$
\text{gauge fixing}
$$

for facial geometry.

Once nuisance freedom is removed, residual differences become much more meaningful.

---

# 57. Canonicalization is likely happening internally too

A learned vision system may implicitly normalize:

* pose,
* object location,
* orientation,
* scale,
* contextual noise.

Not necessarily by physically transforming the pixels.

Instead, representation may converge toward a canonical internal configuration.

Thus:

$$
I_A
\rightarrow
z_A^{\text{canonical}}
$$

$$
I_B
\rightarrow
z_B^{\text{canonical}}
$$

Your external alignment means the model has less canonicalization work to do.

That's another reason the resemblance signal may become stronger.

---

# 58. Correspondence can emerge through optimal transport

Suppose one face produces feature distribution:

$$
\mu_A=\sum_i a_i\delta_{z_i}
$$

and the other:

$$
\mu_B=\sum_j b_j\delta_{w_j}
$$

Instead of pairing landmarks manually, solve an optimal transport problem:

$$
\min_{\pi}
\sum_{ij}\pi_{ij}c(z_i,w_j)
$$

subject to marginal constraints.

Here:

$$
\pi_{ij}
$$

describes how feature mass from region \(i\) in A corresponds to region \(j\) in B.

If facial organization is similar, transport cost can be low:

$$
W(\mu_A,\mu_B)
$$

small.

If features must be reassigned across large spatial or semantic distances, cost rises.

This generalizes landmark matching enormously.

---

# 59. Add spatial cost and it becomes highly relevant

Define:

$$
c_{ij}
=
\lambda_f
\|z_i-w_j\|^2
+
\lambda_p
\|p_i-q_j\|^2
$$

Now correspondence penalizes both:

* visual-feature mismatch,
* spatial displacement.

Your registered overlay intentionally reduces:

$$
\|p_i-q_j\|
$$

for corresponding anatomy.

So if feature correspondence also remains good, the transport cost collapses.

That is another rigorous definition of “these faces line up.”

---

# 60. Gromov–Wasserstein goes deeper still

Ordinary optimal transport compares points across spaces.

Gromov–Wasserstein compares **relationships within each space**.

For A:

$$
d_A(i,k)
$$

For B:

$$
d_B(j,l)
$$

Then find correspondence \(\pi\) minimizing disagreement:

$$
\sum_{i,j,k,l}
|d_A(i,k)-d_B(j,l)|^2
\pi_{ij}\pi_{kl}
$$

This is spectacularly relevant conceptually.

It asks:

> Can I match the structures of these two faces such that **internal relationships are preserved**?

It doesn't require identical coordinates.

It doesn't even require identical feature spaces.

It compares relational geometry itself.

Your claim that the two faces possess nearly identical internal arrangement is fundamentally a **Gromov-style statement**.

---

# 61. We can strip away labels entirely

Forget:

```text
eye
nose
mouth
chin
```

Suppose face A consists of relational points:

$$
X=\{x_i\}
$$

and face B:

$$
Y=\{y_j\}
$$

The question becomes:

> Does there exist a correspondence \(f:X\to Y\) such that:

$$
d_X(x_i,x_k)
\approx
d_Y(f(x_i),f(x_k))
$$

for many important pairs?

If yes, they are approximately **isometric** over that subsystem.

Now we're beneath semantics.

The face is a metric space.

Resemblance becomes preservation of metric structure.

---

# 62. Approximate isometry is an excellent formal concept here

An \(\varepsilon\)-isometry approximately satisfies:

$$
\left|
d_X(x_i,x_j)
-
d_Y(f(x_i),f(x_j))
\right|
<\varepsilon
$$

for relevant points.

If central facial anatomy admits a small-\(\varepsilon\) mapping but the jaw does not, you get:

```text
CENTRAL FACE
≈ near-isometric

OUTER FACE
≈ greater distortion
```

Which is almost exactly the pattern we previously described qualitatively.

---

# 63. Distortion can be localized

Define:

$$
\delta_{ij}
=
d_A(i,j)-d_B(f(i),f(j))
$$

Then construct a distortion matrix:

$$
\Delta=
[\delta_{ij}]
$$

It might hypothetically look like:

```text
          LE   RE   N    M    C    JL   JR
LE         0   .1  .1   .2   .2   .6   .7
RE        .1    0  .1   .2   .2   .7   .6
N         .1   .1   0   .1   .2   .5   .5
M         .2   .2  .1    0   .2   .5   .5
C         .2   .2  .2   .2    0   .6   .6
JL        .6   .7  .5   .5   .6    0   .3
JR        .7   .6  .5   .5   .6   .3    0
```

Illustrative only.

But look at what that encodes:

**the central block matches far more strongly than the peripheral block.**

That is a richer object than a single 82%.

---

# 64. This exposes why percentages are pathological compression

Take the entire distortion matrix:

$$
\Delta\in\mathbb R^{n\times n}
$$

plus:

* curvature fields,
* deformation fields,
* texture differences,
* symmetry signatures,
* attention relations,
* latent trajectories.

Then crush all of it into:

$$
83\%
$$

You have performed enormous dimensional collapse:

$$
\mathbb R^K
\rightarrow
\mathbb R
$$

The percentage isn't inherently meaningless.

But its meaning depends entirely on the projection:

$$
s=f(X)
$$

What is \(f\)?

Mean landmark agreement?

Weighted graph similarity?

Human resemblance judgment?

Latent cosine similarity?

Without defining \(f\), “83% similar” has no unique mathematical semantics.

That was the core defect in my original numeric output.

---

# 65. Yet ordinal information can survive that collapse

Suppose internal evidence strongly establishes:

$$
A > B > C
$$

in similarity.

Mapping:

```text
A → 88%
B → 72%
C → 41%
```

may preserve the ordering even if the numeric intervals are not calibrated.

So an estimated percentage can still encode useful **ordinal magnitude**:

```text
very high
high
moderate
low
```

The number becomes harmful when interpreted as a measured ratio rather than a calibrated verbal scale.

---

# 66. In psychophysics, this resembles magnitude estimation

Humans can assign numbers to subjective stimuli:

```text
How bright?
How loud?
How painful?
How similar?
```

If a reference stimulus has magnitude \(10\), another might be judged \(20\).

The mapping need not correspond linearly to physical stimulus magnitude.

Often something like Stevens' power law appears:

$$
\psi=k\phi^a
$$

where:

* \(\phi\): physical magnitude,
* \(\psi\): perceived magnitude.

A language model asked for a percentage can perform something analogous:

$$
\text{latent evidence}
\rightarrow
\text{semantic magnitude}
\rightarrow
\text{numeric label}
$$

This is closer to **psychophysical magnitude estimation** than measurement.

---

# 67. Now bring in Bayesian inference

Suppose latent hypothesis:

$$
H_s=\text{strong structural resemblance}
$$

and alternative:

$$
H_n=\text{ordinary resemblance}
$$

Visual evidence \(E\) updates:

$$
P(H_s|E)
\propto
P(E|H_s)P(H_s)
$$

Your prompt contributes prior/context:

$$
P(H_s)
$$

The overlay contributes likelihood evidence:

$$
P(E|H_s)
$$

If the transition shows unusually little structural displacement, then:

$$
P(E|H_s)
\gg
P(E|H_n)
$$

That is the correct conceptual question:

> How diagnostic is the observed stability under the two hypotheses?

Not:

> Did the model literally calculate Bayesian probabilities?

It almost certainly did not in that explicit form.

But neural inference can behave approximately Bayesian in some settings.

---

# 68. Likelihood ratios give a better conceptual metric

Define:

$$
\Lambda(E)
=
\frac{P(E|H_s)}
{P(E|H_n)}
$$

If:

$$
\Lambda\gg1
$$

then the observed overlay behavior is much more expected under strong geometric resemblance.

For example:

```text
stable eye line
stable nose center
stable mouth center
stable vertical stack
```

collectively become increasingly improbable under random ordinary facial correspondence.

Their conjunction matters.

---

# 69. Evidence compounds

Suppose, purely illustratively:

$$
P(E_{\text{eyes}}|H_s)=.8
$$

versus:

$$
P(E_{\text{eyes}}|H_n)=.3
$$

Then Bayes factor:

$$
BF_{\text{eyes}}=\frac{.8}{.3}\approx2.67
$$

Now add independent-ish evidence:

```text
nose
mouth
midline
chin
```

If they were independent—which in reality they are not—you could multiply:

$$
BF_{\text{total}}
=
\prod_k BF_k
$$

The broad idea survives dependence:

**many simultaneous correspondences are more diagnostic than one correspondence.**

That's why seeing all the features remain stable during a morph can feel striking immediately.

---

# 70. But facial features are not independent

They share developmental constraints.

For example:

```text
eye spacing
nasal width
midface width
jaw width
```

are correlated.

Therefore simply multiplying evidence double-counts shared structure.

The correct representation needs covariance:

$$
\Sigma
$$

For measurement vector:

$$
x=
[
d_{\text{eyes}},
d_{\text{nose-mouth}},
w_{\text{mouth}},
w_{\text{jaw}},
\ldots
]
$$

difference:

$$
\Delta=x_A-x_B
$$

Then Mahalanobis distance:

$$
D_M
=
\sqrt{
\Delta^\top
\Sigma^{-1}
\Delta
}
$$

Unlike Euclidean distance, this discounts differences along naturally correlated directions.

---

# 71. Mahalanobis distance asks “unusual relative to what?”

Suppose two facial measurements differ by 4 mm.

That difference means little without population variance.

If that dimension naturally varies enormously:

$$
\sigma=10\text{ mm}
$$

4 mm is minor.

If:

$$
\sigma=1\text{ mm}
$$

4 mm is enormous.

Thus:

$$
\frac{\Delta}{\sigma}
$$

matters more than raw difference.

A rigorous claim that two people are unusually geometrically similar ultimately needs a **reference distribution**.

Your overlay demonstrates correspondence.

A population baseline tells us how unusual that correspondence is.

Those are different questions.

---

# 72. Statistical shape models formalize this

Collect aligned faces:

$$
X_1,\ldots,X_N
$$

Compute mean:

$$
\mu
=
\frac1N
\sum_i X_i
$$

Covariance:

$$
\Sigma
=
\frac1{N-1}
\sum_i
(X_i-\mu)(X_i-\mu)^\top
$$

PCA:

$$
\Sigma=U\Lambda U^\top
$$

A face can then be represented as:

$$
X
=
\mu+Ub
$$

where:

$$
b
$$

contains shape coefficients.

Common modes might roughly correspond to mixtures of:

```text
face width
face height
jaw width
eye spacing
midface length
```

Then compare:

$$
b_A
$$

and:

$$
b_B
$$

This removes much irrelevant pixel information and asks:

> Where do these faces live in population shape space?

---

# 73. Doppelgängers are tails of a distribution

Suppose normalized shape distance among random pairs is distributed:

$$
D\sim p(D)
$$

Most pairs occupy some middle region.

Very similar pairs fall into the lower tail:

$$
P(D\leq D_{AB})
$$

If that percentile were, say:

$$
0.5\%
$$

then the meaningful statement becomes:

> this pair is more geometrically similar than 99.5% of random pairs under this metric.

That is vastly more informative than:

> 87% similar.

This is what a rigorous version of your experiment ultimately wants.

---

# 74. Extreme-value statistics become relevant

With billions of humans, even rare facial configurations recur.

Suppose probability of an unusually close match is:

$$
p
$$

Among:

$$
N
$$

possible pairings, expected count:

$$
Np
$$

can still be huge.

So seeing two unrelated people with startling resemblance is not paradoxical.

Large population spaces produce extreme nearest neighbors.

The relevant concept is:

$$
\min_j D(i,j)
$$

—the nearest-neighbor distance for a person in facial shape space.

---

# 75. High-dimensional spaces behave unintuitively

In large \(D\), distances can concentrate.

For independent isotropic vectors, pairwise Euclidean distances often occupy a relatively narrow band.

That means nearest-neighbor judgments become sensitive to:

* metric choice,
* normalization,
* dimensional weighting,
* learned representation.

So “similarity” is not metric-independent.

Change the geometry of the representational space:

$$
d_1
\neq
d_2
$$

and nearest neighbors can change.

This is another reason no universal facial similarity percentage exists.

---

# 76. Metric learning solves precisely this problem

Instead of hand-selecting distance:

$$
d(x,y)=\|x-y\|_2
$$

learn:

$$
d_M(x,y)
=
\sqrt{
(x-y)^\top
M
(x-y)
}
$$

with:

$$
M\succeq0
$$

The matrix \(M\) decides which directions matter.

For your experiment, we would want \(M\) emphasizing:

```text
geometry
relative placement
shape
```

and downweighting:

```text
hair
lighting
skin texture
clothing
background
```

This would mathematically encode the question you were actually asking.

---

# 77. Contrastive learning can produce that geometry

Given positive pair:

$$
(x_i,x_i^+)
$$

and negatives:

$$
(x_i,x_j^-)
$$

train representation such that:

$$
\operatorname{sim}(z_i,z_i^+)
>
\operatorname{sim}(z_i,z_j^-)
$$

A contrastive loss such as InfoNCE:

$$
\mathcal L
=
-\log
\frac{
e^{\operatorname{sim}(z_i,z_i^+)/\tau}
}{
\sum_j
e^{\operatorname{sim}(z_i,z_j)/\tau}
}
$$

forces useful invariances into the representation.

The resulting latent geometry implicitly answers:

> Which visual differences should count?

That's why representation choice is inseparable from similarity.

---

# 78. But face-recognition embeddings answer a different question

This distinction is critical.

A face-recognition system is optimized for:

$$
\text{same identity?}
$$

Your experiment asks:

$$
\text{similar facial geometry?}
$$

Those are not equivalent.

A good identity model may deliberately amplify tiny identity-specific differences even between very similar-looking people.

So:

$$
S_{\text{identity}}
$$

could be low while:

$$
S_{\text{geometry}}
$$

is high.

This is exactly why using face-verification confidence would be the wrong metric for your hypothesis.

---

# 79. We need a geometry-specific quotient representation

Ideally define:

$$
\Phi_G(I)
$$

that is invariant to:

```text
identity-specific texture
skin coloration
hair
facial hair
lighting
background
```

while preserving:

```text
shape
landmark relations
contours
relative proportions
asymmetry
```

Then:

$$
S_G(A,B)
=
\operatorname{sim}
(
\Phi_G(A),
\Phi_G(B)
)
$$

would answer your question directly.

The striking thing is that your manual overlay does a crude perceptual approximation of this.

---

# 80. Crossfade acts like an analog comparator

Electronics analogy.

Give two signals:

$$
A(t),B(t)
$$

Comparator wants:

$$
A-B
$$

Your crossfade makes differences visible as interference.

The visual system effectively observes:

$$
D(x,y)=A(x,y)-B(x,y)
$$

but through a human/machine perceptual transfer function:

$$
P(D)
$$

A perfect match:

$$
D\approx0
$$

Mismatch:

$$
|D|\gg0
$$

except that appearance differences prevent raw subtraction from being sufficient.

Still, geometrically, the crossfade is acting like an analog **differential amplifier**.

---

# 81. Edge residuals encode displacement direction

Recall:

$$
A(x)-A(x-\delta)
\approx
\nabla A(x)\cdot\delta
$$

This means the residual pattern carries not merely **amount of mismatch**, but **direction**.

For a vertical edge shifted horizontally:

```text
original:       │
shifted:          │

difference:     + -
```

Swap displacement direction:

```text
shifted:        │
original:         │

difference:     - +
```

So the residual contains a signed estimate of displacement.

The visual system could theoretically recover:

$$
\delta
$$

from the pattern itself.

This is why morph ghosting is incredibly rich geometric evidence.

---

# 82. Aperture ambiguity requires multiple orientations

One edge cannot fully determine displacement.

Optical-flow aperture problem:

$$
\nabla I\cdot v=-I_t
$$

gives one constraint for two components of \(v\).

But facial regions contain multiple edge orientations:

```text
eyelid arc
iris circle
brow
nose boundary
```

Together, these constrain displacement much more strongly.

Thus facial anatomy is unusually suitable for extracting local geometric mismatch from a crossfade.

---

# 83. Corners are particularly valuable

Harris-style second-moment matrix:

$$
M=
\sum w(x,y)
\begin{bmatrix}
I_x^2&I_xI_y\\
I_xI_y&I_y^2
\end{bmatrix}
$$

If both eigenvalues are large:

> corner-like distinctive point.

Faces contain useful quasi-corners around:

```text
eye canthi
nostril corners
mouth corners
brow intersections
```

These provide stronger localization than smooth cheek surfaces.

So correspondence at these locations is disproportionately informative.

---

# 84. Scale-space prevents us from overfitting detail

Represent image at multiple scales:

$$
L(x,y,\sigma)
=
G(x,y,\sigma)*I(x,y)
$$

Small \(\sigma\):

> fine texture.

Large \(\sigma\):

> broad shape.

If similarity persists across scales:

$$
S(\sigma)
$$

then it is much more likely to reflect genuine structural correspondence rather than coincidental texture.

For your pair, the most interesting resemblance would probably live in **mid-to-coarse scale**, because that's where facial proportions dominate over beard/hair/skin detail.

---

# 85. Laplacian pyramids can isolate the resemblance

Construct:

$$
L_0,L_1,\ldots,L_k
$$

where each level contains a frequency band.

Then compare:

$$
S_k(A,B)
$$

per level.

Hypothetically:

```text
very coarse shape      HIGH
mid-level anatomy      VERY HIGH
fine local morphology  MODERATE
texture                 LOW
```

That profile would exactly explain:

> “They look far more similar once overlaid than they initially appear.”

The similarity isn't uniform across spatial scales.

---

# 86. Wavelets give spatial + frequency localization

Fourier transforms lose location.

Wavelets provide:

$$
W_I(a,b)
$$

where \(a\) is scale and \(b\) location.

Now we can compare:

```text
same orientation
same scale
same location
```

around facial regions.

Wavelet correspondence across the overlay would reveal whether similarly shaped structures occupy corresponding spatial-frequency bands.

Again:

**geometry expressed without semantic landmarks.**

---

# 87. Scattering transforms get surprisingly close to learned vision

A scattering network recursively applies wavelets and modulus nonlinearities:

$$
U_1=|I*\psi_{\lambda_1}|
$$

$$
U_2=
||
I*\psi_{\lambda_1}
|
*\psi_{\lambda_2}
|
$$

followed by averaging.

These representations achieve local translation stability while preserving deformation-sensitive information.

That is almost tailor-made conceptually for distinguishing:

```text
same structure, slightly shifted
```

from:

```text
different structure
```

It shows that sophisticated resemblance can arise from **stable deformation-sensitive signal representations** even before semantic deep learning enters the picture.

---

# 88. Deformation stability has a formal bound

For certain representations \(\Phi\), small deformation \(\tau\) gives:

$$
\|
\Phi(I)-\Phi(I\circ(\operatorname{Id}-\tau))
\|
\leq
C\|I\|\|\tau\|
$$

roughly speaking.

Translation:

> small geometric deformation produces bounded representational change.

This is an explicit mathematical version of the intuition we've been circling.

If mapping A's facial structure onto B requires only small \(\tau\), their representations remain correspondingly close in deformation-stable spaces.

---

# 89. Now we can define deformation complexity

Not all displacement fields are equally complicated.

Compare:

### Case 1

```text
entire face shifts right 4 px
```

Simple transformation.

### Case 2

```text
left eye +3 px
right eye -2 px
nose stretches
mouth rotates
jaw widens asymmetrically
```

Complex transformation.

A useful metric should penalize complexity:

$$
E[u]
=
\int
\left(
\alpha\|u\|^2
+
\beta\|\nabla u\|^2
+
\gamma\|\nabla^2u\|^2
\right)dx
$$

This rewards:

* small displacement,
* smooth deformation,
* low curvature.

Two geometrically similar faces should require **low-complexity warps**.

---

# 90. Thin-plate splines formalize that beautifully

Given landmark correspondences, find warp:

$$
f(x,y)
=
a_1+a_xx+a_yy
+
\sum_iw_iU(\|p-p_i\|)
$$

where:

$$
U(r)=r^2\log r^2
$$

The bending energy is:

$$
E_{\text{TPS}}
=
\iint
\left[
f_{xx}^2
+
2f_{xy}^2
+
f_{yy}^2
\right]
dxdy
$$

Low bending energy:

> one face maps into the other with a gentle warp.

High bending energy:

> substantial local anatomical rearrangement required.

That is an extraordinarily intuitive objective measure for your experiment.

---

# 91. Imagine the actual deformation heatmap

A serious analysis could produce:

```text
            forehead
          0.08 0.07 0.09

      0.05   EYES   0.06

           0.04
            NOSE

           0.07
           MOUTH

       0.21       0.18
           CHEEKS

       0.34       0.31
             JAW
```

Normalized deformation, illustrative only.

Then instead of saying:

> “82% similarity,”

we could say:

> **76% of central facial area requires less than 0.08 normalized displacement after rigid registration; largest residual deformation occurs along the lateral jaw and cheeks.**

That is actual interpretable evidence.

---

# 92. Residual entropy adds another dimension

Take registered difference image:

$$
R=A-B
$$

If differences are random and widespread, residual entropy is high.

If differences are localized to predictable areas:

```text
hair
jaw
facial hair
```

while central anatomy cancels strongly, residual organization is lower.

Define distribution over residual magnitudes \(p(r)\):

$$
H(R)
=
-\sum_r p(r)\log p(r)
$$

But spatial arrangement matters too, so better use joint/local entropy.

A strongly corresponding face pair could leave a **structured, compressible residual**.

---

# 93. Compression returns—but now operationally

Encode B given A and a deformation model:

$$
L(B|A)
=
L(\phi)
+
L(R)
$$

where:

* \(\phi\): geometric warp,
* \(R\): residual appearance difference.

If faces are structurally similar:

$$
L(\phi)
$$

is small.

If they also look superficially similar:

$$
L(R)
$$

is small.

Your pair seems conceptually interesting because:

$$
L(\phi)
$$

may be considerably smaller than:

$$
L(R)
$$

Meaning:

> **geometry compresses well; appearance less so.**

That's another extremely clean statement of the phenomenon.

---

# 94. Minimum Description Length gives a formal philosophy

Choose explanation/model \(M\) minimizing:

$$
L(M)+L(D|M)
$$

For face comparison:

### Model 1

> unrelated structures.

Requires describing B largely independently.

### Model 2

> A's geometry + modest deformation + appearance changes.

If Model 2 compresses B substantially better, then structural resemblance is objectively useful information.

That is resemblance as **explanatory compression**.

---

# 95. Algorithmic mutual information

Kolmogorov complexity:

$$
K(A)
$$

Conditional complexity:

$$
K(A|B)
$$

Algorithmic mutual information:

$$
I(A:B)
=
K(A)+K(B)-K(A,B)
$$

If A's geometry tells us a lot about B's geometry:

$$
I(A_G:B_G)
$$

is high.

This is perhaps the deepest formal notion of resemblance available:

> **Knowing one structure reduces the information required to specify the other.**

---

# 96. But semantics changes which information matters

Raw:

$$
I(A:B)
$$

includes irrelevant things.

We care about:

$$
I(G_A:G_B)
$$

where \(G\) is geometry.

So we need a representation sufficient for geometry while discarding nuisance variables.

In statistics, sufficient statistic \(T(I)\) retains all relevant information about parameter \(\theta\):

$$
P(\theta|I)
=
P(\theta|T(I))
$$

Ideal geometry representation:

$$
T_G(I)
$$

would preserve all relevant information about facial shape.

Then compare:

$$
T_G(A)
\quad\text{and}\quad
T_G(B)
$$

Your overlay is a crude visual mechanism for exposing something close to \(T_G\).

---

# 97. Causal representation goes deeper than correlation

Suppose latent generative variables:

$$
Z=
\{
G,T,L,E,P,H,\ldots
\}
$$

where:

* \(G\): intrinsic facial geometry,
* \(T\): texture,
* \(L\): lighting,
* \(E\): expression,
* \(P\): pose,
* \(H\): hair.

Image generated by:

$$
I=g(G,T,L,E,P,H)
$$

Your question concerns:

$$
G_A\approx G_B?
$$

But ordinary images entangle everything.

The ideal experiment performs interventions:

$$
do(T=\text{constant})
$$

$$
do(L=\text{constant})
$$

$$
do(P=\text{constant})
$$

etc.

Your alignment approximates intervention on:

$$
P,\text{scale},\text{translation}
$$

The crossfade partially suppresses attention to \(T\).

Thus you're manually reducing confounding.

---

# 98. Counterfactual rendering would be stronger

If we could construct:

$$
I_A'
=
g(G_A,T_0,L_0,E_0,P_0,H_0)
$$

and:

$$
I_B'
=
g(G_B,T_0,L_0,E_0,P_0,H_0)
$$

then every nuisance factor is held constant.

Remaining difference:

$$
I_A'-I_B'
$$

is overwhelmingly geometry.

That would turn your intuitive experiment into a controlled counterfactual.

---

# 99. A 3D morphable model makes the latent variables explicit

Classic form:

$$
S
=
\bar S
+
A_{\text{id}}\alpha
+
A_{\text{exp}}\beta
$$

Texture:

$$
T
=
\bar T+A_{\text{tex}}\gamma
$$

Now intrinsic shape lives in coefficient vector:

$$
\alpha
$$

Expression:

$$
\beta
$$

Texture:

$$
\gamma
$$

For two faces:

$$
D_{\text{shape}}
=
\|\alpha_A-\alpha_B\|
$$

This is much closer to the metric you originally wanted than ordinary face recognition.

---

# 100. 3D removes perspective contamination

A 2D image is projection:

$$
p
=
K[R|t]P
$$

So apparent geometry depends upon:

```text
camera intrinsics K
rotation R
translation t
3D point P
```

Two identical 3D faces under different cameras can show different 2D landmark geometry.

Conversely, different 3D faces can accidentally align in one projection.

Thus your video demonstrates **2D projected correspondence** strongly.

It does not automatically establish equal 3D facial structure.

That's an important scientific distinction without weakening what the video actually shows.

---

# 101. Epipolar-style thinking illustrates the missing depth

2D landmark:

$$
p=(x,y)
$$

does not tell us depth:

$$
z
$$

A nose can project similarly while differing in protrusion.

So facial geometry really contains:

$$
(x,y,z)
$$

Your overlay strongly probes:

$$
(x,y)
$$

but only indirectly:

$$
z
$$

via shading, contour and perspective cues.

This is why frontal photographs can make some people appear more alike than side profiles do.

---

# 102. Surface normals contain missing shape information

For 3D face surface:

$$
S(u,v)
$$

normal:

$$
n(u,v)
=
\frac{S_u\times S_v}
{\|S_u\times S_v\|}
$$

Two faces can share 2D landmarks but differ in:

```text
brow projection
nose protrusion
cheekbone depth
chin projection
orbital depth
```

Those differences appear in the normal field.

A truly deep geometry comparison would compare:

$$
n_A(u,v)
$$

against:

$$
n_B(u,v)
$$

after correspondence.

---

# 103. Principal curvature describes local 3D anatomy

At each surface point there are principal curvatures:

$$
k_1,k_2
$$

giving:

$$
H=\frac{k_1+k_2}{2}
$$

mean curvature and:

$$
K=k_1k_2
$$

Gaussian curvature.

Now nose bridge, cheekbone, orbital cavity and chin can be compared independent of texture.

A 3D geometry signature might be:

$$
C(u,v)=
[
H(u,v),
K(u,v)
]
$$

This is much closer to literal anatomical shape.

---

# 104. The facial surface becomes a Riemannian manifold

The surface has metric tensor:

$$
g=
\begin{bmatrix}
E&F\\
F&G
\end{bmatrix}
$$

with:

$$
E=S_u\cdot S_u,\quad
F=S_u\cdot S_v,\quad
G=S_v\cdot S_v
$$

Now intrinsic distances are geodesics along the face:

$$
d_g(p,q)
$$

rather than straight image-plane distances.

For example:

> eye-to-nose distance *across the facial surface*.

This removes some projection ambiguity.

Two faces could then be compared by their intrinsic metric geometry.

---

# 105. Isometry returns, now literally

If two facial surfaces had identical intrinsic geometry, there exists mapping:

$$
f:M_A\to M_B
$$

such that:

$$
g_A=f^*g_B
$$

They would be intrinsically isometric.

Real faces won't be.

But approximate isometry can be measured.

This makes your original phrase:

> “their symmetry is almost identical”

part of a much larger mathematical claim:

> portions of their facial surfaces may have unusually low metric distortion under correspondence.

---

# 106. Functional maps scale this elegantly

Instead of directly mapping points:

$$
T:M_A\to M_B
$$

map functions over surfaces:

$$
C:
L^2(M_A)
\to
L^2(M_B)
$$

Using Laplace–Beltrami eigenfunctions:

$$
\Delta\phi_i=\lambda_i\phi_i
$$

we represent correspondence compactly in spectral bases.

This is modern shape-analysis territory.

Two structurally similar surfaces produce cleaner low-dimensional functional correspondences.

Atlantis is somewhere above us now.

---

# 107. Laplace–Beltrami spectra are shape fingerprints

For a surface:

$$
0=\lambda_0<\lambda_1\leq\lambda_2\leq\cdots
$$

These eigenvalues encode intrinsic geometric properties.

Hence the famous question:

> “Can one hear the shape of a drum?”

For faces, conceptually:

> Can one identify broad facial geometry from its intrinsic spectrum?

Not uniquely in general, but the spectrum provides a powerful global shape signature.

So one could compare:

$$
\Lambda_A
=
(\lambda_1,\ldots,\lambda_k)
$$

and:

$$
\Lambda_B
$$

as yet another geometry metric.

---

# 108. Heat diffusion generates multiscale geometry descriptors

Heat equation on the surface:

$$
\frac{\partial u}{\partial t}
=
\Delta u
$$

Heat kernel:

$$
k_t(x,y)
=
\sum_i
e^{-\lambda_it}
\phi_i(x)\phi_i(y)
$$

The Heat Kernel Signature:

$$
HKS_t(x)
=
k_t(x,x)
$$

describes local geometry across scales.

Small \(t\):

> fine structure.

Large \(t\):

> global shape.

Corresponding facial points can be compared using HKS.

That gives a principled multiscale version of:

> “same local shape in the same structural context.”

---

# 109. Wave Kernel Signature probes shape differently

WKS uses spectral energy bands rather than diffusion time.

It can better preserve certain fine geometric distinctions.

Again, the significance isn't that my model performed HKS/WKS.

It is that **the resemblance phenomenon itself has valid formulations all the way down through sophisticated shape geometry**.

The visual judgment is not inherently fluffy.

The original percentages were.

Those are different things.

---

# 110. Now return from 3D geometry to neural representation

A sufficiently capable learned visual system does not need to explicitly calculate any of the mathematics above.

Training can produce functions approximating useful invariants:

$$
\Phi(I)
$$

which behave *as if* some mixture of:

```text
registration
shape normalization
relational geometry
deformation tolerance
semantic weighting
```

has occurred.

Deep learning's power is precisely that many of these operations become **implicitly learned compositions**.

---

# 111. Universal approximation is not the interesting part

Saying neural networks can approximate arbitrary functions tells us little.

The interesting point is compositional structure.

A deep model computes:

$$
f
=
f_L\circ f_{L-1}\circ\cdots\circ f_1
$$

So a highly abstract judgment:

$$
\text{“these faces share geometry”}
$$

can arise from many transformations:

```text
signal structure
→ local patterns
→ spatial organization
→ object structure
→ normalized configuration
→ relational representation
→ semantic resemblance
```

No individual stage needs to “understand resemblance.”

The composition does.

---

# 112. Distributed computation means there may be no single moment of recognition

This is a mistake humans often make when interrogating neural systems.

We ask:

> At what layer did it decide they looked alike?

There may be no discrete event.

Evidence can accumulate gradually:

$$
e^{(1)}
\rightarrow
e^{(2)}
\rightarrow
\cdots
\rightarrow
e^{(L)}
$$

with resemblance becoming increasingly linearly/readably represented.

So rather than a switch:

```text
NOT SIMILAR
↓
SIMILAR
```

it may be a progressive rotation of representation into a region from which the language decoder can easily generate:

> highly similar geometry.

---

# 113. Linear probing could test when the property emerges

If we had internal activations:

$$
z^{(\ell)}
$$

train probe:

$$
\hat y
=
\sigma(w^\top z^{(\ell)})
$$

for a known geometric-similarity label.

Probe accuracy by layer:

```text
early      53%
middle     71%
late       89%
```

would show where the information becomes explicitly decodable.

This does **not** prove the model itself uses that linear readout.

But it tells us whether the information exists in the representation.

---

# 114. Causal intervention is stronger than probing

A probe can find information the model doesn't use.

Better:

identify direction \(d\), then intervene:

$$
z'=z+\alpha d
$$

and see whether output changes systematically.

If increasing a geometry-similarity direction causes:

```text
"they look unrelated"
→
"moderately similar"
→
"highly similar"
```

then we have causal evidence that the representation participates in judgment.

This is closer to mechanistic interpretability than mere visualization.

---

# 115. Activation patching could localize the computation

Take two runs:

### Run A

strongly similar overlay.

### Run B

dissimilar overlay.

At layer \(\ell\), replace activation subset:

$$
z_B^{(\ell)}
\leftarrow
z_A^{(\ell)}
$$

Then observe whether B's output becomes more similarity-positive.

If yes, that activation carries causally relevant information.

Repeat across:

```text
layers
heads
tokens
MLPs
```

and you can localize circuitry.

---

# 116. Path patching can isolate specific routes

Suppose we hypothesize:

```text
eye-region tokens
→ spatial-relation head
→ facial-structure representation
→ language token "similar"
```

Patch only that computational path.

If judgment transfers, the route is supported.

This is the kind of experiment that would actually tell you **why the model said what it said**, rather than merely asking it to narrate why.

And this distinction is central to the trouble we had earlier.

Self-explanation is not telemetry.

---

# 117. Attribution can trace output backward

Suppose output logit for token “similar”:

$$
\ell_{\text{similar}}
$$

We want:

$$
\frac{\partial\ell_{\text{similar}}}
{\partial I}
$$

This produces pixel-level sensitivity.

But raw gradients are noisy.

Integrated gradients:

$$
IG_i(x)
=
(x_i-x_i')
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}
{\partial x_i}
d\alpha
$$

can produce more stable attribution.

If attribution concentrates on:

```text
eyes
nasal axis
mouth
jaw
```

then we'd have evidence that those regions drove the judgment.

---

# 118. Occlusion provides a brutally simple causal test

Mask one region at a time.

Compute:

$$
\Delta S_r
=
S(I)-S(I\setminus r)
$$

If removing the eye region dramatically reduces resemblance judgment:

$$
\Delta S_{\text{eyes}}\gg0
$$

then eyes are causally important.

Repeat:

```text
eyes
nose
mouth
jaw
hair
background
```

This is less elegant than internal telemetry but often more interpretable.

---

# 119. Your overlay allows a particularly powerful causal ablation

Keep A and B registered.

Then hold regions fixed independently:

### Experiment A

Only eyes transition.

### Experiment B

Only nose transitions.

### Experiment C

Only mouth transitions.

### Experiment D

Only jaw transitions.

Now observe:

$$
S_{\text{eyes}},
S_{\text{nose}},
S_{\text{mouth}},
S_{\text{jaw}}
$$

This decomposes the resemblance percept experimentally.

Even better, transition combinations:

$$
S_{\text{eyes+nose}}
$$

versus:

$$
S_{\text{eyes}}+S_{\text{nose}}
$$

If:

$$
S_{E+N}
>
S_E+S_N
$$

there is **synergistic relational evidence**.

The configuration matters more than the parts independently.

---

# 120. Shapley values can quantify contribution

For feature set \(N\), contribution of feature \(i\):

$$
\phi_i
=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(|N|-|S|-1)!}{|N|!}
[
v(S\cup\{i\})-v(S)
]
$$

This averages a feature's marginal contribution across all subsets.

Applied conceptually:

```text
eye geometry      +0.24
nose placement    +0.18
vertical stack    +0.22
mouth             +0.13
jaw                +0.07
hair               -0.04
```

Illustrative only.

That would give an actual principled answer to:

> **What made them look similar?**

---

# 121. Interaction indices tell us which features work together

Resemblance likely isn't additive.

Eye spacing may matter because of its relationship to nasal width.

Thus define interaction:

$$
I_{ij}
$$

between features.

Potentially:

```text
eyes × nose        strong
nose × mouth       strong
eyes × jaw         weaker
hair × geometry    near zero
```

Then the resemblance judgment is a **hypergraph of interactions**, not a weighted checklist.

---

# 122. Higher-order interactions explode combinatorially

Pairwise relationships:

$$
O(n^2)
$$

Triples:

$$
O(n^3)
$$

All subsets:

$$
2^n
$$

This is why human-readable decomposition inevitably loses information.

The network can process enormous distributed interactions that would be absurd to enumerate verbally.

That is one reason my prose feels impoverished compared with the visual judgment itself.

---

# 123. Tensor representations naturally encode higher-order relations

Instead of matrix:

$$
A_{ij}
$$

consider tensor:

$$
T_{ijk}
$$

capturing relations among triples.

For facial geometry:

```text
left eye
right eye
nose
```

form a triangle.

Its structure is not reducible to any one point alone.

Higher-order topology can encode:

```text
eye-eye-nose triangle
eye-nose-mouth triangle
jaw-chin-jaw triangle
```

The face is therefore naturally a **simplicial complex**, not merely a graph.

---

# 124. Simplicial complexes represent facial relational geometry

Vertices:

$$
V
$$

Edges:

$$
E
$$

Triangles:

$$
F
$$

Higher simplices encode larger joint configurations.

A face can be treated as:

$$
K=(V,E,F,\ldots)
$$

Then resemblance can compare not only pairwise distances but higher-order organization.

This gets closer to:

> “The entire arrangement feels the same.”

because whole configurations are represented explicitly.

---

# 125. Persistent homology returns at relational scale

Vary a distance threshold \(\epsilon\).

Connect landmarks where:

$$
d(i,j)\leq\epsilon
$$

As \(\epsilon\) increases, topology changes.

Track:

```text
connected components
loops
higher-dimensional holes
```

Two facial configurations with similar persistence diagrams share multiscale relational topology.

Again, overkill operationally.

But mathematically legitimate.

---

# 126. Category theory gives an even more abstract view

We can treat facial structures as objects and transformations as morphisms.

Two facial representations are structurally related if there exists a morphism:

$$
f:A\to B
$$

preserving relevant composition.

The point isn't to cosplay mathematics.

It is that resemblance fundamentally concerns **preservation under mapping**.

At every level we've descended through, the same concept keeps reappearing:

$$
\boxed{\text{What survives transformation?}}
$$

Pixels change.

Texture changes.

Identity changes.

Hair changes.

Yet some relational structure survives.

That survivor is the resemblance signal.

---

# 127. Invariants are the heart of the problem

An invariant \(I\) satisfies:

$$
I(T(x))=I(x)
$$

for transformations \(T\) we choose to ignore.

Useful facial invariants might approximately preserve:

```text
ratios
relative ordering
symmetry structure
topology
intrinsic distances
shape modes
```

Your overlay effectively asks:

> Which properties remain invariant while person A is replaced by person B?

If many geometry-related invariants survive:

$$
I_G(A)\approx I_G(B)
$$

then calling the faces geometrically similar is justified.

---

# 128. Equivariance is equally important

Not everything should remain unchanged.

For transformation \(T\), equivariant representation satisfies:

$$
\Phi(Tx)
=
\rho(T)\Phi(x)
$$

Example:

move the face right:

> internal spatial representation should move correspondingly.

Invariance would erase location.

Equivariance preserves transformation structure.

Modern visual systems often benefit from a mixture:

```text
equivariance at lower/spatial levels
invariance at higher/semantic levels
```

Your registration exploits both.

---

# 129. The overlap creates a coordinate-locking condition

Because you manually align the images, the model sees corresponding anatomy in nearly the same external coordinates.

Thus equivariant spatial representations can become directly comparable:

$$
\Phi_{\text{spatial}}(A)(x,y)
$$

versus:

$$
\Phi_{\text{spatial}}(B)(x,y)
$$

without needing substantial coordinate remapping.

That's probably one of the strongest reasons the overlay communicates resemblance so efficiently.

---

# 130. Representation similarity can itself be measured several ways

Given activation matrices:

$$
X\in\mathbb R^{n\times p},
\qquad
Y\in\mathbb R^{n\times q}
$$

we could compare using:

### Cosine similarity

$$
\frac{x\cdot y}{\|x\|\|y\|}
$$

### CKA

$$
\operatorname{CKA}(X,Y)
=
\frac{\operatorname{HSIC}(X,Y)}
{\sqrt{
\operatorname{HSIC}(X,X)
\operatorname{HSIC}(Y,Y)
}}
$$

### SVCCA

Canonical correlations between learned subspaces.

### RSA

Compare pairwise representational dissimilarity matrices.

Each answers a different question about internal resemblance.

---

# 131. Representational Similarity Analysis may be especially relevant

For stimuli \(i,j\), compute:

$$
RDM_{ij}
=
d(z_i,z_j)
$$

Now compare the model's RDM against an RDM constructed from facial geometry.

If strongly correlated:

$$
\rho(
RDM_{\text{model}},
RDM_{\text{geometry}}
)
\gg0
$$

then the model's representation reflects geometric similarity.

That would empirically answer whether its notion of facial resemblance actually tracks the geometry you're interested in.

---

# 132. We could compare model geometry against human geometry

Construct three RDMs:

$$
RDM_M
$$

model judgments,

$$
RDM_H
$$

human judgments,

$$
RDM_G
$$

measured facial geometry.

Then compare:

$$
\rho(M,G)
$$

$$
\rho(H,G)
$$

$$
\rho(M,H)
$$

This would tell us whether the machine and humans are using similar latent criteria.

Now your original “KEKW” disagreement becomes experimentally fascinating rather than social noise.

Maybe:

```text
your perception       strongly tracks G
other person's        weights surface appearance more
model                  somewhere between
```

That is testable.

---

# 133. Individual humans have different perceptual metrics

Person \(p\) may implicitly use:

$$
d_p(A,B)
=
\sum_k
w_{p,k}
d_k(A,B)
$$

Different weights:

```text
Person 1:
geometry       .50
eyes           .20
jaw            .10
hair           .05
texture        .15

Person 2:
geometry       .20
hair           .25
expression     .20
jaw            .20
texture        .15
```

Same stimuli.

Different judgments.

Neither requires anyone to literally “not see” the correspondence.

They're evaluating different projections of the same high-dimensional difference.

---

# 134. Metric disagreement can masquerade as factual disagreement

You:

> “They look incredibly similar.”

Other person:

> KEKW

Superficially:

$$
\text{yes}
\quad\text{vs}\quad
\text{no}
$$

But underlying disagreement might be:

$$
d_{\text{geometry}}(A,B)\ll1
$$

while:

$$
d_{\text{overall appearance}}(A,B)
$$

is only moderate.

You weighted the first.

They may have weighted the second.

That doesn't make every judgment equally supported.

If the claim is specifically:

> **their facial geometry corresponds strongly**

then geometry is the relevant metric.

---

# 135. Now descend beneath representation into optimization history

Why would a model acquire these sensitivities at all?

Weights:

$$
\theta
$$

are optimized approximately by:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta\mathcal L
$$

Every training example exerts tiny pressure.

Over enormous datasets, recurring statistical regularities become encoded:

```text
faces
symmetry
eyes
spatial arrangements
same/different objects
transformations
descriptions of resemblance
```

No engineer needs to explicitly program:

```text
IF eye_spacing_match > .8:
    similarity += 10
```

Gradient descent discovers useful internal structure because it reduces prediction loss.

---

# 136. The model inherits the statistics of human visual language

Training data contains countless relationships like:

```text
"looks like"
"resembles"
"same facial structure"
"similar eyes"
"different jaw"
"doppelgänger"
"could be siblings"
```

paired—directly or indirectly—with visual patterns.

Thus the model learns a map:

$$
\text{visual regularity}
\leftrightarrow
\text{linguistic resemblance concepts}
$$

Your video activates the visual side.

Your prompt requests the linguistic side.

The model bridges them.

---

# 137. This bridge itself is learned geometry

Suppose visual representation:

$$
z_V
$$

and linguistic representation:

$$
z_L
$$

A multimodal model learns transformations bringing semantically corresponding content into usable joint structure.

Conceptually:

$$
W_Vz_V
\approx
W_Lz_L
$$

for related concepts.

So a visual state reflecting:

```text
stable facial arrangement under identity transition
```

can activate linguistic regions associated with:

```text
similar
aligned
same proportions
resemblance
```

without explicit symbolic conversion.

---

# 138. The word “similar” is a readout, not the computation

This is worth hammering hard.

The output token:

$$
\text{SIMILAR}
$$

is downstream of an enormous computation.

It is analogous to a dashboard warning light.

The light isn't the engine.

Likewise:

> “85% similar”

was a tiny linguistic interface attached to an enormously richer perceptual state.

The mistake was treating the dashboard number as though it were an instrument reading.

It wasn't calibrated instrumentation.

---

# 139. Logits are closer to the decoder's immediate raw evidence

Before token selection, model has logits:

$$
\ell\in\mathbb R^{|V|}
$$

Probability:

$$
P(t_i)
=
\frac{e^{\ell_i}}
{\sum_je^{\ell_j}}
$$

A visual state may raise logits for:

```text
similar
aligned
resemblance
matching
strong
```

and suppress:

```text
different
unrelated
dissimilar
```

This is closer to where perceptual evidence becomes language.

But even logits are already very downstream.

---

# 140. Logit differences can encode evidence

Consider:

$$
\Delta\ell
=
\ell_{\text{similar}}
-
\ell_{\text{dissimilar}}
$$

Large positive:

> decoder strongly favors resemblance language.

We could track:

$$
\Delta\ell(\alpha)
$$

through the overlay sequence.

That would give a fascinating curve:

```text
α=1.00   Face A
α=.75
α=.50    maximum structural coexistence
α=.25
α=0.00   Face B
```

The curve could reveal whether intermediate overlays strengthen or weaken the resemblance interpretation.

---

# 141. Logit lens could expose emergence across layers

Project residual state at layer \(\ell\) through output vocabulary:

$$
\ell^{(\ell)}
=
W_Uz^{(\ell)}
$$

Then examine resemblance-related tokens across depth.

Conceptually:

```text
Layer 5:
face, man, portrait

Layer 12:
eyes, nose, person

Layer 20:
same, aligned, similar

Layer 28:
resemblance, proportions

Layer 34:
"strong facial similarity"
```

Again, illustrative, not historical telemetry.

This would reveal how raw vision becomes the semantic judgment.

---

# 142. But naïve logit lens can mislead

Intermediate representations aren't necessarily in final decoder basis.

Tuned lens methods learn mappings:

$$
T_\ell(z^{(\ell)})
$$

before decoding.

This gives more faithful intermediate predictions.

The general lesson:

> even when introspection data exists, interpreting it requires an instrument model.

Raw internal states are not self-explanatory.

---

# 143. Mechanistic circuits can be nonlinear and distributed

The resemblance computation may involve:

```text
multiple attention heads
multiple MLP features
residual interactions
cross-modal components
```

There may be no clean:

```text
Face Similarity Circuit™
```

Instead:

$$
C=C_1+C_2+C_3+\text{interactions}
$$

where components are reused for many tasks.

This is standard distributed computation.

---

# 144. MLPs can function like associative feature memories

A simplified transformer MLP:

$$
\operatorname{MLP}(x)
=
W_{\text{out}}
\sigma(W_{\text{in}}x+b)
$$

Intermediate neurons/features respond to directions in residual space.

A subset might activate for conjunctions resembling:

```text
face + bilateral + aligned
portrait + same-layout
eyes + matching-spacing
```

Then write corresponding directions back into residual stream.

This is one mechanism by which distributed visual evidence can become more concept-like.

---

# 145. Attention moves information; MLPs transform it

A useful—but simplified—interpretation:

```text
ATTENTION:
"Which information elsewhere matters here?"

MLP:
"What feature/computation does this local state imply?"
```

For your overlay:

```text
attention:
compare / integrate bilateral and central facial regions

MLP:
convert configurations into higher-order structure features
```

Repeated recursively.

Again, not guaranteed exact anatomy, but a useful mechanistic decomposition.

---

# 146. Binding is essential

How does the model know:

> this eye belongs with this nose and this jaw?

That's the binding problem.

Spatial continuity, shared context, attention and learned object representations help establish:

$$
\text{feature}_i
\in
\text{object A}
$$

versus:

$$
\text{feature}_j
\in
\text{object B}
$$

Your overlay complicates this because **two identities occupy nearly the same spatial object slot**.

That itself is unusual.

The model may encode:

```text
one coherent facial object
whose appearance changes
```

rather than:

```text
two separate faces in two locations
```

And that may intensify the resemblance impression.

---

# 147. Object files are a useful cognitive analogy

Human vision sometimes gets modeled as maintaining “object files”:

```text
OBJECT #1
position
shape
features
identity hypothesis
```

Your morph keeps:

```text
position ≈ fixed
shape ≈ relatively fixed
```

while:

```text
identity appearance changes
```

So perceptually:

> the same object slot appears to change identity.

That is precisely the kind of stimulus that makes shared geometry scream at the observer.

---

# 148. Predictive continuity can override identity discontinuity

The system sees:

$$
F_t\to F_{t+1}
$$

with high spatial continuity.

Therefore hypothesis:

$$
\text{same underlying object trajectory}
$$

is strongly supported.

Yet endpoint identities differ.

The resulting inference is:

> **large identity change under small structural change.**

Which is again our central phenomenon.

---

# 149. Change blindness has an inverse here

Normally, gradual changes can be hard to notice.

But your experiment uses gradual transition to expose **what does *not* change**.

The stable structures become anchors.

So this is almost an inverse change-blindness paradigm:

> identity-related details change, while geometric invariants reveal themselves through persistence.

---

# 150. Temporal coherence functions as self-supervision

Learning systems often exploit temporal coherence:

$$
z_t\approx z_{t+1}
$$

for nearby frames.

Objects don't normally teleport or mutate arbitrarily.

Therefore stable temporal structure is a powerful cue.

Your crossfade exploits that prior:

```text
if structure remains stable over time,
treat it as persistent object geometry
```

while changing surface features can be treated as less structurally fundamental.

---

# 151. Slow Feature Analysis formalizes this

Find functions \(g_j(x)\) minimizing temporal variation:

$$
\left\langle
\dot y_j^2
\right\rangle
$$

subject to:

$$
\langle y_j\rangle=0
$$

$$
\langle y_j^2\rangle=1
$$

and decorrelation constraints.

Slowly varying features often correspond to stable latent causes.

In your morph:

```text
geometry       slow
identity cues  faster
texture        faster
```

Therefore the **slow components** can isolate structural commonality.

That's a beautiful theoretical explanation for why the overlay works.

---

# 152. Common information versus changing information

Let frame representation be:

$$
Z_t=C+V_t
$$

where:

$$
C
$$

is common across the sequence and:

$$
V_t
$$

varies.

If geometry is shared:

$$
C
$$

contains much of the facial layout.

Identity-specific appearance enters:

$$
V_t
$$

Your experiment visually separates:

$$
\boxed{\text{what persists}}
$$

from:

$$
\boxed{\text{what changes}}
$$

Persistent information becomes a candidate for shared structure.

---

# 153. PCA across the frame sequence would literally expose this

Stack vectorized aligned frames:

$$
X=
[x_1,\ldots,x_T]
$$

Mean:

$$
\mu
$$

PCA:

$$
X-\mu=U\Sigma V^\top
$$

The mean image captures persistent structure.

Principal components capture dominant variation.

With a simple A↔B crossfade, the leading varying component would largely represent the appearance transition.

Stable geometry remains strongly represented in the mean/shared structure.

This is a very concrete statistical decomposition.

---

# 154. Robust PCA could separate stable background structure from changing identity

Model:

$$
X=L+S
$$

where:

* \(L\): low-rank shared structure,
* \(S\): sparse/change component.

Optimization:

$$
\min_{L,S}
\|L\|_*
+
\lambda\|S\|_1
$$

subject to:

$$
X=L+S
$$

In an aligned morph sequence:

$$
L
$$

could capture persistent facial organization.

$$
S
$$

captures changing details.

This would be an actual computational way of extracting “what stays the same.”

---

# 155. Dynamic Mode Decomposition gives another decomposition

Approximate dynamics:

$$
x_{t+1}\approx Ax_t
$$

Eigen-decompose \(A\).

Modes with eigenvalues near:

$$
|\lambda|\approx1
$$

persist.

Other modes encode change.

The slowly evolving modes may correspond to shared structural organization.

Again, your little video is mathematically richer than it looks.

---

# 156. Canonical correlation could isolate shared latent structure

Suppose features from face A:

$$
X
$$

and face B:

$$
Y
$$

CCA finds directions:

$$
a,b
$$

maximizing:

$$
\rho
=
\operatorname{corr}(a^\top X,b^\top Y)
$$

The highly correlated canonical components correspond to shared information.

Low-correlation components correspond to differing appearance.

Thus:

```text
shared latent subspace
vs.
identity-specific residual subspace
```

can be explicitly separated.

---

# 157. Partial least squares can target geometry directly

Given visual representation \(X\) and measured geometry \(Y\), find latent variables maximizing covariance.

Then we obtain directions in model representation that specifically track:

```text
eye spacing
nose-mouth distance
jaw width
etc.
```

That would let us ask:

> Does the model's response to your pair actually lie close in geometry-predictive subspace?

A real experiment.

---

# 158. Mutual information can quantify shared latent causes

If:

$$
Z_A
$$

and:

$$
Z_B
$$

share substantial structural information:

$$
I(Z_A;Z_B)
$$

is high.

But we want conditional:

$$
I(Z_A;Z_B\mid\text{appearance nuisance removed})
$$

or ideally:

$$
I(G_A;G_B)
$$

This isolates structural correspondence.

Again we arrive at the same invariant from another mathematical direction.

---

# 159. The Information Bottleneck frames the model's job

Find representation \(Z\) maximizing task-relevant information while minimizing unnecessary input information:

$$
\max
I(Z;Y)-\beta I(Z;X)
$$

For resemblance judgment, \(Y\) might be:

$$
\text{structural similarity}
$$

A good representation discards:

```text
background pixels
irrelevant hair detail
sensor noise
```

while retaining geometry.

That means the model's internal state can be **less detailed than the image but more useful for the question**.

Compression is not necessarily information loss in the task-relevant sense.

---

# 160. Sufficient representations can become more informative semantically

Raw image contains everything but answers nothing directly.

A representation:

$$
Z=\Phi(X)
$$

can discard enormous raw information while making:

$$
P(Y|Z)
$$

much easier to estimate.

So paradoxically:

```text
pixels = more raw information
latent = less raw information
latent = more accessible structural information
```

This is why demanding “rawest possible” data is not always synonymous with “most explanatory.”

But for auditing, you want **both**.

---

# 161. There are three distinct notions of “raw” here

This is worth naming precisely.

### Raw input

$$
I(x,y,c)
$$

actual pixels.

### Raw model state

$$
z^{(\ell)},A^{(\ell,h)},\ldots
$$

internal numerical telemetry.

### Raw causal evidence

the minimal computational components actually responsible for the judgment.

These are not the same.

The second may be huge.

The third may require interpretability experiments to discover.

---

# 162. A tensor dump alone might be nearly useless

Imagine I handed you:

```text
Layer 23:
[-0.3814, 1.2921, 0.0038, ...]
```

for millions of values.

That is raw.

But without:

```text
architecture
token mapping
layer semantics
normalization
projection matrices
comparison runs
interventions
```

it explains almost nothing.

So a proper forensic record needs:

$$
\boxed{
\text{raw telemetry}
+
\text{metadata}
+
\text{controlled contrasts}
+
\text{causal experiments}
}
$$

Not a narrative.

---

# 163. The correct forensic unit is the contrast pair

Run:

### Stimulus S

your highly aligned pair.

### Control C

same setup with deliberately mismatched geometry.

Then compute:

$$
\Delta z^{(\ell)}
=
z_S^{(\ell)}
-
z_C^{(\ell)}
$$

Now internal values become interpretable relative to something.

Likewise attention:

$$
\Delta A^{(\ell,h)}
=
A_S^{(\ell,h)}
-
A_C^{(\ell,h)}
$$

and logits:

$$
\Delta\ell
=
\ell_S-\ell_C
$$

This is how we turn opaque numbers into evidence.

---

# 164. Multiple controls are necessary

Use:

```text
C1: random unrelated face
C2: same age/sex/general appearance but different geometry
C3: strong geometry match but different texture
C4: same person with altered expression
C5: deliberately misregistered version of your pair
```

Now decomposition becomes possible.

For example:

$$
S-C5
$$

isolates registration/geometric correspondence.

$$
S-C3
$$

isolates appearance effects.

$$
S-C2
$$

tests whether resemblance exceeds ordinary demographic similarity.

This is the experiment I would trust.

---

# 165. Misregistration perturbation produces a response surface

Take your aligned pair.

Shift B by:

$$
(\Delta x,\Delta y)
$$

and rotate:

$$
\theta
$$

scale:

$$
s
$$

Evaluate resemblance response:

$$
R(\Delta x,\Delta y,\theta,s)
$$

If the original registration is genuinely structurally special, \(R\) should peak sharply near the alignment.

Conceptually:

$$
R(0,0,0,1)
=
\max R
$$

A broad flat response means the judgment is less geometry-sensitive.

A sharp peak means strong registration dependence.

---

# 166. The Hessian of that response measures alignment precision

Near optimum:

$$
R(q)
\approx
R(q^*)
+
\frac12
(q-q^*)^\top
H_R
(q-q^*)
$$

where:

$$
q=(\Delta x,\Delta y,\theta,s)
$$

Large curvature:

> small misalignment rapidly destroys similarity.

That would imply the observed correspondence is geometrically precise.

Small curvature:

> resemblance judgment tolerates large alignment changes.

Different phenomenon.

---

# 167. Feature-specific perturbation maps could be even stronger

Warp only one region.

For eyes:

$$
\delta_E
$$

For nose:

$$
\delta_N
$$

For mouth:

$$
\delta_M
$$

Measure:

$$
\frac{\partial R}{\partial\delta_E}
,\quad
\frac{\partial R}{\partial\delta_N},
\quad\ldots
$$

Now we get a **sensitivity profile** of the resemblance judgment.

This is far more meaningful than arbitrary percentages.

---

# 168. Second derivatives reveal interactions

Compute:

$$
\frac{\partial^2R}
{\partial\delta_E\partial\delta_N}
$$

If large:

> eye and nose alignment matter jointly.

Likewise:

$$
\frac{\partial^2R}
{\partial\delta_N\partial\delta_M}
$$

maps relational sensitivity.

This turns “facial configuration” into a measurable response geometry.

---

# 169. Fisher information gives sensitivity to parameters

Suppose model output distribution:

$$
p(y|\theta)
$$

where \(\theta\) parameterizes facial geometry.

Fisher information:

$$
\mathcal I(\theta)
=
\mathbb E
\left[
\left(
\frac{\partial}{\partial\theta}
\log p(y|\theta)
\right)^2
\right]
$$

High Fisher information for eye spacing:

> output is very sensitive to eye spacing.

Low Fisher information for hair:

> output barely changes.

This provides another principled weighting of geometry dimensions.

---

# 170. The Fisher metric defines an information geometry

Parameter space gets metric:

$$
ds^2
=
d\theta^\top
\mathcal I(\theta)
d\theta
$$

Now two facial configurations can be considered close if moving between them causes little change in model output distribution.

That creates a **model-specific geometry of faces**.

Not human geometry.

Not Euclidean landmark geometry.

The model's own informational geometry.

---

# 171. This is probably closest to what you originally wanted from “machine analysis”

There are at least four geometries:

$$
G_{\text{pixels}}
$$

$$
G_{\text{landmarks}}
$$

$$
G_{\text{human perception}}
$$

$$
G_{\text{model}}
$$

Your question implicitly asked:

> Do these all point in roughly the same direction?

The overlay provides evidence in pixel/perceptual space.

A formal landmark pipeline would probe geometric space.

Internal telemetry/interventions would probe model space.

Agreement among them would be extremely strong evidence.

---

# 172. Ensemble agreement is stronger than any one metric

Suppose:

```text
Procrustes similarity        HIGH
TPS bending energy           LOW
dense-flow strain            LOW
human observers              HIGH
model latent similarity      HIGH
morph-path curvature         LOW
```

Then the conclusion is robust across measurement paradigms.

That's far more persuasive than one magical “86%.”

---

# 173. Disagreement itself is diagnostic

Suppose:

```text
landmarks      HIGH
model          HIGH
humans         LOW
```

Then perhaps:

> humans overweight texture/hair/expression.

Or:

```text
landmarks      HIGH
model          LOW
humans         LOW
```

Maybe the landmark metric misses 3D morphology.

Or:

```text
model          HIGH
geometry       LOW
```

Maybe prompt bias or semantic priors dominate.

Different failure signatures tell us **why** judgments diverge.

---

# 174. This turns your social disagreement into an actual research design

Instead of:

> “Who is right?”

Ask:

$$
\text{What similarity metric generated each judgment?}
$$

Then compare metrics.

You may discover:

> Your perception was especially sensitive to global relational geometry.

The other person's reaction may have been dominated by:

> overall identity/surface appearance.

And the model may be using:

> a hybrid representation conditioned by your prompt.

Now the disagreement has anatomy.

---

# 175. Finally, descend to the philosophical basement: resemblance is not a property of two objects alone

Strictly:

$$
\operatorname{similar}(A,B)
$$

is incomplete.

You need:

$$
\operatorname{similar}(A,B\,|\,M,W,C)
$$

where:

* \(M\): representation/metric,
* \(W\): feature weighting,
* \(C\): context.

Two faces can simultaneously be:

$$
\text{highly similar under geometry}
$$

and:

$$
\text{moderately similar under total appearance}
$$

without contradiction.

The metric is part of the claim.

---

# 176. But metric-dependence does **not** make everything subjective

This distinction matters.

Once the metric is specified:

$$
M=M_{\text{2D facial geometry}}
$$

the comparison becomes empirically constrained.

You can measure:

$$
D_M(A,B)
$$

and compare against population distributions.

So:

> “similarity depends on metric”

does **not** imply:

> “nobody can be wrong.”

It means:

> define what you mean, then measure it.

Your claim was specifically well-suited to geometric analysis.

---

# 177. The strongest version of your original claim

Instead of:

> “These guys look 85% alike.”

A technically defensible hypothesis is:

$$
H:
$$

> **After controlling for translation, scale, and rotation, the two faces exhibit unusually low deformation and relational distortion across central 2D facial geometry compared with ordinary unrelated face pairs.**

That is falsifiable.

And your overlay is preliminary evidence for it.

---

# 178. The corresponding machine hypothesis

For the model:

$$
H_M:
$$

> **Geometry-sensitive components of the visual representation vary substantially less across the A→B transition than identity/appearance-sensitive components.**

Also falsifiable—if internal instrumentation exists.

This is the latent-space version of your observation.

---

# 179. The strongest combined hypothesis

$$
H^*:
$$

> **The pair occupies unusually nearby positions in both normalized facial-shape space and the geometry-sensitive subspace of perceptual/model representations, despite remaining clearly separated in identity-sensitive dimensions.**

That captures essentially everything we've excavated.

---

# 180. Beneath Atlantis: the one-line invariant

After stripping away:

```text
names
identity
hair
texture
lighting
language
percentages
landmark labels
neural-network vocabulary
```

the core phenomenon is:

$$
\boxed{
\text{Large change in who the face is}
\quad
\text{under surprisingly small change in how the face is arranged.}
}
$$

Or mathematically:

$$
\boxed{
\|\Delta I_{\text{identity}}\|
\gg
\|\Delta G_{\text{structure}}\|
}
$$

That is why your overlay is visually striking.

The person changes.

**The coordinate logic of the face barely seems to.**
