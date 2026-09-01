# AMME5710-Assignment-1

## Question 1 — Photometric Stereo

Reconstructing a 3D height map of a face from 64 images of it taken under 64
known lighting directions. Run with `python3 mainQ1.py`.

The pipeline is three stages: recover a surface *normal* at every pixel from the
images, convert those normals into surface *gradients*, then *integrate* the
gradients into a height map.

---

### Step 0 — Images and lighting to albedo and surface normals

Each face is assumed **Lambertian**: it scatters light equally in all
directions, so its apparent brightness depends on the angle between the surface
and the light, but not on where the camera is. For a pixel with albedo (its
intrinsic reflectivity) `ρ` and unit surface normal `n`, lit by a unit lighting
direction `l_k`, the measured brightness is

```
I_k = ρ (n · l_k)
```

This is non-linear in the unknowns, because `ρ` multiplies `n` and `n` is
constrained to unit length. The standard trick is to absorb the albedo into the
normal by defining

```
g = ρ n
```

so that the model becomes **linear** in `g`:

```
I_k = l_k · g
```

Each of the 64 images gives one such equation for the pixel, and `g` has only 3
components. So each pixel supplies a 64×3 over-determined system, solved in the
least-squares sense. Because the lighting directions are shared by every pixel,
the same 64×3 matrix `L` applies everywhere, and all 192×168 = 32,256 pixels can
be solved in a single call:

```python
g, *_ = np.linalg.lstsq(light_dirs, I, rcond=None)   # (64,3) \ (64,N) -> (3,N)
```

The albedo and normal are then recovered by splitting `g` back into its
magnitude and direction, using the fact that `n` is a unit vector:

```
ρ = |g|          n = g / |g|
```

Only `B01` ships with pre-computed albedo and normals; `B02`, `B05` and `B07`
provide images and lighting directions only, which is why this stage is
implemented rather than loaded. As a check, the solve is run on B01 and compared
against the supplied pickle — it agrees to **7.8e-16** in albedo and **1.7e-15**
in the normals, i.e. to machine precision. This matters because it isolates the
error: anything wrong with the final height maps comes from the integration
stage or from the Lambertian assumption itself, not from the linear solve.

The division by `|g|` is guarded against zero albedo. The guard never fires here
(the smallest albedo across the four datasets is 0.019) but a zero would produce
`nan` silently rather than failing loudly.

---

### Step 1 — Surface normals to surface gradients

Normals alone are not enough; integration needs *slopes*. Model the face as a
height function over the image plane, a so-called **Monge surface**:

```
z = f(x, y)
```

Walking one step in `x` moves along the tangent vector `(1, 0, ∂z/∂x)`, and one
step in `y` along `(0, 1, ∂z/∂y)`. Both lie in the surface's tangent plane, so
the normal is perpendicular to both — their cross product:

```
        | i  j  k  |
n  ∝    | 1  0  z_x |  =  (−z_x, −z_y, 1)
        | 0  1  z_y |
```

This is proportional to, not equal to, the unit normal, but that is enough: the
ratios of the components are independent of the scale factor. Writing
`p = ∂z/∂x` and `q = ∂z/∂y` and taking ratios against the third component:

```
p = ∂z/∂x = −n_x / n_z          q = ∂z/∂y = −n_y / n_z
```

This is the key inversion — it turns a field of directions into a field of
slopes, which is something that can be integrated.

**Sign convention.** With the textbook signs above, these particular normals
reconstruct the face *inside-out*: the nose comes out as a pit rather than a
peak. Both gradients are therefore flipped in the code:

```python
p = surface_normals[:, :, 0] / nz
q = surface_normals[:, :, 1] / nz
```

Flipping *both* is simply `z → −z`, so this is not a fudge on one axis — it
means the supplied normals are expressed relative to a height axis pointing
*away* from the camera rather than toward it. This was confirmed empirically:
for each of the four sign combinations, the best-fit plane was removed and the
highest remaining point located. Only this combination puts the peak on the nose
(row 87, column 82 for B01); the other three put it in an image corner. The
result is consistent across all four subjects.

**Why there is no `n_z` guard.** `p` and `q` blow up as `n_z → 0`, which is the
usual reason to clamp it. But `n_z = 0` means the surface is exactly edge-on to
the camera — a silhouette — and such a patch is by definition invisible, so it
cannot appear inside a tight face crop. Measured over all four datasets, the
smallest `|n_z|` is **0.134** (B05, about 82° from head-on), far from the
singularity, so the division is safe unguarded.

---

### Step 2 — Integrating the gradients into a height map

`p` and `q` give the *rate of change* of height, so height itself is recovered
by accumulating those changes along a path from a chosen reference pixel:

```
z(x, y) = z(x₀, y₀) + ∫_C ( p dx + q dy )
```

The assignment fixes the top-left pixel as the start point with height 0. That
choice does not measure anything — it just pins down the arbitrary constant of
integration. Photometric stereo recovers only **relative** shape; the absolute
distance from the camera is unrecoverable, because scaling the whole scene
toward or away from the camera changes no image.

**Discretisation.** With pixels one unit apart, `dx = dy = 1`, so each line
integral collapses into a running total — a cumulative sum, `np.cumsum`.

**Why three different paths?** For a genuine surface the gradient field is
*conservative*, which requires the **integrability constraint**

```
∂p/∂y = ∂q/∂x
```

i.e. the mixed second partials of `z` commute. When this holds, the integral
depends only on the endpoints and every path gives the same answer. Real normals
come from noisy measurements of a face that is not perfectly Lambertian, so the
constraint is violated, and different paths genuinely disagree. Comparing paths
is therefore a direct measure of how much the data has broken the model.

Three strategies are computed, each an L-shaped path from the top-left corner:

**(a) Along the top row first, then down each column.**

```python
z_a = np.zeros_like(p)
z_a[0, 1:] = np.cumsum(p[0, 1:])
z_a[1:, :] = z_a[0, :] + np.cumsum(q[1:, :], axis=0)
```

**(b) Down the first column first, then across each row.**

```python
z_b = np.zeros_like(q)
z_b[1:, 0] = np.cumsum(q[1:, 0])
z_b[:, 1:] = z_b[:, 0:1] + np.cumsum(p[:, 1:], axis=1)
```

**(c) The average of the two.**

```python
z_c = 0.5 * (z_a + z_b)
```

**The index-1 detail.** Every cumulative sum starts at index **1**, not 0. Index
0 would include the reference pixel's own gradient in its own height, so
`z(0,0) ≠ 0`. In strategy (a) the damage is worse than a harmless global offset:
it would add the whole top row of `q` as a **per-column bias**, distorting the
shape rather than just shifting it. With the slicing as written, `z(0,0) = 0`
exactly for all three strategies.

**Why the streaks appear, and why they run the way they do.**

Write out strategy (a) explicitly for a pixel at row `r`, column `c`:

```
z_a[r, c]  =  Σ_{j=1..c} p[0, j]   +   Σ_{i=1..r} q[i, c]
              ─────────────────       ─────────────────
              along the top row       straight down column c
```

The second term is the important one: it is a cumulative sum **confined to
column `c`**. Nothing in column `c` is ever compared against column `c ± 1`.
The only thing the columns share is their single starting value from the top
row. So the image is not really being integrated as a surface at all — it is
192 independent one-dimensional integrations, run side by side, that happen to
start from a common row.

That has two consequences.

*Down a column, the result is smooth.* Consecutive entries of a cumulative sum
differ by exactly one gradient step, which is small.

*Across columns, it is not.* Each column accumulates its own noise
independently, so neighbouring columns drift apart and never get pulled back
together — there is no mechanism in the algorithm that couples them. A
cumulative sum is an integrator, so white noise in the gradients becomes a
random walk in the height, and adjacent columns are performing *different*
random walks.

Smooth along the direction of integration, discontinuous across it, is exactly
what a stripe is. Hence strategy (a), which integrates down columns, produces
**vertical** streaks. Strategy (b) integrates along rows and gives
**horizontal** streaks for the mirror-image reason.

This is measurable. For B01, taking the RMS height step between neighbouring
pixels:

| | step down a column | step across a row | ratio |
|---|---|---|---|
| `z_a` (row first, then down) | 0.414 px | 0.819 px | **2.0×** |
| `z_b` (column first, then across) | 0.639 px | 0.603 px | 0.9× |

`z_a` is twice as rough across a row as it is down a column — the surface is
continuous in the direction it was integrated and jumps sideways. `z_b` shows
the reverse, though more weakly.

The random-walk picture also predicts the streaks should worsen with distance
from the seed row, since a random walk's spread grows as `√r`. The trend is
there, but slower than `√r`:

```
z_a: RMS difference between adjacent columns
   row  10:  0.65 px      row 100:  0.95 px
   row  50:  0.52 px      row 190:  1.03 px
```

The growth is real but sub-`√r` because the gradient errors are not white noise
— they are spatially correlated, coming from shared structure (shadow edges,
the specular ridge of the nose) that affects neighbouring columns similarly and
partly cancels in the difference.

Because the two artefacts are orthogonal, averaging in (c) cross-cancels them
and halves the noise variance, which is why (c) is used for the 3D renders.

A pronounced global tilt survives in every reconstruction (best-fit plane slopes
of 0.23–0.30 px/row). This is accumulated low-frequency drift from the running
sums, not real head geometry — a small systematic bias in the gradients is
integrated over ~190 steps. It matches the warp visible in the reference figures
in the assignment brief.

---

### Step 3 — Outlier detection

With `ρ` and `n` recovered for a pixel, and all 64 lighting directions known,
the Lambertian model can **predict** how bright that pixel should have been in
every image. The residual is measured minus predicted:

```
r_k(x,y) = I_k(x,y) − ρ(x,y) ( n(x,y) · l_k )
```

giving 64 residuals per pixel. Large ones mark observations the model cannot
explain — cast shadows, specular highlights on the nose and forehead,
inter-reflections around the eye sockets and nostrils: all the ways a real face
fails to be Lambertian. The `g = ρn` substitution from Step 0 reappears here,
used forwards instead of backwards, so the whole 64×192×168 prediction cube is
a single matrix product with no loops.

**The prediction is deliberately not clamped.** Physically the correct model is
`ρ·max(n·l, 0)`, because a negative `n·l` means the light is behind the surface
and the patch is unlit. But clamping defeats the purpose: inside a cast shadow
the camera records ≈0 and a clamped model *also* predicts ≈0, so the two agree,
the residual vanishes, and the shadow — the very thing being hunted — becomes
invisible. Left un-clamped the model keeps predicting a brightly lit pixel
against a measurement of zero, producing the large negative residual that flags
it.

**The threshold is per pixel, not global.**

```python
sigma = residuals.std(axis=0, keepdims=True)   # axis 0 = the 64 images
return np.abs(residuals) > 2.0 * sigma
```

Collapsing `axis=0` leaves one standard deviation per pixel, computed from that
pixel's own 64 residuals. This matters because residuals scale with albedo: a
bright forehead has larger residuals than dark hair simply because it reflects
more light, not because anything anomalous happened. A single global threshold
would mostly flag "the bright parts of the face". The per-pixel version asks
the right question instead — *is this observation unusual relative to the other
63 of this same pixel?*

**Montage.** One 8×8 figure per face, one panel per lighting condition. Flagged
pixels are drawn in red over the image itself rather than as a bare binary
mask, so it is possible to see *what* was rejected: shadow boundaries, the
specular ridge of the nose, the dark side of the face under grazing light. Many
exposures are very dark, so each panel is stretched to its own 99th percentile
for display only — every calculation uses the raw values.

---

### Step 3b — Dead frames in B02

Two B02 exposures failed outright:

| frame | peak brightness | mean |
|-------|-----------------|------|
| `image_028` | 0.067 | 0.0246 |
| `image_052` | 0.012 | 0.0007 |

This is a capture fault, not a lighting effect. Dark frames are *normal* here:
14 of the 64 lighting directions in every dataset point behind the subject
(`l_z < 0`, up to 127° off the camera axis). But every other back-lit frame
still peaks near 1.0 from a rim highlight — B01's `image_016` and B05's
`image_025` both reach ≈1.0 at the same `l_z = −0.604` where B02's `image_028`
caps at 0.067. Only these two never illuminate anything, and no other dataset
contains any.

They are therefore detected on **peak** brightness, not mean:

```python
def find_dead_frames(imgs, peak_thresh=0.25):
    peaks = imgs.reshape(imgs.shape[0], -1).max(axis=1)
    return peaks < peak_thresh
```

**They must be removed separately, before the residual test, because the 2σ
rule does not reliably catch them.** `image_028` is caught (59.5% of its pixels
flagged, rank 3 of 64). `image_052` is not: only **4.9% flagged, rank 26 of
64**, barely above the 1.6% median. Its `l_z = −0.087` is nearly perpendicular
to the camera axis, so the model already predicts almost no brightness for a
front-facing pixel — measured ≈0, predicted ≈0, they agree, no residual. A
failed capture whose lighting direction happens to predict darkness anyway is
invisible to a residual test, while still biasing the least-squares solve with
a row of zeros. The dead-frame mask is OR-ed into the outlier mask so those
frames are excluded at every pixel.

---

### Step 4 — Re-solving without the outliers

In Step 0 every pixel used all 64 images, so every pixel shared the same 64×3
matrix `L` and one `lstsq` handled all 32,256 at once. After rejection **each
pixel keeps a different subset**, so there is no shared matrix any more — every
pixel has its own small least-squares problem.

Looping over 32,256 pixels would work but is slow. Instead the weighted normal
equations are formed, which are the closed-form solution of a weighted
least-squares problem:

```
A g = b        A = Σ_k w_k l_k l_kᵀ        b = Σ_k w_k I_k l_k
```

with `w_k = 0` for a rejected observation and 1 otherwise. The useful property
is that `A` is only 3×3 and each of its nine entries is a plain sum over `k`,
which vectorises into one matrix–vector product across all pixels:

```python
for i in range(3):
    for j in range(3):
        A[:, i, j] = (light_dirs[:, i] * light_dirs[:, j]) @ W
b = (W * I).T @ light_dirs
```

Nine products for the whole image. `np.linalg.solve` then handles a *stack* of
matrices natively, so all 32,256 3×3 systems are solved in one call.

A determinant check guards against pixels that lost too many views, or whose
survivors all cluster in one direction, leaving `A` singular; those fall back
to the unweighted solution rather than producing a blown-up normal that would
propagate down an entire column of the `cumsum`. In practice **zero pixels**
hit this on any of the four datasets — the fewest surviving views for any pixel
was 48 of 64.

**Measuring whether it helped.** There is no ground truth, so the disagreement
between the two integration paths is used as a proxy:

```python
rms = np.sqrt(np.mean((z_a - z_b) ** 2))
```

For a true surface the gradient field is conservative and paths (a) and (b)
must agree exactly; they diverge only to the extent the recovered normals
violate `∂p/∂y = ∂q/∂x`. So `RMS |z_a − z_b|` is a self-consistency score for
the normals — lower is more physically plausible.

---

### Results

**Baseline reconstruction.** `z(0,0) = 0` exactly for all three strategies on
all four subjects, as required.

| Subject | Height range of `z_c` |
|---------|-----------------------|
| B01     | 105.9 px              |
| B02     | 107.5 px              |
| B05     |  63.4 px              |
| B07     | 125.8 px              |

The B01 solve reproduces the supplied pre-computed pickle to **7.8e-16** in
albedo and **1.7e-15** in the normals, i.e. machine precision.

**Outlier rejection.**

| Subject | flagged | fewest surviving views | normals moved (mean / max) | singular pixels |
|---------|---------|------------------------|-----------------------------|-----------------|
| B01     | 12.0 %  | 48 | 4.30° / 25.1° | 0 |
| B02     | 12.8 %  | 50 | 4.51° / 37.6° | 0 |
| B05     | 10.9 %  | 49 | 4.15° / 41.7° | 0 |
| B07     | 12.2 %  | 50 | 4.15° / 30.5° | 0 |

The largest normal changes land on the eyes, the nostrils and the specular
ridge of the nose — exactly the regions where the Lambertian assumption is
weakest.

**Effect on path self-consistency**, `RMS |z_a − z_b|`:

| Subject | baseline | after rejection | change |
|---------|----------|-----------------|--------|
| B01 |  7.28 px |  7.38 px | **+1.3 %** |
| B02 | 22.10 px | 17.85 px | **−19.2 %** |
| B05 | 16.09 px | 17.39 px | **+8.1 %** |
| B07 | 11.94 px |  7.99 px | **−33.1 %** |

(B02's figure was −12.8 % from the 2σ rule alone; removing the two dead frames
as well took it to −19.2 %.)

**Rejection is not uniformly beneficial**, and this is worth stating plainly
rather than presenting it as a straight win. B07 and B02 improve substantially;
B01 and B05 get marginally worse. The likely explanation is a conditioning
trade-off. Discarding observations improves the quality of the data feeding
each pixel, but it also narrows the cone of surviving lighting directions,
which makes the 3×3 matrix `A` worse-conditioned and amplifies whatever noise
remains. Where the data was already fairly clean, the conditioning loss
outweighs the cleanup.

It is also worth remembering what this metric does and does not measure. It is
a *self-consistency* score — how nearly the recovered gradient field satisfies
the integrability constraint — not accuracy against a ground-truth face, of
which none is available. A reconstruction could in principle be smooth,
self-consistent and wrong.

B05 is the hardest dataset: roughly half the height range of the others, heavy
chin and neck shadowing that breaks the Lambertian assumption over a large
area, and the smallest `|n_z|` of the four (0.134) putting it closest to the
gradient singularity.

---

### A note on the alternative

The L-shaped paths above are what the assignment specifies, but they are not the
only option. The principled alternative is to stop insisting on any single path
and instead find the height map that best fits *all* the gradients at once, in a
least-squares sense — minimising

```
∬ (z_x − p)² + (z_y − q)² dx dy
```

Its Euler–Lagrange condition is a Poisson equation, `∇²z = ∂p/∂x + ∂q/∂y`,
solvable efficiently with an FFT (the Frankot–Chellappa method). Because it
weights every gradient equally rather than chaining them serially, it does not
streak and does not accumulate drift. It is mentioned here for contrast only;
the implementation follows the specified cumulative-sum strategies.

---

### Running

```bash
python3 mainQ1.py
```

Requires `numpy`, `opencv-python` and `matplotlib` only. Plots produced, in
order:

1. a 4×4 grid — one row per subject: albedo and the three height maps `z_a`,
   `z_b`, `z_c`;
2. four 8×8 outlier montages, one per subject, flagged pixels in red;
3. a 4×3 before/after grid — baseline `z_c`, outlier-rejected `z_c`, and the
   difference;
4. 3D renders via the supplied `plot_face_3d`, baseline then outlier-rejected
   for each face.
