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

### Step 2b — Rendering the model (from the week 2 tutorial)

Exercise 1 of the week 2 tutorial rendered the face under each of the 64 lighting
directions from the pre-supplied albedo and normals, and compared the render
against the photograph. `render_images()` is that exercise, with the double
`for` loop over all 32,256 pixels replaced by the single matrix product that
computes the same thing:

```python
g = (albedo[:, :, None] * normals).reshape(h * w, 3)   # rho * n
rendered = (light_dirs @ g.T).reshape(-1, h, w)        # all 64 images at once
```

`clamp=True` reproduces the tutorial exactly, flooring `n · l` at zero because a
negative dot product means the light is behind the surface. The measured and
rendered montages are plotted side by side for all four subjects.

The differences are the point of the exercise. The renders have no specular
highlight on the nose or forehead, no cast shadow from the nose across the
cheek, no inter-reflection filling the eye sockets, and no saturation clipping —
none of which a Lambertian model contains. Those same differences are what the
residual test in Step 3 measures numerically.

Each subject therefore gets three montages side by side — **measured**,
**rendered**, and their **difference** — so the residual can be read directly
against the image that produced it, before any thresholding is applied:

| colour in the difference panel | meaning | cause |
|---|---|---|
| blue (measured darker than predicted) | the model says lit, the camera saw dark | cast and attached shadows |
| red (measured brighter than predicted) | the model says dark, the camera saw bright | specular highlights, inter-reflection in the eye sockets and nostrils |
| white | model and measurement agree | Lambertian assumption holding |

A single symmetric colour scale, set to the 99.5th percentile of `|residual|`,
is shared by all 64 panels so they can be compared against one another rather
than each being stretched to its own range. The two dead B02 frames stand out
immediately as solid blue panels: the camera recorded nothing, but the model
still predicts a lit face.

`montage()` is also taken from the tutorial: instead of 64 separate axes, the
images are tiled into one large array and drawn with a single `imshow`. The
panels then sit flush against each other, which makes neighbouring lighting
conditions much easier to compare, and it draws far faster.

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

### Step 3b — Dead frames in B02 (investigated, then disabled)

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

**This mechanism is commented out in the final code.** The question specifies
the 2σ residual test and nothing else, so these two frames are left in and put
through exactly the same test as every other image. `find_dead_frames` is kept
commented in Part 1 because the behaviour below is worth reporting.

What the residual test does with them is itself the interesting result.
`image_028` is caught easily — 59.5% of its pixels flagged, rank 3 of 64 —
because its lighting direction predicts a well-lit face, so the all-black
measurement disagrees with the model everywhere. `image_052` largely escapes:
only **4.9% flagged, rank 26 of 64**, barely above the 1.6% median. Its
`l_z = −0.087` is nearly perpendicular to the camera axis, so the model already
predicts almost no brightness for a front-facing pixel — measured ≈0,
predicted ≈0, they agree, and no residual appears. A failed capture whose
lighting direction happens to predict darkness anyway is invisible to a
residual test, while the row of zeros still biases the least-squares solve.

This is a genuine limitation of a purely residual-based criterion, and it is
easily fixed by adding the peak test above to the rejection mask
(`outliers | dead[:, None, None]`). Doing so takes B02's `RMS |z_a − z_b|`
improvement from −12.8 % to −19.2 %. It is left out of the reported results
because it goes beyond what the question specifies.

---

### Step 4 — Re-solving without the outliers

In Step 0 every pixel used all 64 images, so every pixel shared the same 64×3
matrix `L` and one `lstsq` handled all 32,256 at once. After rejection **each
pixel keeps a different subset**, so there is no shared matrix any more — every
pixel has its own small least-squares problem.

Both cases are handled by the one function, `solve_photometric_stereo(imgs,
light_dirs, weights=None)`: with no weights it takes the shared-`lstsq` path of
Step 0, and with a weight mask it takes the path below. Splitting `g` into
`ρ = |g|` and `n = g/|g|` is identical either way, so it is written once.

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

### Step 5 — What rejection does to the albedo and normals

Step 4 re-computes albedo and normals, and only then re-integrates. It is worth
looking at those two quantities directly, before the change is propagated
through `p`, `q` and the cumulative sums, because the pattern is clean and
consistent across all four subjects.

**The albedo rises almost everywhere.** The mean change is positive for every
dataset (+0.042, +0.034, +0.029, +0.034 for B01, B02, B05, B07 — about 7–8 % of
mean albedo). This is not an artefact; it follows from which observations get
rejected. Shadows are far more common than specular highlights on a face, and a
shadowed observation is a *spuriously dark* measurement. Least squares over all
64 images averages those dark measurements in and pulls the estimated albedo
down. Removing them lets the albedo rise to what the unshadowed observations
alone imply.

The exceptions are the regions that go the other way, shown in blue: the eyes,
the eyebrows, hairline and the bridge of the nose. There the dominant outlier
is a specular highlight — a *spuriously bright* measurement — so rejecting it
lowers the albedo instead.

**The normals move most at the eyes.** The per-pixel angular change between the
old and new unit normal is near zero over most of the face (4.1–4.3° mean) but
spikes to 25–42° in tight blobs on the corneas of every single subject, with
smaller peaks at the nostrils and along the specular ridge of the nose. This is
exactly what should be expected: the cornea is a wet, curved mirror and is the
single most non-Lambertian surface on a face, so it is where the model was
worst and where removing the offending observations changes the answer most.

Three figures cover this: albedo before / after / difference plus the normal
change map, then the normal X and Y components before and after (displayed as
in the week 2 tutorial), then the height maps.

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
| B02     | 10.7 %  | 51 | 4.11° / 39.2° | 0 |
| B05     | 10.9 %  | 49 | 4.15° / 41.7° | 0 |
| B07     | 12.2 %  | 50 | 4.15° / 30.5° | 0 |

The largest normal changes land on the eyes, the nostrils and the specular
ridge of the nose — exactly the regions where the Lambertian assumption is
weakest.

**Effect on path self-consistency**, `RMS |z_a − z_b|`:

| Subject | baseline | after rejection | change |
|---------|----------|-----------------|--------|
| B01 |  7.28 px |  7.38 px | **+1.3 %** |
| B02 | 22.10 px | 19.27 px | **−12.8 %** |
| B05 | 16.09 px | 17.39 px | **+8.1 %** |
| B07 | 11.94 px |  7.99 px | **−33.1 %** |

(B02's two dead frames are left in, as described in Step 3b. Force-rejecting
them as well would take its figure from −12.8 % to −19.2 %.)

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
2. four measured / rendered / difference figures, one per subject, each three
   8×8 montages side by side;
3. four 8×8 outlier montages, one per subject, flagged pixels in red;
4. a 4×4 grid of the re-computed albedo — baseline, rejected, the difference,
   and the per-pixel normal change in degrees;
5. a 4×4 grid of the normal X and Y components, baseline vs rejected;
6. a 4×3 before/after grid — baseline `z_c`, outlier-rejected `z_c`, and the
   difference;
7. 3D renders via the supplied `plot_face_3d`, baseline then outlier-rejected
   for each face (8 windows, one at a time).

20 figures in total.

---

## Question 2 — Connect-Four Board State

Determining the state of a connect-four board — a 6×7 matrix of `0` (empty),
`1` (yellow) or `2` (red) — from a single photograph of it, taken from an
arbitrary perspective. Run with `python3 mainQ2.py`.

The dataset is 15 images of the same board at 3472×2598, in varying states of
play and from varying viewpoints, with validation data in
`assign1Q2_validationdata/`: `board_corners.pkl` (the four board corners per
image, ordered upper-left, upper-right, lower-left, lower-right) and
`board_states.pkl` (the ground-truth 6×7 matrix per image).

The approach follows "Idea 1" from the assignment brief: isolate the board by
its colour, find its four corners, rectify it with a perspective transform, then
read off each of the 42 cells.

---

### Step 0 — Isolating the board by colour

The board is a distinct saturated blue, so the natural first move is a colour
threshold in HSV. HSV is preferred over BGR here because it separates *what
colour* something is (hue) from *how vivid* (saturation) and *how bright*
(value) it is — so shading across the board changes mainly `V` while leaving `H`
almost untouched, which a BGR threshold cannot express.

Rather than hard-coding threshold bounds, the bounds are *derived per image*
from the histogram of that image. This adapts to the different lighting
conditions across the dataset.

#### The hue peak

`cv2.calcHist` builds a histogram of the hue channel and the peak within the
blue range is taken as the board's hue:

```python
hist_h = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
peak_hue = blue_hue_range[0] + int(np.argmax(hist_h[blue_hue_range[0]:blue_hue_range[1]]))
```

Two details matter here:

- **OpenCV packs hue into 0–179**, not 0–255, so that a full 360° hue circle fits
  in a `uint8` at 2° per step. The histogram is therefore built with 180 bins over
  `[0, 180]`. Using 256 bins over `[0, 256]` leaves 76 bins permanently empty.
- **`np.argmax` over a slice returns an index into the slice**, not into the
  original array. `np.argmax(hist_h[100:140])` returns a number in 0–39, so the
  slice start must be added back on to recover the true hue bin. Omitting this
  silently yields a hue in the wrong part of the spectrum entirely.

This step is reliable: across all 15 images the recovered peak hue lands in a
range of just **100–102**.

#### Why the saturation peak needs Otsu

The obvious next step — take the peak of the saturation histogram the same way —
does not work, and it is worth recording why, because the failure is silent.

Measuring the board region (using the ground-truth corners) against everything
else gives:

| region | median H | median S | median V |
| --- | --- | --- | --- |
| board | 101 | **218** | 130 |
| background | 20 | **61** | 168 |

The wall behind the board is *the same hue* as the board — it is a pale
blue-grey — but far less saturated. Critically, it also covers roughly **five
times more of the image** than the board does. So a saturation histogram taken
over all blue-hued pixels is dominated by the wall: on `001.jpg` the peak lands
at `S = 39`, and the resulting ±50 window of `[0, 89]` selects the wall and
excludes the board completely. The mask comes out inverted from what was
intended, with the board appearing as a silhouette-shaped hole in it.

Hue cannot separate these two populations. Saturation can — 218 versus 61 — but
only if the peak is measured on the correct population. **Otsu's method** is used
to find the split point:

```python
blue_saturations = hsv_frame[:, :, 1][hue_mask > 0].reshape(-1, 1)
sat_split, _ = cv2.threshold(blue_saturations, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Otsu assumes the data is bimodal and picks the threshold minimising the
intra-class variance of the two resulting groups — exactly the situation here
(a low-saturation wall mode and a high-saturation board mode). Everything below
the split is discarded, and the saturation and value peaks are then measured on
the surviving pixels only.

The split point is chosen independently per image and lands in the narrow band
**123–138** across the dataset, which is reassuring: it is adapting slightly to
each image's lighting rather than jumping around.

#### Final bounds

```python
lower_bound = np.array([hue_lo, peak_saturation - 60, peak_value - 100], np.uint8)
upper_bound = np.array([hue_hi, peak_saturation + 60, peak_value + 100], np.uint8)
mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
```

The value window (±100) is deliberately much wider than the hue (±10) and
saturation (±60) windows, because brightness varies substantially across the
board with shading and specular highlights while hue and saturation stay tight.
A ±50 value window was measured to clip shadowed regions of the board and cost
~5% of recall for no gain in precision.

All bounds are clipped to their valid channel ranges before being cast to
`uint8`. Without this, `peak_value - 100` can go negative and wrap around to a
large positive number, inverting the test.

#### Colour selection results

Scored against the ground-truth board polygons:

| metric | value |
| --- | --- |
| mean precision | **0.971** (worst image 0.915) |
| mean recall | 0.648 |

Recall is capped near 0.65 *by construction* and is not a defect: the
ground-truth polygon spans the whole board rectangle, which includes the 42
holes and the tokens sitting in them. Those are not blue and correctly are not
selected. Precision is the meaningful number here, and at 97% almost nothing
outside the board survives the threshold. The residual 3% is mostly the board's
own side edge, which protrudes slightly beyond the four labelled corners — again
a correct detection rather than an error.

---

### Step 1 — Binarisation and morphology

The masked colour image is reduced to a binary mask and cleaned with an
opening followed by a closing, repeated `n` times, with a 5×5 rectangular
structuring element:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
for _ in range(n):
    bin_board = cv2.morphologyEx(bin_board, cv2.MORPH_OPEN, kernel)
    bin_board = cv2.morphologyEx(bin_board, cv2.MORPH_CLOSE, kernel)
```

**Opening** (erode then dilate) removes small bright specks smaller than the
kernel; **closing** (dilate then erode) fills small dark holes. Together they
tidy the mask boundary without materially changing the board's shape.

Sweeping the iteration count and measuring the effect on the final corner
accuracy:

| iterations | mean corner error | contours per image |
| --- | --- | --- |
| 0 (none) | **8.2 px** | 2 – 123 |
| 1 | 9.1 px | 1 – 24 |
| 3 | 9.1 px | 1 – 24 |
| 5 | 9.1 px | 1 – 24 |
| 10 | 9.1 px | 1 – 24 |

The honest reading of this table is that **morphology is not improving corner
accuracy on this dataset** — it costs about 0.9 px — and that everything beyond
the first iteration does nothing at all, because the mask reaches a fixed point.
What it does achieve is cutting the number of spurious contours by roughly 5×
(from as many as 123 down to 24), which makes the contour stage cheaper and less
dependent on the largest-area selection being right.

The colour mask is simply already clean enough (97% precision) that there is
little left for morphology to fix. It is retained because the cost is negligible
and the reduction in spurious contours is a genuine robustness gain, but the
measurement above is the reason, rather than an assumption that cleaning always
helps.

---

### Step 2 — From contour to four corners

This is the step that converts a blob into four usable coordinates. It has three
parts, and each one exists for a specific reason.

#### 2a — Select the board contour

```python
contours, _ = cv2.findContours(bin_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
board_contour = max(contours, key=cv2.contourArea)
```

`RETR_EXTERNAL` keeps only outermost contours, discarding the 42 hole boundaries
nested inside the board — those are interior contours and are not wanted when
looking for the outline. `CHAIN_APPROX_SIMPLE` collapses straight runs of pixels
into their endpoints, which reduces memory without losing shape.

Even after morphology, an image can contain up to 24 contours. On `001.jpg`
there are 2, with areas of **1,423,978** and **25,623** pixels — the board and a
leftover speck. Selecting by maximum area is what guarantees the board is the
one carried forward. Drawing or processing *all* contours instead is a common
mistake here and will corrupt the corner estimate.

The selected contour is an **outline**, not a set of corners: on `001.jpg` it
consists of **1349 points**, roughly one every few pixels around the perimeter.
Across the dataset the largest contour ranges from 662 to 2051 points. The task
of the next step is to reduce that to exactly 4.

#### 2b — Simplify the outline to a quadrilateral

`cv2.approxPolyDP` implements the **Douglas–Peucker** algorithm. It recursively
discards any vertex lying within `epsilon` pixels of the straight line joining
its neighbours — a point in the middle of a straight edge carries no shape
information, whereas a point where the outline turns does. What survives are the
corners.

`epsilon` is expressed as a fraction of the contour perimeter so that it scales
with the apparent size of the board rather than being fixed in pixels:

```python
epsilon = epsilon_fraction * cv2.arcLength(board_contour, True)
candidate = cv2.approxPolyDP(board_contour, epsilon, True)
```

The choice of fraction matters. On `001.jpg`:

| epsilon fraction | resulting vertices |
| --- | --- |
| 0.001 | 10 |
| 0.005 | 4 |
| 0.010 | 4 |
| 0.020 | 4 |

Too small a value retains bumps along the edges and yields more than four
vertices; too large begins to cut corners off the shape. Rather than trusting a
single hard-coded fraction to be correct on every image, the implementation
**sweeps the fraction from 0.01 upward in steps of 0.005 and stops at the first
value that produces exactly four vertices**. If no value in the range succeeds,
it falls back to `cv2.minAreaRect` / `cv2.boxPoints`, which returns the minimum
enclosing rotated rectangle and therefore always has exactly four corners. On
this dataset the sweep succeeds on all 15 images and the fallback is never
reached, but it prevents a hard failure on an unseen image.

#### 2c — Sort the corners into a known order

`approxPolyDP` returns the four points **in the order the contour was traced**,
which depends on the board's orientation and on where in the outline the trace
happened to begin. It is *not* a fixed geometric order. On `001.jpg` the raw
output is:

```
[0] x= 437  y=1079     <- upper left
[1] x= 652  y=2003     <- lower LEFT
[2] x=1958  y=2164     <- lower right
[3] x=2136  y=1186     <- upper RIGHT
```

Index `1` is the lower-left corner, not the upper-right. On a differently
oriented photograph the assignment of indices changes, so the points cannot be
used positionally and must be identified geometrically.

The standard trick uses the sums and differences of the coordinates. Recall that
in image coordinates **x increases to the right and y increases downward**:

- the **upper-left** corner is nearest the origin, so it has the **smallest**
  `x + y`;
- the **lower-right** corner is furthest from it, so it has the **largest**
  `x + y`;
- the **upper-right** corner has large `x` and small `y`, so `y − x` is at its
  **most negative**;
- the **lower-left** corner has small `x` and large `y`, so `y − x` is at its
  **most positive**.

Evaluated on the points above:

| index | `x + y` | `y − x` | identified as |
| --- | --- | --- | --- |
| 0 | **1516** (min) | 642 | upper left |
| 1 | 2655 | **1351** (max) | lower left |
| 2 | **4122** (max) | 206 | lower right |
| 3 | 3322 | **−950** (min) | upper right |

which recovers the correct labelling. This works for any convex quadrilateral
under moderate perspective, and is orientation-independent.

The function returns the corners ordered **(upper-left, upper-right, lower-left,
lower-right)** to match the convention used in `board_corners.pkl`, so detected
corners can be compared against ground truth directly without reordering.

> **Note.** That ordering is *not* a valid polygon winding — tracing
> UL → UR → LL → LR crosses over itself in a bow-tie. Anywhere the corners are
> used as a polygon (drawing the outline, or feeding
> `cv2.getPerspectiveTransform`) they must be reordered to UL → UR → LR → LL,
> which is what the `corners[[0, 1, 3, 2]]` indexing in the display code does.
> Getting this wrong produces a mirrored or rotated rectification and a
> transposed board matrix.

#### Corner detection results

Mean Euclidean distance from each detected corner to its ground-truth
counterpart, in pixels, on 3472×2598 images:

| image | UL | UR | LL | LR | mean |
| --- | --- | --- | --- | --- | --- |
| 001 | 35.5 | 30.1 | 2.2 | 6.3 | 18.5 |
| 002 | 43.8 | 23.3 | 3.6 | 9.8 | 20.2 |
| 003 | 8.1 | 2.2 | 2.0 | 10.0 | 5.6 |
| 004 | 34.7 | 17.1 | 1.4 | 5.7 | 14.7 |
| 005 | 2.2 | 4.1 | 1.0 | 5.8 | 3.3 |
| 006 | 11.4 | 1.0 | 2.0 | 8.6 | 5.8 |
| 007 | 23.7 | 3.2 | 3.2 | 16.3 | 11.6 |
| 008 | 11.4 | 2.2 | 1.4 | 7.2 | 5.6 |
| 009 | 6.4 | 1.4 | 1.4 | 6.4 | 3.9 |
| 010 | 8.6 | 1.4 | 1.0 | 9.9 | 5.2 |
| 011 | 5.8 | 29.5 | 1.0 | 13.0 | 12.4 |
| 012 | 2.0 | 3.6 | 0.0 | 11.3 | 4.2 |
| 013 | 7.2 | 2.2 | 1.4 | 9.2 | 5.0 |
| 014 | 35.4 | 2.2 | 2.2 | 13.0 | 13.2 |
| 015 | 14.1 | 5.0 | 1.4 | 7.2 | 6.9 |

**Mean 9.1 px, worst image 20.2 px** — approximately 0.3% of the board's width,
and a small fraction of a cell, so the error is well within tolerance for
locating cells.

The error is **systematically worse at the top of the board**: the two upper
corners average 12.6 px against 5.5 px for the two lower ones, and every large
outlier in the table is a UL or UR entry. This is not random noise. The physical
board has a raised lip around the token entry slot along its top edge, so the
detected contour follows the true silhouette of the plastic while the
hand-labelled corners sit at the playing-field boundary slightly below it. The
two are measuring genuinely different things, and the detection is not wrong so
much as differently defined. The bottom corners, where no such lip exists, agree
to within a few pixels.

---

### Status

Implemented in `mainQ2.py`:

- `histogram_color_select` — per-image adaptive HSV threshold isolating the board
- `morphological_operations` — opening/closing cleanup of the binary mask
- `find_board_corners` — contour → quadrilateral → ordered corners
- `show` — display helper that scales the 3472×2598 images down to fit on screen

Still to do:

1. `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to rectify the board
   to a fronto-parallel view using the detected corners;
2. sampling each of the 42 cells in the rectified image and classifying it as
   empty / yellow / red by colour;
3. the accuracy reporting the brief requires — per-image board accuracy, average
   board accuracy across all images, and overall accuracy (the percentage of
   images with all 42 cells correct).

An exploratory prototype of stages 1 and 2 — warping to 700×600, sampling a disc
at each cell centre and classifying by hue with a saturation gate — reached
**99.5% cell accuracy with 14 of 15 boards perfect** using the detected corners.
That figure is indicative only and is not yet part of `mainQ2.py`.

---

### Running

```bash
python3 mainQ2.py
```

Requires `numpy`, `opencv-python` and `matplotlib` only.

> **Environment note.** `cv2.Mat` exists only in the `opencv-python` wheels from
> version 4.7 onward. Ubuntu's apt-packaged `python3-opencv` (4.6.0) does not
> provide it, and using it as a type annotation raises `AttributeError: module
> 'cv2' has no attribute 'Mat'` at import time, because annotations are evaluated
> when the `def` statement runs. All annotations here use `np.ndarray` instead,
> which is portable across versions and matches the brief's requirement that the
> function take "a numpy array containing the colour image data".
