"""
AMME5710 Assignment 1, Question 1 - Photometric Stereo.

Recovers a 3D height map of a face from 64 images of it taken under 64 known
lighting directions, for the four Yale face datasets B01, B02, B05 and B07.

The file is in two parts:
    PART 1 - all functions, grouped by the step of the pipeline they belong to
    PART 2 - the main script, which runs those steps in order and plots

Run with:  python3 mainQ1.py
"""

import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from plot_face_3d import plot_face_3d


# ===========================================================================
#
#   PART 1 - FUNCTIONS
#
# ===========================================================================


# ---------------------------------------------------------------------------
# Step 0 - Loading
# ---------------------------------------------------------------------------
# The loading code as supplied in the week 2 tutorial, wrapped in a function so
# that it can be reused for each of the four subjects.
def load_subject(tag):
    """Load the 64 images and the 64 lighting directions for one face."""
    img_dir_path = 'yale_face_data/image_dir_%s' % tag
    light_dirs_path = 'yale_face_data/light_dir%s.pkl' % tag

    # Load images of face (greyscale, scaled to 0..1)
    imgs = []
    for i in range(64):
        im = cv2.imread(os.path.join(img_dir_path, 'image_%03d.png' % (i)),
                        cv2.IMREAD_GRAYSCALE)
        imgs.append(im.astype(np.float64) / 255.0)
    imgs = np.array(imgs)                      # (64, 192, 168)

    # Load lighting direction vectors
    light_dirs = pickle.load(open(light_dirs_path, "rb"))   # (64, 3), unit
    return imgs, light_dirs


# ---------------------------------------------------------------------------
# Step 1 - Images + lighting -> albedo and surface normals
# ---------------------------------------------------------------------------
# Lambertian model for pixel (x,y) under light k:   I_k = rho * (n . l_k)
# Substituting g = rho * n makes it linear:         I_k = l_k . g
# 64 equations, 3 unknowns per pixel -> over-determined, solve by least squares.
# Then rho = |g| and n = g / |g|.
#
# The same function serves Step 5. With weights=None every pixel uses all 64
# images, so they all share the same 64x3 matrix L and one lstsq solves all
# 32,256 pixels at once. Once outliers are rejected each pixel keeps a
# different subset, so there is no shared matrix; instead of looping over
# pixels, form the weighted 3x3 normal equations
#
#     A g = b,    A = sum_k w_k l_k l_k^T,    b = sum_k w_k I_k l_k
#
# with w_k = 0 for a rejected observation. Each of the 9 entries of A is one
# matrix-vector product over all pixels at once, and the whole stack of 3x3
# systems is solved by a single batched np.linalg.solve.
def solve_photometric_stereo(imgs, light_dirs, weights=None):
    n_img, h, w = imgs.shape
    N = h * w
    I = imgs.reshape(n_img, N)                               # (64, N)
    bad = np.zeros(N, dtype=bool)

    if weights is None:
        g = np.linalg.lstsq(light_dirs, I, rcond=None)[0].T  # (N, 3)
    else:
        W = weights.reshape(n_img, N).astype(np.float64)
        A = np.empty((N, 3, 3))
        for i in range(3):
            for j in range(3):
                A[:, i, j] = (light_dirs[:, i] * light_dirs[:, j]) @ W
        b = (W * I).T @ light_dirs                           # (N, 3)

        # A pixel with too few surviving views gives a singular (or nearly
        # singular) system. Fall back to the unweighted solution there rather
        # than letting a blown-up solve poison the height map.
        bad = np.abs(np.linalg.det(A)) < 1e-9
        A[bad] = np.eye(3)
        g = np.linalg.solve(A, b[:, :, None])[:, :, 0]       # (N, 3)
        if bad.any():
            g[bad] = np.linalg.lstsq(light_dirs, I, rcond=None)[0].T[bad]

    albedo = np.linalg.norm(g, axis=1)                       # (N,)
    safe = np.where(albedo < 1e-8, 1e-8, albedo)
    normals = (g / safe[:, None]).reshape(h, w, 3)           # (192, 168, 3)
    return albedo.reshape(h, w), normals, bad.sum()


# ---------------------------------------------------------------------------
# Step 2 - Surface normals -> surface gradients -> height maps
# ---------------------------------------------------------------------------
# Model the face as a height function z = f(x,y) over the image plane.
# Tangents:  (1, 0, dz/dx)  and  (0, 1, dz/dy)
# Normal is their cross product:  n  ~  (-dz/dx, -dz/dy, 1)
# Inverting:   p = dz/dx = -nx/nz,   q = dz/dy = -ny/nz
#
# Sign: these normals reconstruct the face inside-out with the textbook signs
# (the nose comes out as a pit), so both gradients are flipped. Flipping both
# is just z -> -z, i.e. the supplied normals use a height axis pointing away
# from the camera.
#
# Integration:
# z(x,y) = z(0,0) + integral of (p dx + q dy) along a path from the top-left.
# With dx = dy = 1 each integral is a cumulative sum. Sums start at index 1 so
# that z(0,0) = 0 exactly (index 0 would include the reference pixel's own
# gradient).
def heights_from_normals(surface_normals):
    nz = surface_normals[:, :, 2]

    p = surface_normals[:, :, 0] / nz            # height change per column step
    q = surface_normals[:, :, 1] / nz            # height change per row step

    # (a) across the top row first, then down each column
    z_a = np.zeros_like(p)
    z_a[0, 1:] = np.cumsum(p[0, 1:])
    z_a[1:, :] = z_a[0, :] + np.cumsum(q[1:, :], axis=0)

    # (b) down the first column first, then across each row
    z_b = np.zeros_like(q)
    z_b[1:, 0] = np.cumsum(q[1:, 0])
    z_b[:, 1:] = z_b[:, 0:1] + np.cumsum(p[:, 1:], axis=1)

    # (c) the average of (a) and (b)
    z_c = 0.5 * (z_a + z_b)
    return z_a, z_b, z_c


# ---------------------------------------------------------------------------
# Step 3 - Rendering the Lambertian model (week 2 tutorial, Exercise 1)
# ---------------------------------------------------------------------------
# The tutorial rendered each face by evaluating I = rho * (n . l) pixel by
# pixel in a double loop. The same expression written for the whole image
# stack at once is a single matrix product: stack g = rho * n as an (N,3)
# array and multiply by the (64,3) lighting matrix.
#
# clamp=True reproduces the tutorial exactly - a negative n.l means the light
# is behind the surface, so nothing is lit and the value is floored at 0. That
# is the physically correct render and is what should be compared against the
# photographs. clamp=False is used for the residuals in Step 4, for the reason
# given there.
def render_images(light_dirs, albedo, normals, clamp=True):
    h, w = albedo.shape
    g = (albedo[:, :, None] * normals).reshape(h * w, 3)     # rho * n, (N,3)
    rendered = (light_dirs @ g.T).reshape(-1, h, w)          # (64,192,168)
    if clamp:
        rendered = np.maximum(rendered, 0.0)
    return rendered


# ---------------------------------------------------------------------------
# Step 4 - Outlier detection
# ---------------------------------------------------------------------------
# Having albedo and a normal for a pixel, the Lambertian model can *predict*
# how bright that pixel should have been in every one of the 64 images. The
# residual is measured minus predicted:
#
#     r_k(x,y) = I_k(x,y) - rho(x,y) * ( n(x,y) . l_k )
#
# so each pixel gets 64 residuals. Large ones mark observations the model
# cannot explain: cast shadows, specular highlights on the nose and forehead,
# and inter-reflections around the eye sockets and nostrils - all the ways a
# real face fails to be Lambertian.
#
# The prediction is deliberately NOT clamped to max(n.l, 0). Physically a
# negative n.l means the light is behind the surface so nothing is lit, but
# clamping is exactly what hides shadows: inside a shadow the measurement is
# ~0 and a clamped model also predicts ~0, so they agree and the residual
# vanishes. Left un-clamped the model still predicts a lit pixel, giving the
# large negative residual that flags the shadow.
def compute_residuals(imgs, light_dirs, albedo, normals):
    predicted = render_images(light_dirs, albedo, normals, clamp=False)
    return imgs - predicted


# The threshold is per pixel, not global: residuals scale with albedo, so a
# single global threshold would simply flag every bright part of the face
# rather than the genuinely anomalous observations. sigma(x,y) is the standard
# deviation of that one pixel's own 64 residuals.
def outlier_mask(residuals, n_sigma=2.0):
    sigma = residuals.std(axis=0, keepdims=True)             # (1,192,168)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    return np.abs(residuals) > n_sigma * sigma               # (64,192,168) bool


# --- Dead frames: DISABLED -------------------------------------------------
# Two B02 exposures (image_028, image_052) are failed captures rather than
# merely back-lit frames. An earlier version detected them on peak brightness
# and force-rejected them. That goes beyond what the question asks for - the
# question specifies the 2-sigma residual test and nothing else - so the whole
# mechanism is commented out below and those two frames are now treated
# exactly like every other image.
#
# Kept, commented, because the behaviour is worth reporting: the residual test
# catches image_028 easily (59.5% of its pixels flagged, rank 3 of 64) but
# image_052 largely escapes it (4.9% flagged, rank 26 of 64). image_052 has
# l_z = -0.087, almost perpendicular to the camera, so the model already
# predicts near-zero brightness for a front-facing pixel; measured ~0 and
# predicted ~0 agree, no residual appears, and the row of zeros still biases
# the solve. Re-enabling this and OR-ing `dead` into the outlier mask takes
# B02's RMS |z_a - z_b| improvement from -12.8% to -19.2%.
#
# def find_dead_frames(imgs, peak_thresh=0.25):
#     """Frames that never illuminate anything, judged on peak not mean."""
#     peaks = imgs.reshape(imgs.shape[0], -1).max(axis=1)
#     return peaks < peak_thresh


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
# The tutorial's montage: rather than 64 separate axes, tile the images into
# one large array and show it with a single imshow. Faster to draw and the
# panels sit flush against each other, which makes it much easier to compare
# neighbouring lighting conditions.
def montage(imgs, n_cols=8, pad=2, pad_value=1.0):
    n_img, h, w = imgs.shape[:3]
    n_rows = int(np.ceil(n_img / n_cols))
    shape = (n_rows * (h + pad), n_cols * (w + pad)) + imgs.shape[3:]
    tiled = np.full(shape, pad_value, dtype=np.float64)
    for k in range(n_img):
        r, c = k // n_cols, k % n_cols
        tiled[r * (h + pad):r * (h + pad) + h,
              c * (w + pad):c * (w + pad) + w] = imgs[k]
    return tiled


# Every "one row per subject" figure in this file has the same shape: a grid of
# images, each with its own title, colour map and optional colour limits. One
# helper draws all of them. `rows` is a list (one entry per subject) of lists of
# (image, title, cmap, clim) panels.
def plot_grid(rows, width=13):
    n_r, n_c = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_r, n_c, figsize=(width, 3.2 * n_r))
    for ax_row, panels in zip(np.atleast_2d(axes), rows):
        for ax, (img, title, cmap, clim) in zip(ax_row, panels):
            lo, hi = clim if clim else (None, None)
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


# Figure 1: every subject on its own row, showing the albedo alongside the
# three integration strategies so the artefacts of each can be compared.
def plot_albedo_and_heights(results, tags):
    plot_grid([[(results[t]['albedo'], '%s albedo' % t, 'gray', None),
                (results[t]['z_a'], '%s (a) row first' % t, 'viridis', None),
                (results[t]['z_b'], '%s (b) column first' % t, 'viridis', None),
                (results[t]['z_c'], '%s (c) average' % t, 'viridis', None)]
               for t in tags], width=12)


# Figure 2: measured, rendered and their difference, side by side, one figure
# per subject so each montage is large enough to read. The difference panel is
# the residual of Step 4, shown before any thresholding:
#
#     blue  = measured darker than predicted -> cast shadow, attached shadow
#     red   = measured brighter than predicted -> specular highlight,
#             inter-reflection bouncing light back into eye sockets/nostrils
#     white = the Lambertian model fits
#
# A common symmetric colour scale is used across all 64 panels so that they can
# be compared against one another rather than each being stretched to its own
# range.
def plot_render_comparison(tag, imgs, rendered, diff):
    lim = np.percentile(np.abs(diff), 99.5)

    # A 4th narrow column holds the colour bar, so all three montages stay the
    # same size instead of the last one shrinking to make room for it
    fig, axes = plt.subplots(1, 4, figsize=(20, 8),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.04]))
    axes[0].imshow(montage(imgs), cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title('%s measured' % tag, fontsize=11)
    axes[1].imshow(montage(rendered), cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title('%s rendered (Lambertian model)' % tag, fontsize=11)
    im = axes[2].imshow(montage(diff, pad_value=0.0), cmap='coolwarm',
                        vmin=-lim, vmax=lim)
    axes[2].set_title('%s difference (measured - rendered)' % tag, fontsize=11)
    fig.colorbar(im, cax=axes[3])
    for ax in axes[:3]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Figure 3: montage of the outlier masks, one panel per lighting condition, so
# all 64 images of a face are shown at once. Flagged pixels are drawn in red
# over the image itself rather than as a bare binary mask, so it is possible to
# see *what* was rejected - the shadow boundaries, the specular highlight on
# the nose and forehead, the dark side of the face under grazing light.
#
# Many of the 64 exposures are very dark, so each panel is stretched to its own
# 99th percentile before display. This is for visibility only; every
# calculation uses the raw values.
def overlay_montage(imgs, masks, title):
    n_img, h, w = imgs.shape

    # Build a colour version of every frame with the flagged pixels painted red
    rgb = np.empty((n_img, h, w, 3))
    for k in range(n_img):
        scale = max(np.percentile(imgs[k], 99), 1e-3)
        grey = np.clip(imgs[k] / scale, 0.0, 1.0)
        rgb[k] = np.stack([grey] * 3, axis=-1)
        rgb[k][masks[k]] = [1.0, 0.15, 0.15]         # flagged pixels in red

    plt.figure(figsize=(14, 16))
    plt.imshow(montage(rgb))
    # Label each tile with its index and the percentage of pixels flagged
    for k in range(n_img):
        r, c = k // 8, k % 8
        plt.text(c * (w + 2) + 4, r * (h + 2) + 14,
                 '%d: %.0f%%' % (k, 100.0 * masks[k].mean()),
                 color='yellow', fontsize=8)
    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Figure 4: what the outlier rejection did to the recovered albedo and normals
# themselves - the quantities Step 5 actually re-computes - before the effect
# is propagated through to the height map.
#
#   albedo before / after   the two look almost identical at a glance, which is
#                           the point: the change is a correction, not a
#                           re-estimation
#   albedo change           red = albedo raised by rejection, blue = lowered.
#                           Rejecting a shadowed observation removes a
#                           spuriously dark measurement, so the albedo rises;
#                           rejecting a specular one removes a spuriously
#                           bright measurement, so it falls
#   normal change           angle between the old and new unit normal, in
#                           degrees, which is the quantity that actually drives
#                           the height map through p and q
def plot_albedo_normal_change(results, tags):
    rows = []
    for t in tags:
        d = results[t]['albedo2'] - results[t]['albedo']
        lim = np.percentile(np.abs(d), 99.5)
        rows.append([
            (results[t]['albedo'], '%s albedo baseline' % t, 'gray', None),
            (results[t]['albedo2'], '%s albedo rejected' % t, 'gray', None),
            (d, '%s albedo change' % t, 'coolwarm', (-lim, lim)),
            (results[t]['normal_change'], '%s normal change (deg)' % t,
             'magma', None)])
    plot_grid(rows)


# Figure 5: the surface normal X and Y components before and after rejection,
# displayed as in the week 2 tutorial. These are the fields that Step 2 turns
# into the gradients p and q, so any change here is what moves the surface.
def plot_normal_components(results, tags):
    plot_grid([[(results[t][k][:, :, c], '%s normal %s %s' % (t, xy, lbl),
                 'viridis', (-1.0, 1.0))
                for c, xy in [(0, 'X'), (1, 'Y')]
                for k, lbl in [('normals', 'baseline'),
                               ('normals2', 'rejected')]]
               for t in tags])


# Figure 6: before / after comparison of the averaged height maps, with the
# difference between them in the third column to show where rejecting the
# outliers actually changed the reconstruction.
def plot_before_after(results, tags):
    plot_grid([[(results[t]['z_c'], '%s baseline (c)' % t, 'viridis', None),
                (results[t]['z_c2'], '%s outliers rejected (c)' % t,
                 'viridis', None),
                (results[t]['z_c2'] - results[t]['z_c'], '%s difference' % t,
                 'coolwarm', None)]
               for t in tags], width=11)


# ===========================================================================
#
#   PART 2 - MAIN
#
# ===========================================================================

tags = ['B01', 'B02', 'B05', 'B07']
results = {}


# --- Steps 0 to 2: load, solve, integrate, for every dataset ---------------
for tag in tags:
    imgs, light_dirs = load_subject(tag)
    albedo_image, surface_normals, _ = solve_photometric_stereo(
        imgs, light_dirs)
    z_a, z_b, z_c = heights_from_normals(surface_normals)
    results[tag] = dict(imgs=imgs, light_dirs=light_dirs,
                        albedo=albedo_image, normals=surface_normals,
                        z_a=z_a, z_b=z_b, z_c=z_c)
    print('%s: z(0,0) = %.1f %.1f %.1f   height range = %.1f px'
          % (tag, z_a[0, 0], z_b[0, 0], z_c[0, 0], z_c.max() - z_c.min()))

# Sanity check: our B01 solve should match the supplied pre-reconstructed data
data = pickle.load(open('yale_face_data/B01_albedo_normals.pkl', "rb"))
print('B01 check -> max albedo error %.2e, max normal error %.2e'
      % (np.abs(results['B01']['albedo'] - data['albedo_image']).max(),
         np.abs(results['B01']['normals'] - data['surface_normals']).max()))

plot_albedo_and_heights(results, tags)


# --- Step 3: render each face back and compare against the photographs -----
for tag in tags:
    res = results[tag]
    res['rendered'] = render_images(res['light_dirs'], res['albedo'],
                                    res['normals'])
    diff = compute_residuals(res['imgs'], res['light_dirs'],
                             res['albedo'], res['normals'])
    plot_render_comparison(tag, res['imgs'], res['rendered'], diff)


# --- Step 4: flag observations the Lambertian model cannot explain ---------
print('')
for tag in tags:
    res = results[tag]

    # Dead-frame handling is disabled (see the commented block in Part 1):
    # dead = find_dead_frames(res['imgs'])
    # res['outliers'] = outlier_mask(res['residuals']) | dead[:, None, None]
    res['residuals'] = compute_residuals(res['imgs'], res['light_dirs'],
                                         res['albedo'], res['normals'])
    res['outliers'] = outlier_mask(res['residuals'])

    frac = res['outliers'].mean() * 100.0
    kept = (~res['outliers']).sum(axis=0)                    # per-pixel count
    print('%s: %.1f%% of observations flagged as outliers, '
          'fewest surviving views for any pixel = %d'
          % (tag, frac, kept.min()))

for tag in tags:
    res = results[tag]
    overlay_montage(res['imgs'], res['outliers'],
                    '%s - red = |residual| > 2 sigma  (panel number and '
                    'percentage of pixels flagged)' % tag)


# --- Step 5: re-solve without the flagged data, then re-integrate ----------
print('')
for tag in tags:
    res = results[tag]
    albedo2, normals2, n_bad = solve_photometric_stereo(
        res['imgs'], res['light_dirs'], weights=~res['outliers'])
    z_a2, z_b2, z_c2 = heights_from_normals(normals2)
    res.update(albedo2=albedo2, normals2=normals2,
               z_a2=z_a2, z_b2=z_b2, z_c2=z_c2)

    # How far the normals moved, in degrees. Kept per pixel so it can be
    # displayed as a map, not just summarised as a single number.
    dot = np.clip((res['normals'] * normals2).sum(axis=2), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    res['normal_change'] = ang

    # Disagreement between integration paths (a) and (b) is a measure of how
    # badly the integrability constraint dp/dy = dq/dx has been violated, so it
    # doubles as a self-consistency score for the recovered normals.
    rms0 = np.sqrt(np.mean((res['z_a'] - res['z_b']) ** 2))
    rms1 = np.sqrt(np.mean((z_a2 - z_b2) ** 2))
    d_alb = albedo2 - res['albedo']
    print('%s: normals moved %.2f deg mean / %.1f deg max, %d singular pixels; '
          'albedo changed %+.3f mean / %.3f max abs (%.1f%% of mean albedo); '
          'RMS |z_a - z_b| %.2f -> %.2f px (%+.1f%%)'
          % (tag, ang.mean(), ang.max(), n_bad,
             d_alb.mean(), np.abs(d_alb).max(),
             100.0 * np.abs(d_alb).mean() / res['albedo'].mean(),
             rms0, rms1, 100.0 * (rms1 - rms0) / rms0))

# The re-computed albedo and normals themselves, then their effect on height
plot_albedo_normal_change(results, tags)
plot_normal_components(results, tags)
plot_before_after(results, tags)


# --- Step 6: 3D renders of the reconstructed faces -------------------------
# Both reconstructions are rendered for each face: the baseline first, then the
# one built from the outlier-rejected albedo and normals, so the two can be
# compared directly. plot_face_3d opens one window at a time.
for tag in tags:
    plot_face_3d(results[tag]['z_c'], results[tag]['albedo'])
    plot_face_3d(results[tag]['z_c2'], results[tag]['albedo2'])
