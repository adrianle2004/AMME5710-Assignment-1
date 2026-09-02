import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from plot_face_3d import plot_face_3d

# ---------------------------------------------------------------------------
# Loading (as supplied, wrapped in a function so it can be reused per subject)
# ---------------------------------------------------------------------------
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
# Images + lighting -> compute albedo and surface normals
# ---------------------------------------------------------------------------
# Lambertian model for pixel (x,y) under light k:   I_k = rho * (n . l_k)
# Substituting g = rho * n makes it linear:         I_k = l_k . g
# 64 equations, 3 unknowns per pixel -> over-determined, solve by least squares
# for every pixel at once. Then rho = |g| and n = g / |g|.
def solve_photometric_stereo(imgs, light_dirs):
    n_img, h, w = imgs.shape
    I = imgs.reshape(n_img, h * w)                     # (64, N)
    g, *_ = np.linalg.lstsq(light_dirs, I, rcond=None)  # (3, N)

    albedo = np.linalg.norm(g, axis=0)                 # (N,)
    safe = np.where(albedo < 1e-8, 1e-8, albedo)
    normals = (g / safe).T.reshape(h, w, 3)            # (192, 168, 3)
    return albedo.reshape(h, w), normals


# ---------------------------------------------------------------------------
# Surface normals -> surface gradients
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
# Integrate the gradients into a height map
# ---------------------------------------------------------------------------
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
# Run every available face dataset
# ---------------------------------------------------------------------------
tags = ['B01', 'B02', 'B05', 'B07']
results = {}
for tag in tags:
    imgs, light_dirs = load_subject(tag)
    albedo_image, surface_normals = solve_photometric_stereo(imgs, light_dirs)
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

# ---------------------------------------------------------------------------
# Display all four faces together: albedo and the three integration strategies
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(tags), 4, figsize=(12, 3.2 * len(tags)))
cols = ['albedo', '(a) row first', '(b) column first', '(c) average']
for r, tag in enumerate(tags):
    res = results[tag]
    for c, (img, name) in enumerate(zip(
            [res['albedo'], res['z_a'], res['z_b'], res['z_c']], cols)):
        ax = axes[r, c]
        im = ax.imshow(img, cmap='gray' if c == 0 else 'viridis')
        ax.set_title('%s %s' % (tag, name), fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Outlier detection
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
    n_img, h, w = imgs.shape
    g = (albedo[:, :, None] * normals).reshape(h * w, 3)     # rho * n, (N,3)
    predicted = (light_dirs @ g.T).reshape(n_img, h, w)      # (64,192,168)
    return imgs - predicted


# The threshold is per pixel, not global: residuals scale with albedo, so a
# single global threshold would simply flag every bright part of the face
# rather than the genuinely anomalous observations. sigma(x,y) is the standard
# deviation of that one pixel's own 64 residuals.
def outlier_mask(residuals, n_sigma=2.0):
    sigma = residuals.std(axis=0, keepdims=True)             # (1,192,168)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    return np.abs(residuals) > n_sigma * sigma               # (64,192,168) bool


# --- Dead frames -----------------------------------------------------------
# Two of the B02 exposures failed: image_028 peaks at 0.067 and image_052 at
# 0.012, i.e. they are essentially black everywhere. That is a capture fault,
# not lighting: 14 of the 64 lighting directions in every dataset point behind
# the subject (l_z < 0, up to 127 deg off the camera axis) and are therefore
# very dark on average, but they all still reach a peak near 1.0 from the rim
# highlight. Only these two never light anything.
#
# They have to be removed separately, before the residual test, because the
# 2-sigma rule does not reliably catch them. image_052 has l_z = -0.087, so the
# model already predicts almost no brightness for a front-facing pixel; the
# measurement is 0, the prediction is ~0, they agree, and only 4.9% of the
# frame is flagged - rank 26 of 64, barely above the 1.6% median. A failed
# capture whose lighting direction happens to predict darkness anyway is
# invisible to a residual test, yet it still biases the least-squares solve
# with a row of zeros.
def find_dead_frames(imgs, peak_thresh=0.25):
    """Frames that never illuminate anything, judged on peak not mean."""
    peaks = imgs.reshape(imgs.shape[0], -1).max(axis=1)
    return peaks < peak_thresh


for tag in tags:
    res = results[tag]
    dead = find_dead_frames(res['imgs'])
    res['dead'] = dead
    if dead.any():
        print('%s: dead frames %s (peak brightness %s)'
              % (tag, list(np.where(dead)[0]),
                 np.round(res['imgs'].reshape(64, -1).max(axis=1)[dead], 3)))

    res['residuals'] = compute_residuals(res['imgs'], res['light_dirs'],
                                         res['albedo'], res['normals'])
    # A dead frame is an outlier at every pixel, whatever its residual says.
    res['outliers'] = outlier_mask(res['residuals']) | dead[:, None, None]

    frac = res['outliers'].mean() * 100.0
    kept = (~res['outliers']).sum(axis=0)                    # per-pixel count
    print('%s: %.1f%% of observations flagged as outliers, '
          'fewest surviving views for any pixel = %d'
          % (tag, frac, kept.min()))

# Montage of the outlier masks: one panel per lighting condition, so all 64
# images of a face are shown at once. Flagged pixels are drawn in red over the
# image itself rather than as a bare binary mask, so it is possible to see
# *what* was rejected - the shadow boundaries, the specular highlight on the
# nose and forehead, the dark side of the face under grazing light.
#
# Many of the 64 exposures are very dark, so each panel is stretched to its own
# 99th percentile before display. This is for visibility only; every
# calculation uses the raw values.
def overlay_montage(imgs, masks, title):
    n_img = imgs.shape[0]
    fig, axes = plt.subplots(8, 8, figsize=(11, 12))
    for k, ax in enumerate(axes.flat):
        if k < n_img:
            scale = max(np.percentile(imgs[k], 99), 1e-3)
            grey = np.clip(imgs[k] / scale, 0.0, 1.0)
            rgb = np.stack([grey] * 3, axis=-1)
            rgb[masks[k]] = [1.0, 0.15, 0.15]        # flagged pixels in red
            ax.imshow(rgb)
            ax.set_title('%d  (%.0f%%)' % (k, 100.0 * masks[k].mean()),
                         fontsize=6, pad=1.5)
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


for tag in tags:
    res = results[tag]
    overlay_montage(res['imgs'], res['outliers'],
                    '%s - red = |residual| > 2 sigma  (panel number and '
                    'percentage of pixels flagged)' % tag)


# ---------------------------------------------------------------------------
# Re-solve for albedo and normals, ignoring the outliers
# ---------------------------------------------------------------------------
# Once every pixel keeps a different subset of the 64 images, the single shared
# lstsq of Step 0 no longer applies - each pixel now has its own system. Rather
# than looping over 32,256 pixels, form the weighted 3x3 normal equations
#
#     A g = b,    A = sum_k w_k l_k l_k^T,    b = sum_k w_k I_k l_k
#
# with w_k = 0 for a rejected observation and 1 otherwise. Each of the 9
# entries of A is one matrix-vector product over all pixels at once, and the
# whole stack of 3x3 systems is then solved in a single batched np.linalg.solve.
def solve_photometric_stereo_weighted(imgs, light_dirs, weights):
    n_img, h, w = imgs.shape
    N = h * w
    I = imgs.reshape(n_img, N)
    W = weights.reshape(n_img, N).astype(np.float64)

    A = np.empty((N, 3, 3))
    for i in range(3):
        for j in range(3):
            A[:, i, j] = (light_dirs[:, i] * light_dirs[:, j]) @ W
    b = (W * I).T @ light_dirs                               # (N,3)

    # A pixel with too few surviving views gives a singular (or near-singular)
    # system. Fall back to the unweighted solution there rather than letting a
    # blown-up solve poison the height map.
    det = np.linalg.det(A)
    bad = np.abs(det) < 1e-9
    A[bad] = np.eye(3)
    g = np.linalg.solve(A, b[:, :, None])[:, :, 0]           # (N,3)
    if bad.any():
        g_all, *_ = np.linalg.lstsq(light_dirs, I, rcond=None)
        g[bad] = g_all.T[bad]

    albedo = np.linalg.norm(g, axis=1)
    safe = np.where(albedo < 1e-8, 1e-8, albedo)
    normals = (g / safe[:, None]).reshape(h, w, 3)
    return albedo.reshape(h, w), normals, bad.sum()


print('')
for tag in tags:
    res = results[tag]
    albedo2, normals2, n_bad = solve_photometric_stereo_weighted(
        res['imgs'], res['light_dirs'], ~res['outliers'])
    z_a2, z_b2, z_c2 = heights_from_normals(normals2)
    res.update(albedo2=albedo2, normals2=normals2,
               z_a2=z_a2, z_b2=z_b2, z_c2=z_c2)

    # How far the normals moved, in degrees
    dot = np.clip((res['normals'] * normals2).sum(axis=2), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))

    # Disagreement between integration paths (a) and (b) is a measure of how
    # badly the integrability constraint dp/dy = dq/dx has been violated, so it
    # doubles as a self-consistency score for the recovered normals.
    rms0 = np.sqrt(np.mean((res['z_a'] - res['z_b']) ** 2))
    rms1 = np.sqrt(np.mean((z_a2 - z_b2) ** 2))
    print('%s: normals moved %.2f deg mean / %.1f deg max, %d singular pixels; '
          'RMS |z_a - z_b| %.2f -> %.2f px (%+.1f%%)'
          % (tag, ang.mean(), ang.max(), n_bad, rms0, rms1,
             100.0 * (rms1 - rms0) / rms0))

# Before / after comparison of the averaged height maps
fig, axes = plt.subplots(len(tags), 3, figsize=(11, 3.2 * len(tags)))
for r, tag in enumerate(tags):
    res = results[tag]
    panels = [(res['z_c'], 'baseline (c)'),
              (res['z_c2'], 'outliers rejected (c)'),
              (res['z_c2'] - res['z_c'], 'difference')]
    for c, (img, name) in enumerate(panels):
        ax = axes[r, c]
        im = ax.imshow(img, cmap='viridis' if c < 2 else 'coolwarm')
        ax.set_title('%s %s' % (tag, name), fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# 3D renders, baseline then outlier-rejected, for each face
for tag in tags:
    plot_face_3d(results[tag]['z_c'], results[tag]['albedo'])
    #plot_face_3d(results[tag]['z_c2'], results[tag]['albedo2'])
