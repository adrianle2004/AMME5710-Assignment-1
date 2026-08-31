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
# Step 0: images + lighting -> albedo and surface normals
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
# Step 1: surface normals -> surface gradients
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
# Step 2: integrate the gradients into a height map
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
    results[tag] = dict(albedo=albedo_image, normals=surface_normals,
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

# 3D render of each face using the averaged height map
for tag in tags:
    plot_face_3d(results[tag]['z_c'], results[tag]['albedo'])
