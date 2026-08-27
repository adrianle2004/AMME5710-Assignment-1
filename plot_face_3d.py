import numpy as np
from matplotlib import pyplot as plt

"""
plot_face_3d: produces textured 3D mesh of face data

height_map: [192x168] numpy array of heights
albedo: [192x168] numpy array of albedos (between 0-1)

Note: x-axis displayed in figure is flipped for better viewing
"""
def plot_face_3d(height_map, albedo):

    h, w = albedo.shape[:2]

    X, Y = np.meshgrid(np.arange(w, 0, -1), np.arange(0, h))
    fc = np.empty([h, w, 3])
    fc[:,:,0] = albedo
    fc[:,:,1] = albedo
    fc[:,:,2] = albedo

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, height_map, facecolors=fc, rstride=1, cstride=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis('equal')

    plt.show()
