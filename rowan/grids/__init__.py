r"""Low-dispersion grids on manifolds related to rotation.

Finding a uniform of `n` rotations (in any representation) is a generalized Tammes
problem on the manifold associated with that representation. Grids of rotation axes
reduce to the standard Tammes problem on the 2-sphere, and grids of quaternions are a
generalization to the 3-sphere. Closed form solutions for general `n` are not possible,
but we can create nearly uniform (low-dispersion) grids using Fibonacci lattices.
"""

import numpy as np

"""The Golden ratio, 1.61803398875..."""
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


"""The angle that divides 2π radians into the golden ratio.
This takes the longer of the two segments π(√5 - 1), where π[(√5 - 1) + (3-√5)] = 2π
"""
GOLDEN_ANGLE = np.pi * (np.sqrt(5) - 1)


def spherical_fibonacci_lattice(n):
    """A Fibonacci lattice of `n` points on the 2-sphere.

    Args:
        n (int): The number of points in the lattice.
    """  # TODO: example image!
    t = np.arange(n)

    z = 1 - (t / (n - 1)) * 2
    radius = np.sqrt(1 - z**2)
    theta = (GOLDEN_ANGLE * t) % (2 * np.pi)

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    return np.column_stack([x, y, z]).T


def quaternion_fibonacci_lattice(n):
    """A near-uniform grid of `n` quaternions.

    This is equivalent to a Fibonacci lattice on the 3-sphere. See
    `this paper <https://ieeexplore.ieee.org/document/9878746>`_ for a derivation.

    Args:
        n (int): The number of points in the lattice.
    """
    PSI = 1.533751168755204288118041413  # Solution to ψ**4 = ψ + 4
    s = np.arange(n) + 1 / 2
    t = s / n
    d = 2 * np.pi * s
    r0, r1 = (np.sqrt(t), np.sqrt(1 - t))
    α, β = (d / np.sqrt(2), d / PSI)

    # Allocate as rows and then transpose, rather than stacking columns
    result = np.empty((4, n))
    result[...] = r0 * np.sin(α), r0 * np.cos(α), r1 * np.sin(β), r1 * np.cos(β)
    return result.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import spatula

    import rowan

    n = 2000
    q = quaternion_fibonacci_lattice(n)

    dists = rowan.geometry.sym_intrinsic_distance(q, q[:, None])
    dists[np.diag_indices(n)] = np.nan

    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=False)
    # , subplot_kw={"projection": "3d"})

    print(dists[np.isfinite(dists)])
    ax[0].hist(np.nanmedian(dists, axis=0), bins=64, density=True, color="#71618D")
    ax[0].set(title="Median Distance (Fib)", yticks=[])

    qlat = spatula.optimize.Mesh.from_grid(n_axes=100, n_angles=20)._points
    latists = rowan.geometry.sym_intrinsic_distance(qlat, qlat[:, None])
    latists[np.diag_indices(n)] = np.nan
    ax[1].hist(np.nanmedian(latists, axis=0), bins=64, density=True, color="#71618D")
    ax[1].set(title="Median Distance (Tammes)", yticks=[])

    qrand = rowan.random.rand(n)
    randists = rowan.geometry.sym_intrinsic_distance(qrand, qrand[:, None])
    randists[np.diag_indices(n)] = np.nan
    ax[2].hist(np.nanmedian(randists, axis=0), bins=64, density=True, color="#71618D")
    ax[2].set(title="Median Distance (rand)", yticks=[])
    plt.show()
