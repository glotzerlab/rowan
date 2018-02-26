import numpy as np
import quaternion as quat
import os

# Create angles
alphas = np.pi*np.random.rand(100)
betas = np.pi*np.random.rand(100)
gammas = np.pi*np.random.rand(100)
angles = np.vstack((alphas, betas, gammas)).T

quats = []
for i in range(len(alphas)):
    quats.append(
            quat.as_float_array(quat.from_euler_angles(alphas[i], betas[i], gammas[i]))
            )
euler_arrays = np.asarray(quats)

# Load test files
TESTDATA_FILENAME = os.path.join(
    os.path.dirname(__file__),
    'tests/files/test_arrays.npz')
with np.load(TESTDATA_FILENAME) as data:
    tosave = dict(
        input1=data['input1'],
        input2 = data['input2'],
        rotated_vectors = data['rotated_vectors'],
        vector_inputs = data['vector_inputs'],
        product = data['product'],
        euler_angles = angles,
        euler_quats = euler_arrays,
        )

np.savez_compressed(
        os.path.join(
            os.path.dirname(__file__),
            'tests/files/test_arrays2.npz'
            ),
        **tosave
    )
