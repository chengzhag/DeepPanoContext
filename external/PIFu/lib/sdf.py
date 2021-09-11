import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1])):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4, dtype=np.float32)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    coords = coords.reshape(3, resX, resY, resZ).astype(np.float32)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512, batch_size=1):
    num_pts = points.shape[1]
    sdf = np.zeros([batch_size, num_pts], dtype=np.float32)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf_batch = eval_func(points[:, i * num_samples:i * num_samples + num_samples])
        sdf[:, i * num_samples:i * num_samples + num_samples] = sdf_batch
    if num_pts % num_samples:
        sdf_batch = eval_func(points[:, num_batches * num_samples:])
        sdf[:, num_batches * num_samples:] = sdf_batch

    return sdf


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512, batch_size=1):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples, batch_size=batch_size)
    return sdf.reshape(-1, *resolution)
