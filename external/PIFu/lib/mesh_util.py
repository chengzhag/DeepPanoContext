from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid
from skimage import measure
import trimesh


def reconstruction(structured_implicit,
                   resolution, b_min, b_max,
                   num_samples=10000):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param structured_implicit: a StructuredImplicit object.
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max)

    # Then we define the lambda function for cell evaluation
    def eval_func(points, structured_implicit):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=structured_implicit.device).float()
        samples = samples.transpose(-1, -2)
        samples = samples.expand(structured_implicit.batch_size, -1, -1)
        pred = structured_implicit.sdf_at_samples(samples)[..., 0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    sdf = eval_grid(coords, lambda p:eval_func(p, structured_implicit),
                    num_samples=num_samples, batch_size=structured_implicit.batch_size)

    # Finally we do marching cubes
    mesh = []
    for s in sdf:
        try:
            verts, faces, _, _ = measure.marching_cubes(s, -0.07)
            # transform verts into world coordinate system
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
            verts = verts.T
            mesh.append(trimesh.Trimesh(vertices=verts, faces=faces))
        except (ValueError, RuntimeError) as e:
            print('Failed to extract mesh with error %s. Setting to unit sphere.' % repr(e))
            mesh.append(trimesh.primitives.Sphere(radius=0.5))
    return mesh
