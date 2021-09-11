import numpy as np
import trimesh
import torch
from shapely.geometry import Polygon, Point

from gibson2.utils.mesh_util import homotrans, lookat

from configs import data_config
from utils.mesh_utils import normalize_to_unit_square
from utils.basic_utils import recursively_to, get_any_array


def vector_rotation(v1, v2, axis, left_hand=False, range_pi=False):
    """
    Calculate rotation around axis in rad from v1 to v2, where v1 and v2 are 3-dim vectors.
    The rotation is counter-clockwise when axis is pointing at viewer,
    As defined in right-handed coordinate system.

    Parameters
    ----------
    v1: n x 3 numpy array or tensor
    v2: n x 3 numpy array or tensor

    Returns
    -------
    n-dim vector in the range of [0, 2 * pi)
    """
    if isinstance(v1, torch.Tensor):
        backend, atan2 = torch, torch.atan2
    else:
        backend, atan2 = np, np.arctan2

    ori = atan2((backend.cross(v2, v1) * axis).sum(axis=-1), (v2 * v1).sum(axis=-1)) * (1. if left_hand else -1.)
    if not range_pi:
        ori = backend.remainder(ori, np.pi * 2)

    return ori


def point_polygon_dis(points, polygon):
    backend = torch if isinstance(points, torch.Tensor) else np
    if (polygon[0] != polygon[-1]).any():
        polygon = backend.cat([polygon, polygon[:1]], 0)
    dis = backend.zeros([len(points), len(polygon) - 1], dtype=points.dtype)
    if backend == torch:
        dis = dis.to(points.device)

    # find distance to each line segment
    for i_line, (p1, p2) in enumerate(zip(polygon[:-1], polygon[1:])):
        dis[:, i_line] = point_line_segment_dis(points, p1, p2)

    # use nearest distance
    dis = dis.min(axis=-1)
    if backend == torch:
        dis = dis[0]

    # points inside room layout should have negative distance
    layout_2d = Polygon(polygon)
    inside_layout = [layout_2d.contains(Point(c)) for c in points]
    dis[inside_layout] *= -1

    return dis


def point_line_segment_dis(points, line_start, line_end):
    backend = torch if isinstance(points, torch.Tensor) else np
    line_vec = line_end - line_start
    r = (line_vec * (points - line_start)).sum(-1) / backend.linalg.norm(line_vec, axis=-1) ** 2
    line_length = backend.linalg.norm(line_vec, axis=-1)
    dis_start = backend.linalg.norm(points - line_start, axis=-1)
    dis_end = backend.linalg.norm(points - line_end, axis=-1)
    dis_line = backend.sqrt(backend.abs(dis_start.pow(2) - (r * line_length).pow(2)) + 1e-8)
    dis_start = dis_start * (r < 0)
    dis_end = dis_end * (r > 1)
    dis_line = dis_line * ((r <= 1) & (r > 0))
    return dis_line + dis_start + dis_end


def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def num2bins(bins, loc):
    '''
    Given bins and value, compute where the value locates and the distance to the center.

    :param bins: list
    The bins, eg. [[-x, 0], [0, x]]
    :param loc: float
    The location
    :return cls: int, bin index.
    indicates which bin is the location for classification.
    :return reg: float, [-0.5, 0.5].
    the distance to the center of the corresponding bin.
    '''
    if bins.ndim == 1:
        backend = torch if isinstance(loc, torch.Tensor) else np
        dist = [backend.abs(loc - b) for b in bins]
        dist = backend.stack(dist, -1)
        cls = backend.argmin(dist, -1)
        return cls
    else:
        width_bin = bins[0][1] - bins[0][0]
        # get the distance to the center from each bin.
        if isinstance(loc, torch.Tensor):
            dist = [torch.abs(loc - (bn[0] + bn[1]) / 2) for bn in bins]
            dist = torch.stack(dist, -1)
            cls = torch.argmin(dist, -1)
            bins = torch.tensor(bins, device=loc.device)
            reg = (loc - bins[cls].mean(-1)) / width_bin
        else:
            dist = ([float(abs(loc - float(bn[0] + bn[1]) / 2)) for bn in bins])
            cls = dist.index(min(dist))
            reg = float(loc - float(bins[cls][0] + bins[cls][1]) / 2) / float(width_bin)
    return cls, reg


def label_or_num_from_cls_reg(cls, reg=None, bins=None, return_score=False, threshold=0.5):
    if isinstance(cls, torch.Tensor):
        if cls.dtype == torch.float32:
            if cls.shape[-1] == 1:
                cls = cls.squeeze(-1)
                score = torch.sigmoid(cls)
                label = score > threshold
            else:
                score = torch.softmax(cls, dim=-1)
                score, label = score.max(-1)
        else:
            label = cls
            score = torch.ones_like(label, device=cls.device, dtype=torch.float32)
    else:
        label = cls
        score = np.ones_like(label, dtype=np.float32)

    if bins is None:
        if cls.shape[-1] == 2 and reg is None:
            if isinstance(cls, torch.Tensor):
                bin_center = label.type(torch.bool)
            else:
                bin_center = label.astype(np.bool)
        else:
            bin_center = label
    else:
        if bins.ndim == 1:
            bin_center = bins[label]
        else:
            bin_width = (bins[0][1] - bins[0][0])
            bin_center = (bins[label, 0] + bins[label, 1]) / 2

    if reg is None:
        return (bin_center, score) if return_score else bin_center

    if label is not cls:
        reg = torch.gather(reg, 1, label.unsqueeze(-1)).squeeze(1)
    num = bin_center + reg * bin_width
    return (num, score) if return_score else num


def size2reg(size, class_id=None, avg_key='size_avg'):
    size_avg = data_config.metadata[avg_key]
    if class_id is not None:
        size_avg = size_avg[class_id]
    if isinstance(size, torch.Tensor):
        size_avg = torch.FloatTensor(size_avg).to(size.device)
    size_residual = size / size_avg - 1
    return size_residual


def bins2layout(layout_total3d):
    lo_ori_reg, lo_ori_cls, centroid_reg, size_reg = \
        layout_total3d['ori_reg'], layout_total3d['ori_cls'], \
        layout_total3d['centroid_reg'], layout_total3d['size_reg']
    bins = recursively_to(data_config.metadata, dtype='tensor', device=lo_ori_reg.device)
    cuboid_layout = {
        'ori': label_or_num_from_cls_reg(lo_ori_cls, lo_ori_reg, bins['layout_ori_bins']),
        'centroid_total3d': centroid_reg + bins['layout_centroid_avg'],
        'size': (size_reg + 1) * bins['layout_size_avg']
    }
    return cuboid_layout


def bins2camera(layout_total3d):
    pitch_cls, pitch_reg, roll_cls, roll_reg = \
        layout_total3d['pitch_cls'], layout_total3d['pitch_reg'], \
        layout_total3d['roll_cls'], layout_total3d['roll_reg']
    pitch_bins = torch.FloatTensor(data_config.metadata['pitch_bins']).to(pitch_cls.device)
    roll_bins = torch.FloatTensor(data_config.metadata['roll_bins']).to(pitch_cls.device)
    return {
        'pitch': label_or_num_from_cls_reg(pitch_cls, pitch_reg, pitch_bins),
        'roll': label_or_num_from_cls_reg(roll_cls, roll_reg, roll_bins),
    }


def bins2bdb3d(data):
    bdb3d_pix = {}
    objs = data['objs']
    transform = IGTransform(data)
    if 'K' in data['camera']:
        bdb2d, bdb3d = objs['bdb2d'], objs['bdb3d']
        bdb2d_center = torch.stack([bdb2d['x1'] + bdb2d['x2'], bdb2d['y1'] + bdb2d['y2']], 1) / 2
        bdb2d_wh = torch.stack([bdb2d['x2'] - bdb2d['x1'], bdb2d['y2'] - bdb2d['y1']], 1)
        bdb3d_pix['center'] = bdb2d_center - bdb2d_wh * objs['delta2d']
        dis_name = 'dis' # try to regress dis instead of depth in Total3D
    else:
        bfov, bdb3d = objs['bfov'], objs['bdb3d']
        bfov_center = torch.stack([bfov['lon'], bfov['lat']], 1)
        bfov_wh = torch.stack([bfov['x_fov'], bfov['y_fov']], 1)
        bdb3d_pix['center'] = transform.camrad2pix(bfov_center - bfov_wh * objs['delta2d'])
        dis_name = 'dis'

    bins = recursively_to(data_config.metadata, dtype='tensor', device=bdb3d_pix['center'].device)
    size_avg, dis_bins, ori_bins = bins['size_avg'], bins['dis_bins'], bins['ori_bins']
    bdb3d_pix['size'] = (bdb3d['size_reg'] + 1) * size_avg[objs['label'], :]
    bdb3d_pix[dis_name], bdb3d_pix[dis_name + '_score'] = label_or_num_from_cls_reg(
        bdb3d['dis_cls'], bdb3d['dis_reg'], dis_bins, return_score=True)
    bdb3d_pix['ori'], bdb3d_pix['ori_score'] = label_or_num_from_cls_reg(
        bdb3d['ori_cls'], bdb3d['ori_reg'], ori_bins, return_score=True)

    return bdb3d_pix


def bdb3d_corners(bdb3d: (dict, np.ndarray)):
    """
    Get ordered corners of given 3D bounding box dict or disordered corners

    Parameters
    ----------
    bdb3d: 3D bounding box dict

    Returns
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    if isinstance(bdb3d, np.ndarray):
        centroid = np.mean(bdb3d, axis=0)
        z = bdb3d[:, -1]
        surfaces = []
        for surface in (bdb3d[z < centroid[-1]], bdb3d[z >= centroid[-1]]):
            surface_2d = surface[:, :2]
            center_2d = centroid[:2]
            vecters = surface_2d - center_2d
            angles = np.arctan2(vecters[:, 0], vecters[:, 1])
            orders = np.argsort(-angles)
            surfaces.append(surface[orders][(0, 1, 3, 2), :])
        corners = np.concatenate(surfaces)
    else:
        corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
                                axis=1, bitorder='little', count=-5).astype(np.float32)
        corners = corners - 0.5
        if isinstance(bdb3d['size'], torch.Tensor):
            corners = torch.from_numpy(corners).to(bdb3d['size'].device)
        corners = IGTransform.obj2frame(corners, bdb3d)
    return corners


def expand_bdb3d(bdb3d, dis):
    bdb3d = bdb3d.copy()
    size = bdb3d['size']
    size = size + dis * 2
    size[size <= 0.01] = 0.01
    bdb3d['size'] = size
    return bdb3d


def bdb3d_from_front_face(front_face, length):
    """
    Get 3D bounding box dict from given front face and length

    Parameters
    ----------
    front_face: four 3D corners of front face
        right-forward-down
        left-forward-down
        right-forward-up
        left-forward-up
    length: length along y axis (forward-backward axis)

    Returns
    -------
    bdb3d dict
    """
    up = front_face[2] - front_face[0] # z
    left = front_face[1] - front_face[0] # x
    back = np.cross(up, left)
    back = back / np.linalg.norm(back) * length # y

    basis = np.stack([left, back, up])
    size = np.linalg.norm(basis, axis=1)
    basis = basis.T / size
    centroid = front_face.sum(0) / 4 + back / 2

    return {
        'centroid': centroid,
        'basis': basis,
        'size': size
    }


def bdb3d_from_corners(corners: np.ndarray):
    front_face = corners[(0, 1, 4, 5), :]
    length = np.linalg.norm(corners[0] - corners[3])
    bdb3d = bdb3d_from_front_face(front_face, length)
    return bdb3d


bdb3d_axis_map = {'forward': [0, -1, 0], 'back': [0, 1, 0], 'left': [1, 0, 0], 'right': [-1, 0, 0],
                  'up': [0, 0, 1], 'down': [0, 0, -1], 'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
def bdb3d_axis(bdb3d, axis='forward'):
    basis = bdb3d['basis']
    axis_obj = np.array(bdb3d_axis_map[axis], dtype=np.float32)
    if isinstance(basis, torch.Tensor):
        axis_obj = torch.tensor(axis_obj, dtype=basis.dtype, device=basis.device)
    axis_ori =  basis @ axis_obj
    return axis_ori


cam_axis_map = {'forward': [0, 0, 1], 'back': [0, 0, 1], 'left': [-1, 0, 0], 'right': [1, 0, 0],
                'up': [0, -1, 0], 'down': [0, 1, 0], 'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
def cam_axis(camera=None, axis='forward'):
    axis_cam3d = np.array(cam_axis_map[axis], dtype=np.float32)
    if camera is not None:
        cam3d2world = camera['cam3d2world']
        if isinstance(cam3d2world, torch.Tensor):
            axis_cam3d = torch.tensor(axis_cam3d, dtype=cam3d2world.dtype, device=cam3d2world.device)
            axis_cam3d = axis_cam3d[None, -1, None].expand(len(cam3d2world), -1, -1)
        axis_cam3d = cam3d2world[..., :3, :3] @ axis_cam3d
    return axis_cam3d


def points2bdb2d(points):
    points = np.stack([points['x'], points['y']]).T if isinstance(points, dict) else points
    if isinstance(points, torch.Tensor):
        xy_max = torch.max(points, -2)[0]
        xy_min = torch.min(points, -2)[0]
    else:
        xy_max = points.max(-2)
        xy_min = points.min(-2)
    return {
        'x1': xy_min[..., 0],
        'x2': xy_max[..., 0],
        'y1': xy_min[..., 1],
        'y2': xy_max[..., 1]
    }


def contour2bfov(contour, height=None, width=None, camera=None):
    if camera is None:
        camera = {'height': height, 'width': width}
    transform = IGTransform({'camera': camera})

    contour_np = np.stack([contour['x'], contour['y']]).T if isinstance(contour, dict) else contour
    bdb2d = points2bdb2d(contour_np)

    center_pix = np.array([(bdb2d['x1'] + bdb2d['x2']), (bdb2d['y1'] + bdb2d['y2'])], dtype=np.float32) / 2
    center_rad = transform.campix2rad(center_pix)
    center_world = transform.campix2world(center_pix, 1.)
    contour_world = transform.campix2world(contour_np, 1.)

    transform = transform.copy()
    transform.look_at(center_world)
    contour_pers3d = transform.world2cam3d(contour_world)
    contour_rad = transform.cam3d2rad(contour_pers3d)
    min_rad = contour_rad.min(axis=0)
    max_rad = contour_rad.max(axis=0)
    fov_rad = np.max(np.abs(np.stack([max_rad, min_rad])), 0) * 2
    bfov = {'lon': center_rad[0], 'lat': center_rad[1], 'x_fov': fov_rad[0], 'y_fov': fov_rad[1]}
    bfov = {k: float(v) for k, v in bfov.items()}
    return bfov


class IGTransform:
    """
    3D transformations for iGibson data
    world: right-hand coordinate of iGibson (z-up)
    cam3d: x-right, y-down, z-forward
    cam2d: x-right, y-down
    object: x-left, y-back, z-up (defined by iGibson)
    """
    def __init__(self, data: dict=None, split='objs'):
        self.data = data
        self.camera = data['camera'] if data else {}
        self.split = split
        if isinstance(self.camera, dict) and self.camera \
                and 'world2cam3d' not in self.camera and 'cam3d2world' not in self.camera:
            if any(k not in self.camera for k in ('pos', 'target', 'up')):
                self.set_camera_to_world_center()
            else:
                self.look_at()

    @classmethod
    def level_look_at(cls, data, target):
        data = data.copy()
        camera = data['camera'].copy()
        if isinstance(target, torch.Tensor):
            target = target.clone()
        else:
            target = target.copy()
        camera['target'] = target
        camera['target'][..., -1] = camera['target'][..., -1]
        data['camera'] = camera
        recentered_trans = cls(data)
        return recentered_trans

    @classmethod
    def world_centered(cls, camera):
        transform_centered = cls()
        transform_centered.set_camera_to_world_center()
        sample = get_any_array(camera)
        if isinstance(sample, torch.Tensor):
            transform_centered.camera = recursively_to(
                transform_centered.camera, dtype='tensor', device=sample.device)
        transform_centered.camera.update({k: camera[k][0] for k in ('height', 'width')})
        return transform_centered

    def look_at(self, camera=None):
        if camera is not None:
            if isinstance(camera, dict):
                self.camera = camera.copy()
            else:
                self.camera['target'] = camera
        world2cam3d = lookat(self.camera['pos'], self.camera['target'], self.camera['up'])
        world2cam3d = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) @ world2cam3d
        self.camera['world2cam3d'] = world2cam3d
        self.camera['cam3d2world'] = np.linalg.inv(world2cam3d)
        return self

    def get_camera_angle(self):
        """
        Get the yaw, pitch, roll angle from the camera.
        The initial state of camera is defined as:
        world_x-forward, world_y-left, world_z-up
        # The rotation is right-handed around world frame (?) with the following order (?):
        # yaw-world_z, pitch-world_y, roll-world_z

        Returns
        -------
        yaw, pitch, roll angles in rad
        """
        R = self['cam3d2world']
        backend, atan2 = (torch, torch.atan2) if isinstance(R, torch.Tensor) else (np, np.arctan2)
        yaw = atan2(R[..., 1, 2], R[..., 0, 2])
        pitch = - atan2(R[..., 2, 2], backend.sqrt(R[..., 0, 2] ** 2 + R[..., 1, 2] ** 2))
        roll = atan2(R[..., 2, 0], - R[..., 2, 1])
        return yaw, pitch, roll

    def set_camera_angle(self, yaw, pitch, roll):
        """
        Set camera rotation from yaw, pitch, roll angles in rad.

        Parameters
        ----------
        yaw, pitch, roll angles in rad
        """
        pitch = -pitch
        roll = -roll
        R = self.camera['cam3d2world']
        use_torch = isinstance(R, torch.Tensor)
        R = R.clone() if use_torch else R.copy()
        backend, inverse = (torch, torch.inverse) if use_torch else (np, np.linalg.inv)
        if use_torch:
            yaw, pitch, roll = [torch.tensor(v) if v is not torch.Tensor else v for v in (yaw, pitch, roll)]
        R[..., 0, 2] = backend.cos(yaw) * backend.cos(pitch)
        R[..., 0, 1] = - backend.sin(yaw) * backend.sin(roll) + backend.cos(yaw) * backend.cos(roll) * backend.sin(pitch)
        R[..., 0, 0] = backend.cos(roll) * backend.sin(yaw) + backend.cos(yaw) * backend.sin(pitch) * backend.sin(roll)
        R[..., 2, 2] = backend.sin(pitch)
        R[..., 2, 1] = - backend.cos(pitch) * backend.cos(roll)
        R[..., 2, 0] = - backend.cos(pitch) * backend.sin(roll)
        R[..., 1, 2] = backend.cos(pitch) * backend.sin(yaw)
        R[..., 1, 1] = backend.cos(yaw) * backend.sin(roll) + backend.cos(roll) * backend.sin(yaw) * backend.sin(pitch)
        R[..., 1, 0] = - backend.cos(yaw) * backend.cos(roll) + backend.sin(yaw) * backend.sin(pitch) * backend.sin(roll)
        self.camera['cam3d2world'] = R
        self.camera['world2cam3d'] = inverse(R)

        target = self.camera['target']
        target = target.clone() if use_torch else target.copy()
        target[..., :2] = 0
        target[..., 2] = 1
        self.camera['target'] = self.cam3d2world(target)

        up = self.camera['up']
        up = up.clone() if use_torch else up.copy()
        up[..., (0, 2)] = 0
        up[..., 1] = -1
        self.camera['up'] = (self.camera['cam3d2world'][..., :3, :3] @ up[..., None])[..., 0]
        return self

    def copy(self):
        data = self.data.copy()
        data['camera'] = self.camera.copy()
        return IGTransform(data, split=self.split)

    def set_camera_to_world_center(self):
        self.camera['pos'] = np.zeros(3, np.float32)
        self.camera['target'] = np.array([1, 0, 0], np.float32)
        self.camera['up'] = np.array([0, 0, 1], np.float32)
        self.look_at()
        return self

    def set_camera_like_total3d(self):
        # set world frame coordinate to camera center
        # and rotate x axis to camera y-z plane around z axis
        pos = self.camera['pos']
        use_torch = isinstance(pos, torch.Tensor)
        inverse = torch.inverse if use_torch else np.linalg.inv

        self.camera['pos'] = pos.clone() if use_torch else pos.copy()
        self.camera['pos'][:] = 0

        cam3d2world = self.camera['cam3d2world']
        cam3d2world = cam3d2world.clone() if isinstance(pos, torch.Tensor) else cam3d2world.copy()
        cam3d2world[..., :3, 3] = 0
        self.camera['cam3d2world'] = cam3d2world
        self.camera['world2cam3d'] = inverse(cam3d2world)

        _, pitch, roll = self.get_camera_angle()
        self.set_camera_angle(0., pitch, roll)
        return self

    def set_camera_level(self):
        self.camera['target'][-1] = self.camera['pos'][-1]
        self.camera['up'] = np.array([0, 0, 1], np.float32)
        self.look_at()
        return self

    def camrad2pix(self, camrad):
        """
        Transform longitude and latitude of a point to panorama pixel coordinate.

        Parameters
        ----------
        camrad: n x 2 numpy array

        Returns
        -------
        n x 2 numpy array of xy coordinate in pixel
        x: (left) 0 --> (width - 1) (right)
        y: (up) 0 --> (height - 1) (down)
        """
        if 'K' in self.camera:
            raise NotImplementedError
        if isinstance(camrad, torch.Tensor):
            campix = torch.empty_like(camrad, dtype=torch.float32)
        else:
            campix = np.empty_like(camrad, dtype=np.float32)
        width, height = self['width'], self['height']
        if isinstance(camrad, torch.Tensor):
            width, height = [x.view([-1] + [1] * (camrad.dim() - 2)) for x in (width, height)]
        campix[..., 0] = camrad[..., 0] * width / (2. * np.pi) + width / 2. + 0.5
        campix[..., 1] = camrad[..., 1] * height / np.pi + height / 2. + 0.5
        return campix

    def campix2rad(self, campix):
        backend, atan2 = (torch, torch.atan2) if isinstance(campix, torch.Tensor) else (np, np.arctan2)
        camrad = backend.empty_like(campix, dtype=backend.float32)
        if 'K' in self.camera:
            camrad[..., 0] = atan2(
                campix[..., 0] - self['K'][..., 0, 2],
                self['K'][..., 0, 0]
            )
            camrad[..., 1] = atan2(
                campix[..., 1] - self['K'][..., 1, 2],
                backend.sqrt(self['K'][..., 0, 0] ** 2 + (campix[..., 0] - self['K'][..., 0, 2]) ** 2)
                / self['K'][..., 0, 0] * self['K'][..., 1, 1]
            )
        else:
            width, height = self['width'], self['height']
            camrad[..., 0] = (campix[..., 0] - width / 2. - 0.5) / width * (2. * np.pi)
            camrad[..., 1] = (campix[..., 1] - height / 2. - 0.5) / height * np.pi
        return camrad

    def cam3d2rad(self, cam3d):
        """
        Transform 3D points in camera coordinate to longitude and latitude.

        Parameters
        ----------
        cam3d: n x 3 numpy array or bdb3d dict

        Returns
        -------
        n x 2 numpy array of longitude and latitude in radiation
        first rotate left-right, then rotate up-down
        longitude: (left) -pi -- 0 --> +pi (right)
        latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
        """
        backend, atan2 = (torch, torch.atan2) if isinstance(cam3d, torch.Tensor) else (np, np.arctan2)
        lon = atan2(cam3d[..., 0], cam3d[..., 2])
        lat = backend.arcsin(cam3d[..., 1] / backend.linalg.norm(cam3d, axis=-1))
        return backend.stack([lon, lat], -1)

    def camrad23d(self, rad, dis):
        backend = torch if isinstance(rad, torch.Tensor) else np
        proj_dis = backend.cos(rad[..., 1]) * dis
        x = backend.sin(rad[..., 0]) * proj_dis
        y = backend.sin(rad[..., 1]) * dis
        z = backend.cos(rad[..., 0]) * proj_dis
        cam3d = backend.stack([x, y, z]).T
        return cam3d

    def camrad2world(self, rad, dis):
        return self.cam3d2world(self.camrad23d(rad, dis))

    def world2camrad(self, world):
        return self.cam3d2rad(self.world2cam3d(world))

    def cam3d2pix(self, cam3d):
        """
        Transform 3D points from camera coordinate to pixel coordinate.

        Parameters
        ----------
        cam3d: n x 3 numpy array or bdb3d dict

        Returns
        -------
        for 3D points: n x 2 numpy array of xy in pixel.
        x: (left) 0 --> width - 1 (right)
        y: (up) 0 --> height - 1 (down)
        """
        if isinstance(cam3d, dict):
            campix = self.world2campix(self.cam3d2world(cam3d))
        else:
            if 'K' in self.camera:
                campix = self.transform(self.camera['K'], cam3d)
            else:
                campix = self.camrad2pix(self.cam3d2rad(cam3d))
        return campix

    def campix23d(self, campix, dis=None):
        if isinstance(campix, dict) and dis is None:
            cam3d = self.world2cam3d(self.campix2world(campix, dis))
        else:
            cam3d = self.camrad23d(self.campix2rad(campix), dis)
        return cam3d

    @staticmethod
    def transform(transform_matrix, input):
        """
        Transform 3D points or 3D bounding boxes with given transformation matrix.

        Parameters
        ----------
        transform_matrix: 4 x 4 transformation matrix
        input: n x 3 numpy array or bdb3d dict or Trimesh

        Returns
        -------
        n x 3 numpy array or bdb3d dict
        """
        if isinstance(input, trimesh.Trimesh):
            input = input.copy()
            input.vertices = IGTransform.transform(transform_matrix, input.vertices)
            return input
        elif isinstance(input, dict):
            size = input['size']
            if isinstance(size, torch.Tensor):
                size = size.clone()
            else:
                size = size.copy()
            output = {
                'centroid': IGTransform.transform(transform_matrix, input['centroid']),
                'basis': transform_matrix[..., :3, :3] @ input['basis'],
                'size': size
            }
        else:
            output = IGTransform.homotrans(transform_matrix, input)
        return output

    def world2cam3d(self, world):
        """
        Transform 3D points or 3D bounding boxes from world coordinate frame to camera coordinate frame.
        world: right-hand coordinate of iGibson (z-up)
        cam3d: x-right, y-down, z-forward

        Parameters
        ----------
        cam3d: n x 3 numpy array or bdb3d dict

        Returns
        -------
        n x 3 numpy array or bdb3d dict
        """

        return self.transform(self['world2cam3d'], world)

    def cam3d2world(self, cam3d):
        return self.transform(self['cam3d2world'], cam3d)

    def ori2basis(self, ori, center=None):
        if isinstance(ori, dict):
            if isinstance(ori['size'], torch.Tensor):
                if 'centroid' in ori:
                    centroid = ori['centroid'].clone()
                size = ori['size'].clone()
                centroid_total3d = ori['centroid_total3d'].clone()
            else:
                if 'centroid' in ori:
                    centroid = ori['centroid'].copy()
                centroid_total3d = ori['centroid_total3d'].copy()
                size = ori['size'].copy()

            if 'centroid_total3d' in ori and 'K' in self.camera:
                trans_centered = self.copy()
                trans_centered.set_camera_like_total3d()
                centroid_cam3d = trans_centered.world2cam3d(centroid_total3d)
                centroid = self.cam3d2world(centroid_cam3d)

            basis = {
                'basis': self.ori2basis(ori['ori']),
                'centroid': centroid,
                'centroid_total3d': centroid_total3d,
                'size': size
            }
        else:
            backend = torch if isinstance(ori, torch.Tensor) else np

            cam_yaw, _, _ = self.get_camera_angle()
            if 'K' in self.camera:
                yaw = cam_yaw - ori
            else:
                lon = self.campix2rad(center)[..., 0]
                yaw = cam_yaw - lon - ori # ori and lon are counter-clockwise about z axis (from above)
            yaw += np.pi / 2 # z axis of the camera rotates from x axis of the world coordinate

            if isinstance(yaw, torch.Tensor):
                basis = torch.zeros((len(yaw), 3, 3), device=yaw.device)
            else:
                basis = np.zeros((3, 3), dtype=np.float32)

            basis[..., 0, 0] = backend.cos(yaw)
            basis[..., 0, 1] = - backend.sin(yaw)
            basis[..., 1, 0] = backend.sin(yaw)
            basis[..., 1, 1] = backend.cos(yaw)
            basis[..., 2, 2] = 1
        return basis

    def campix2world(self, campix, dis=None):
        if isinstance(campix, dict):
            if isinstance(campix['size'], torch.Tensor):
                size = campix['size'].clone()
            else:
                size = campix['size'].copy()

            world = {
                'centroid': self.campix2world(campix['center'], campix['dis']),
                'basis': self.ori2basis(campix['ori'], campix.get('center')),
                'size': size
            }
        else:
            world = self.cam3d2world(self.campix23d(campix, dis=dis))
        return world

    def basis2ori(self, basis, centroid=None):
        """
        Transform basis to ori based on different definitions of ori for panoramic and perspective image.
        Orientation: Defined as the left-handed rotation from line_of_sight to forward vector of object
                     around up axis of the world frame.
        line_of_sight: For panoramic image, is defined as the direction from camera center to object centroid.
                       For perspective image, is defined as the direction of camera forward.

        Parameters
        ----------
        basis: Basis rotation matrix or bdb3d dict
        centroid: For panoramic image, the centroid of object is required

        Returns
        -------
        Orientation in rad or bdb3d dict.
        When output bdb3d, it will also include a parameter 'centroid_total3d',
        indicating the centroid of bdb3d in Total3D frame
        """

        if isinstance(basis, dict):
            if isinstance(basis['size'], torch.Tensor):
                if 'centroid_total3d' in basis:
                    centroid_total3d = basis['centroid_total3d'].clone()
                centroid = basis['centroid'].clone()
                size = basis['size'].clone()
            else:
                if 'centroid_total3d' in basis:
                    centroid_total3d = basis['centroid_total3d'].copy()
                centroid = basis['centroid'].copy()
                size = basis['size'].copy()

            if 'centroid' in basis and 'K' in self.camera:
                trans_centered = self.copy()
                trans_centered.set_camera_like_total3d()
                centroid_cam3d = self.world2cam3d(centroid)
                centroid_total3d = trans_centered.cam3d2world(centroid_cam3d)

            ori = {
                'ori': self.basis2ori(basis['basis']),
                'centroid': centroid,
                'centroid_total3d': centroid_total3d,
                'size': size
            }
        else:
            obj_forward = bdb3d_axis({'basis': basis})
            if 'K' in self.camera:
                # use the definition of orientation in Total3D
                line_of_sight = self.camera['target'] - self.camera['pos']
            else:
                line_of_sight = centroid - self.camera['pos']
            ori = vector_rotation(
                line_of_sight, obj_forward, bdb3d_axis({'basis': basis}, 'up'),
                left_hand=True, range_pi=True
            )
        return ori

    def world2campix(self, world):
        """
        Transform 3D points or 3D bounding boxes from world coordinate to pixel coordinate.

        Parameters
        ----------
        world: n x 3 numpy array or bdb3d dict

        Returns
        -------
        for 3D points: n x 2 numpy array of xy in pixel.
        x: (left) 0 --> width - 1 (right)
        y: (up) 0 --> height - 1 (down)

        for 3D bounding boxes: dict{
            'centroid': centroid projected to camera plane
            'size': original bounding box size
            'dis': distance from centroid to camera in meters
            'depth': depth of centroid
            'ori': orientation of object in rad clockwise to line of sight (from above), [0, 2 * pi]
        }
        """
        if isinstance(world, dict):
            if isinstance(world['size'], torch.Tensor):
                backend = torch
                size = world['size'].clone()
            else:
                backend = np
                size = world['size'].copy()

            cam3d_centroid = self.world2cam3d(world['centroid'])
            campix = {
                'center': self.cam3d2pix(cam3d_centroid),
                'size': size,
                'dis': backend.linalg.norm(cam3d_centroid, axis=-1),
                'depth': cam3d_centroid[..., -1],
                'ori': self.basis2ori(world['basis'], world['centroid'])
            }
        else:
            campix = self.cam3d2pix(self.world2cam3d(world))
        return campix

    @staticmethod
    def homotrans(M, p):
        if isinstance(M, torch.Tensor):
            if p.shape[-1] == M.shape[1] - 1:
                p = torch.cat([p, torch.ones_like(p[..., :1], device=p.device)], -1)
            if p.dim() <= 2:
                p = p.unsqueeze(-2)
            p = torch.matmul(M, p.transpose(-1, -2)).transpose(-1, -2).squeeze(-2)
            return p[..., :-1] / p[..., -1:]
        else:
            return homotrans(M, p)

    @staticmethod
    def obj2frame(obj, bdb3d):
        """
        Transform 3D points or Trimesh from normalized object coordinate frame to coordinate frame bdb3d is in.
        object: x-left, y-back, z-up (defined by iGibson)
        world: right-hand coordinate of iGibson (z-up)

        Parameters
        ----------
        obj: n x 3 numpy array or Trimesh
        bdb3d: dict, self['objs'][id]['bdb3d']

        Returns
        -------
        n x 3 numpy array or Trimesh
        """
        if isinstance(obj, trimesh.Trimesh):
            obj = obj.copy()
            normalized_vertices = normalize_to_unit_square(obj.vertices, keep_ratio=False)[0]
            obj_vertices = normalized_vertices / 2
            obj.vertices = IGTransform.obj2frame(obj_vertices, bdb3d)
            return obj
        if isinstance(obj, torch.Tensor):
            size = bdb3d['size'].unsqueeze(-2)
            centroid = bdb3d['centroid'].unsqueeze(-2)
            return (bdb3d['basis'] @ (obj * size).transpose(-1, -2)).transpose(-1, -2) + centroid
        else:
            return (bdb3d['basis'] @ (obj * bdb3d['size']).T).T + bdb3d['centroid']

    @staticmethod
    def frame2obj(point, bdb3d):
        return (bdb3d['basis'].T @ (point - bdb3d['centroid']).T).T / bdb3d['size']

    def obj2cam3d(self, obj, bdb3d):
        return self.world2cam3d(self.obj2frame(obj, bdb3d))

    def cam3d2obj(self, cam3d, bdb3d):
        return self.frame2obj(self.cam3d2world(cam3d), bdb3d)

    def __getitem__(self, item):
        value = self.camera[item]
        if self.split == 'layout':
            return value
        if isinstance(value, torch.Tensor) and self.data \
                and self.split in self.data and 'split' in self.data[self.split].keys():
            expanded = []
            for t, s in zip(value, self.data[self.split]['split']):
                expanded.append(t.unsqueeze(0).expand([s[1] - s[0]] + list(t.shape)))
            return torch.cat(expanded)
        return value

    def in_cam(self, point, frame='world'):
        if frame == 'world':
            point = self.world2cam3d(point)
            depth = point[..., -1]
            point = self.cam3d2pix(point)
        elif frame == 'cam3d':
            depth = point[..., -1]
            point = self.cam3d2pix(point)
        elif frame != 'campix':
            raise NotImplementedError
        in_cam = np.all(
            (point <= np.array([self.camera['width'], self.camera['height']]) - 0.5)
            & (point >= -0.5), axis=-1
        )
        if frame in ('world', 'cam3d'):
            in_cam = (depth > 0) & in_cam
        return in_cam

    def rotate_layout_like_total3d(self, layout_bdb3d):
        # Rotate the forward vector of layout (by pi/2),
        # to make its dot product (with camera forward vector) to be maximal.

        layout_front = bdb3d_axis(layout_bdb3d)
        cam_front = cam_axis(self.camera)

        rot_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        rot_matrices = [np.linalg.matrix_power(rot_90, i) for i in range(4)]
        rotated_layout_fronts = [rot @ layout_front for rot in rot_matrices]
        dot_products = [f @ cam_front for f in rotated_layout_fronts]
        i_rot = np.argmax(dot_products)
        rot_matrix = rot_matrices[i_rot]

        layout = {
            'centroid': layout_bdb3d['centroid'].copy(),
            'size': np.abs(rot_matrix) @ layout_bdb3d['size'],
            'basis': rot_matrix @ layout_bdb3d['basis']
        }
        return layout
