import hashlib
import os
from glob import glob
import shutil
import numpy as np
import torch
from PIL import Image
import trimesh

from .image_utils import ImageIO
from .layout_utils import layout_line_segment_indexes
from .mesh_utils import MeshIO
from .transform_utils import IGTransform, cam_axis, bdb3d_from_corners, bdb3d_corners
from external.Equirec2Perspec.Equirec2Perspec import Equirectangular
from utils.basic_utils import read_pkl, write_pkl, write_json, recursively_to, \
    dict_of_array_to_list_of_dict, recursively_ignore
from utils.image_utils import load_image, save_image, show_image


def pickle_path(path):
    if not os.path.splitext(path)[1]:
        camera_folder = path
        pickle_file = os.path.join(path, 'data.pkl')
    elif path.endswith('pkl'):
        camera_folder = os.path.dirname(path)
        pickle_file = path
    else:
        raise Exception('Input path can be either folder or pkl file')
    os.makedirs(camera_folder, exist_ok=True)
    return camera_folder, pickle_file


def reverse_fov_split(data, n_splits, offset_bdb2d=False):
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k == 'objs':
                split = v['split']

                # recover bdb2d
                if offset_bdb2d:
                    width = data['camera']['width']
                    split_widths = width / n_splits
                    bdb2d_centers = (v['bdb2d']['x1'] + v['bdb2d']['x2']) / 2
                    bdb2d_widths = v['bdb2d']['x2'] - v['bdb2d']['x1']
                    for i_cam, (start, end) in enumerate(split):
                        offset = int(np.mod(i_cam - 0.5 + float(n_splits) / 2, n_splits) * split_widths[i_cam])
                        bdb2d_center = torch.remainder((bdb2d_centers[start:end]) + offset, width[i_cam])
                        bdb2d_hwidth = bdb2d_widths[start:end] / 2
                        v['bdb2d']['x1'][start:end] = bdb2d_center - bdb2d_hwidth
                        v['bdb2d']['x2'][start:end] = bdb2d_center + bdb2d_hwidth

                # recover batch split
                mask = torch.zeros_like(split , dtype=torch.bool)
                mask[range(0, len(split), n_splits), 0] = True
                mask[range(n_splits - 1, len(split), n_splits), 1] = True
                v['split'] = split[mask].reshape(-1, 2)

            elif k == 'walls':
                # recover batch split
                split = v['split'][range(0, len(split), n_splits)]
                interval = torch.cumsum(split[:, 1] - split[:, 0], 0)
                split = torch.stack([torch.cat(
                    [torch.zeros(1, dtype=interval.dtype, device=interval.device), interval[:-1]]), interval], -1)
                v['split'] = split

            if k not in ('objs', 'split', 'relation'):
                v = reverse_fov_split(v, n_splits)

            if k != 'relation':
                new_dict[k] = v

        return new_dict
    else:
        index = range(0, len(data), n_splits)
        if isinstance(data, list):
            return [data[i] for i in index]
        elif isinstance(data, torch.Tensor):
            return data[index]
        else:
            raise NotImplementedError(f"Not implemented for type {type(data)}")


def split_batch_into_patches(data: dict, keepdim=True, split=None):
    if isinstance(data, dict):
        new_data = []
        for k, v in data.items():
            if k == 'split':
                v = v - v[:, :1]
                patches = v.unsqueeze(1)
            else:
                if split is None and isinstance(v, dict):
                    s = v.get('split')
                else:
                    s = split
                patches = split_batch_into_patches(v, keepdim=keepdim, split=s)

            if new_data:
                for i in range(len(new_data)):
                    new_data[i][k] = patches[i]
            else:
                for patch in patches:
                    new_data.append({k: patch})
    else:
        if split is not None:
            new_data = [data[start:end] for start, end in split]
        elif keepdim:
            new_data = [data[i:i+1] for i in range(len(data))]
        else:
            new_data = data
    return new_data


class Camera(dict):
    _shared_params = {'width', 'height', 'K', 'vertical_fov'}

    def __init__(self, seq=None):
        if isinstance(seq, dict):
            self.cameras = [self]
            super(Camera, self).__init__(seq)
        elif isinstance(seq, list):
            self.cameras = seq
            shared_params = {k: v for k, v in seq[0].items() if k in self._shared_params}
            for c in seq[1:]:
                for k, v in shared_params.items():
                    assert np.all(c[k] == v)
            super(Camera, self).__init__(shared_params)

    def __getitem__(self, item):
        value = super(Camera, self).get(item)
        return value

    def __iter__(self):
        return self.cameras.__iter__()


class IGScene:
    '''
    A class used to store, process and visualize iGibson scene contents.
    '''

    ignore_when_saving = ['image_path', 'mesh_path', 'image_tensor', 'image_np']
    image_types = ['rgb', 'seg', 'sem', 'depth']
    mesh_file = 'mesh_watertight.ply'
    consider_when_transform = {
        'objs': {'bdb3d'},
        'layout': {'manhattan_world', 'cuboid_world'},
        'walls': {'bdb3d'}
    }
    basic_ignore_from_batch = {
        'objs': {'split', 'mesh_extractor', 'g_feature',
                 'lien_afeature', 'lien_activation', 'ben_afeature', 'ben_rfeature', 'ben_arfeature'},
        'walls': {'split'}
    }
    further_ignore_from_batch = {
        'objs': {'cls_code', 'rgb', 'seg'}
    }

    def __init__(self, data):
        assert isinstance(data, (dict, IGScene))
        if isinstance(data, IGScene):
            self.image_io = data.image_io.copy()
            self.mesh_io = data.mesh_io.copy()
            self.pkl_path = data.pkl_path
            self.data = data.data.copy()
            self.transform = data.transform.copy()
        else:
            self.image_io = ImageIO.from_file(data.get('image_path', {}))
            self.mesh_io = MeshIO.from_file(data.get('mesh_path', {}))
            self.pkl_path = None
            self.data = data
            if 'image_np' in data:
                self.image_io = ImageIO(data['image_np'])
            if 'objs' in data and data['objs'] and 'mesh' in data['objs'][0]:
                self.mesh_io = MeshIO()
                for i_obj, obj in enumerate(data['objs']):
                    self.mesh_io[i_obj] = obj.pop('mesh')
            self.transform = IGTransform(self.data)
        self.data['camera'] = Camera(self.data['camera'])

    @classmethod
    def from_pickle(cls, path: str, igibson_obj_dataset=None):
        camera_folder, pickle_file = pickle_path(path)
        data = read_pkl(pickle_file)
        image_path = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(camera_folder, '*.png'))}
        data['image_path'] = {k: v for k, v in image_path.items() if k in cls.image_types}
        if igibson_obj_dataset is not None:
            mesh_path = [os.path.join(igibson_obj_dataset, o['model_path'], cls.mesh_file)
                         for o in data['objs'] if 'model_path' in o]
            if mesh_path:
                data['mesh_path'] = mesh_path
        scene = cls(data)
        scene.pkl_path = pickle_file
        return scene

    @classmethod
    def from_image(cls, path):
        h, w = 512, 1024
        rgb = load_image(path)
        if any(a != b for a, b in zip(rgb.shape[:2], [h, w])):
            image = Image.fromarray(rgb)
            image = image.resize((w, h),Image.ANTIALIAS)
            path = os.path.splitext(path)[0] + '.png'
            save_image(np.array(image), path)
        name = os.path.splitext(os.path.basename(path))[0]
        data = {
            'name': name,
            'scene': '',
            'camera': {'height': h, 'width': w},
            'image_path': {'rgb': path}
        }
        scene = cls(data)
        scene.transform.set_camera_to_world_center()
        return scene

    @classmethod
    def from_batch(cls, data, gt_data=None, full=False):
        data = data.copy()
        if 'objs' in data.keys():
            # update ground truth ids of objs
            objs = data['objs'] = data['objs'].copy()
            if 'gt' in objs and len(data['name']) > 1:
                # assert gt_data is not None, "must specify gt_data for 'gt' mapping shift when data['objs'] has gt"
                if gt_data is None:
                    objs.pop('gt')
                else:
                    id_gt = objs['gt'] = objs['gt'].clone()
                    for (start, end), (start_gt, end_gt) in zip(objs['split'], gt_data['objs']['split']):
                        id_gt[start:end][id_gt[start:end] >= 0] -= start_gt
                        assert (id_gt[start:end] < (end_gt - start_gt)).all() and (id_gt[start:end] >= -1).all()

            # retrieve split for dict_of_array_to_list_of_dict
            split = {k: data[k]['split'] for k in ('objs', 'walls') if 'split' in data.get(k, {})}

            # ignore unwanted intermedia results in data
            data = recursively_ignore(data, cls.basic_ignore_from_batch)
            if not full:
                data = recursively_ignore(data, cls.further_ignore_from_batch)
        else:
            split = None

        scene_datas = dict_of_array_to_list_of_dict(data, split=split)
        scenes = [cls(scene_data) for scene_data in scene_datas]

        return scenes

    def empty(self):
        return all(k in ('name', 'scene', 'camera', 'image_path') for k in self.data.keys())

    def fov_split(self, fov, gt_offset=0, offset_bdb2d=False):
        fov = np.deg2rad(fov)
        assert np.isclose(np.mod(np.pi * 2, fov), 0)

        # split cameras
        yaws = np.arange(0, np.pi * 2, fov).astype(np.float32)
        n_split = len(yaws)
        split_width = self['camera']['width'] / n_split
        targets_rad = np.stack([yaws, np.zeros_like(yaws)], -1)
        targets = self.transform.camrad2world(targets_rad, 1)
        trans_cams = []
        for target in targets:
            trans = self.transform.copy()
            trans.look_at(target)
            trans_cams.append(trans)

        # split objects by cameras
        obj_splits = [[] for _ in range(len(trans_cams))]
        for obj in self['objs']:
            obj = obj.copy()

            # offset ground truth mapping
            if 'gt' in obj:
                obj['gt'] = obj['gt'] + gt_offset

            # find out which camera the object is in
            bfov_center_rad = np.array([obj['bfov']['lon'], obj['bfov']['lat']])
            bfov_center_world = self.transform.camrad2world(bfov_center_rad, 1)
            for i_scene, trans in enumerate(trans_cams):
                # transform object center from world frame to camera frame
                cam_center_rad = trans.world2camrad(bfov_center_world)
                if -fov / 2 < cam_center_rad[0] <= fov / 2:
                    # offset bdb2d to target camera
                    if offset_bdb2d:
                        x1, x2 = obj['bdb2d']['x1'], obj['bdb2d']['x2']
                        bdb2d_width = x2 - x1
                        bdb2d_center = (x1 + x2) / 2
                        bdb2d_center_in_cam = np.mod(
                            bdb2d_center - (0 if np.mod(n_split, 2) else split_width / 2),
                            split_width
                        )
                        x1 = int(bdb2d_center_in_cam - bdb2d_width / 2)
                        x2 = int(bdb2d_center_in_cam + bdb2d_width / 2)
                        obj['bdb2d']['x1'], obj['bdb2d']['x2'] = x1, x2
                    obj_splits[i_scene].append(obj)
                    break

        # create scenes with new objects and cameras
        scenes = []
        for trans, objs in zip(trans_cams, obj_splits):
            data = self.data.copy()
            data['camera'] = trans.camera
            data['objs'] = objs
            scene = IGScene(data)
            scenes.append(scene)

        return scenes

    def set_camera_to_world_center(self):
        # transform to camera centered and orientated world frame
        def apply_on_specified(dic, keys, func):
            if isinstance(dic, list):
                for i in dic:
                    apply_on_specified(i, keys, func)
            elif isinstance(keys, dict):
                for k, v in keys.items():
                    if v is True and k in dic:
                        dic[k] = func(dic[k])
                    elif k in dic:
                        apply_on_specified(dic[k], v, func)
            else:
                for k in keys:
                    if k in dic:
                        if isinstance(dic[k], dict):
                            dic[k].update(func(dic[k]))
                        else:
                            dic[k] = func(dic[k])
        apply_on_specified(self.data, self.consider_when_transform, self.transform.world2cam3d)
        self.transform.set_camera_to_world_center()
        apply_on_specified(self.data, self.consider_when_transform, self.transform.cam3d2world)

    def data_save(self):
        return {k: v for k, v in self.data.items() if k not in self.ignore_when_saving}

    def to_pickle(self, path=None):
        if path is None:
            path = self.pkl_path
        camera_folder, pickle_file = pickle_path(path)
        write_pkl(self.data_save(), pickle_file)

    def to_json(self, path=None):
        camera_folder, pickle_file = pickle_path(path)
        write_json(recursively_to(self.data_save(), 'list'), os.path.splitext(pickle_file)[0] + '.json')

    def remove(self):
        shutil.rmtree(os.path.dirname(self.pkl_path))

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def crop_images(self, perspective=True, short_width=280, crop_types=('rgb',), include=('objs', )):
        for key in include:
            for crop_type in crop_types:
                for obj in self.data[key]:
                    image = self.image_io[crop_type]
                    if crop_type == 'seg':
                        image = (image == obj['id']).astype(np.uint8) * 255

                    if perspective:
                        bfov = obj['bfov']
                        if bfov['x_fov'] > bfov['y_fov']:
                            height = short_width
                            width = bfov['x_fov'] / bfov['y_fov'] * height
                        else:
                            width = short_width
                            height = bfov['y_fov'] / bfov['x_fov'] * width
                        crop = Equirectangular(image).GetPerspective(
                            np.rad2deg(bfov['x_fov']),
                            np.rad2deg(bfov['lon']), -np.rad2deg(bfov['lat']),
                            round(height), round(width)
                        )
                    else:
                        bdb2d = obj['bdb2d']
                        crop = image[bdb2d['y1']:bdb2d['y2'] + 1, bdb2d['x1']:bdb2d['x2'] + 1]
                    obj[crop_type] = crop
                    # show_image(crop)

    def to_horizon(self, path):
        output_name =  f"{self.data['scene']}_{self.data['name']}"

        # save layout as LayoutNet/HorizonNet format
        layout_folder = os.path.join(path, 'label_cor')
        os.makedirs(layout_folder, exist_ok=True)
        layout_txt = os.path.join(layout_folder, output_name + '.txt')
        np.savetxt(layout_txt, self.data['layout']['manhattan_pix'], '%d')

        # link rgb to horizonnet folder
        image_folder = os.path.join(path, 'img')
        os.makedirs(image_folder, exist_ok=True)
        dst = os.path.join(image_folder, output_name + '.png')
        if os.path.exists(dst):
            os.remove(dst)
        os.link(self.image_io['image_path']['rgb'], dst)

    def merge_mesh(self, colorbox=None, separate=False, camera_color=None, layout_color=None, texture=True):
        self.mesh_io.load()
        mesh_io = MeshIO()
        if not self.data['objs']:
            return mesh_io

        # transform each object mesh to world frame
        objs = self.data['objs']
        for k, v in self.mesh_io.items():
            bdb3d = objs[k]['bdb3d']
            mesh_world = self.transform.obj2frame(v, bdb3d)
            if colorbox is not None:
                mesh_world = IGScene.colorize_mesh_for_igibson(mesh_world, colorbox[objs[k]['label']], texture)
            mesh_io[k] = mesh_world

        # add camera marker
        if camera_color is not None:
            camera_mesh = self.camera_marker(color=camera_color, texture=texture)
            mesh_io['camera'] = camera_mesh

        # add layout mesh
        if layout_color is not None:
            layout_mesh = self.layout_mesh(color=layout_color, texture=texture)
            if layout_mesh is not None:
                mesh_io['layout_mesh'] = layout_mesh

        if separate:
            return mesh_io

        return mesh_io.merge()

    def layout_mesh(self, color=(255, 69, 80), radius=0.025, texture=True):
        if 'layout' not in self.data or (
                'manhattan_world' not in self.data['layout']
                and 'total3d' not in self.data['layout']
        ):
            return None
        if 'total3d' in self.data['layout']:
            mesh = self.bdb3d_mesh(self.data['layout']['total3d'], color=color)
        elif 'manhattan_world' in self.data['layout']:
            mesh = []
            layout_points = self.data['layout']['manhattan_world']
            layout_lines = layout_line_segment_indexes(len(layout_points) // 2)
            for indexes in layout_lines:
                line = layout_points[indexes]
                line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
                mesh.append(line_mesh)
            mesh = sum(mesh)
            mesh = IGScene.colorize_mesh_for_igibson(mesh, color, texture)
        return mesh

    def bdb3d_mesh(self, bdb3d, color, radius=0.05):
        corners = bdb3d_corners(bdb3d)
        corners_box = corners.reshape(2, 2, 2, 3)
        mesh = []
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    line = corners_box[idx1], corners_box[idx2]
                    line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
                    mesh.append(line_mesh)
        for idx1, idx2 in [(0, 5), (1, 4)]:
            line = corners[idx1], corners[idx2]
            line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
            mesh.append(line_mesh)
        mesh = sum(mesh)
        mesh = IGScene.colorize_mesh_for_igibson(mesh, color)
        return mesh

    def camera_marker(self, color=(29, 203, 224), length=0.5, radius=0.05, texture=True):
        mesh = []
        vertical_fov = self['camera']['vertical_fov']
        width, height = self['camera']['width'], self['camera']['height']
        horizontal_fov = vertical_fov * width / height

        forward = cam_axis() * length
        right = cam_axis(axis='right') * np.tan(np.deg2rad(horizontal_fov) / 2) * length
        down = cam_axis(axis='down') * np.tan(np.deg2rad(vertical_fov) / 2) * length
        axes = np.stack([forward, right, down]).T

        # draw 3D pyramid
        lines = np.array([
            [[0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 1, -1]],
            [[0, 0, 0], [1, -1, 1]],
            [[0, 0, 0], [1, -1, -1]],
            [[1, 1, 1], [1, 1, -1]],
            [[1, 1, -1], [1, -1, -1]],
            [[1, -1, -1], [1, -1, 1]],
            [[1, -1, 1], [1, 1, 1]]
        ])
        lines = lines[:, :, None, :]
        for line in lines:
            line = line * axes
            line = np.sum(line, axis=-1)
            line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
            mesh.append(line_mesh)
        mesh = sum(mesh)
        mesh_worlds = []
        for camera in self['camera']:
            trans = IGTransform({'camera': camera})
            mesh_world = trans.cam3d2world(mesh)
            mesh_worlds.append(mesh_world)
        mesh_world = sum(mesh_worlds)
        mesh_world = IGScene.colorize_mesh_for_igibson(mesh_world, color, texture)
        return mesh_world

    @staticmethod
    def colorize_mesh_for_igibson(mesh, color, texture=True):
        mesh.visual.vertex_colors[:] = np.append(color, 255).astype(np.uint8)
        if texture:
            mesh.vertex_normals
            mesh.visual = mesh.visual.to_texture()
            # for unknown reason, iGibson renderer cannot correctly
            # render .obj meshes having points with same uv value
            mesh.visual.uv = np.random.rand(*mesh.visual.uv.shape)
        return mesh


def hash_split(train_ratio, key):
    object_hash = np.frombuffer(hashlib.md5(key.encode('utf-8')).digest(), np.uint32)
    rng = np.random.RandomState(object_hash)
    is_train = rng.random() < train_ratio
    return is_train
