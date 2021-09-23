import shutil
import os
import tempfile
import cv2
import seaborn as sns
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np
from glob import glob
from copy import deepcopy

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from configs.data_config import igibson_colorbox
from external.HorizonNet.misc.post_proc import np_coor2xy, np_coory2v
from external.panoramic_object_detection.detect import wrapped_line
from .igibson_utils import IGScene
from .layout_utils import layout_line_segment_indexes
from .transform_utils import bdb3d_corners, bdb3d_axis, interpolate_line
from .render_utils import render_camera, hdr_texture, hdr_texture2
from .image_utils import save_image


def detectron_gt_sample(data, idx=None):
    record = {
        "file_name": data['image_path']['rgb'],
        "image_id": idx,
        "height": data['camera']['height'],
        "width": data['camera']['width']
    }
    annotations = []
    for obj in data['objs']:
        bdb2d, contour = obj['bdb2d'], obj['contour']
        poly = [(x + 0.5, y + 0.5) for x, y in zip(contour['x'], contour['y'])]
        poly = [p for x in poly for p in x]
        obj = {
            "bbox": [bdb2d['x1'], bdb2d['y1'], bdb2d['x2'], bdb2d['y2']],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": obj['label'],
        }
        annotations.append(obj)
    record["annotations"] = annotations
    return record


def visualize_igibson_detectron_gt(sample, image=None, dataset=None):
    if dataset is None:
        dataset = list(MetadataCatalog.keys())[-1]
    if image is None:
        image = np.array(Image.open(sample["file_name"]))
    visualizer = Visualizer(image, metadata=MetadataCatalog.get(dataset))
    image = visualizer.draw_dataset_dict(sample).get_image()
    return image


def visualize_igibson_detectron_pred(prediction, image, dataset=None):
    if dataset is None:
        dataset = list(MetadataCatalog.keys())[-1]
    visualizer = Visualizer(image, metadata=MetadataCatalog.get(dataset))
    image = visualizer.draw_instance_predictions(prediction['instances'].to('cpu')).get_image()
    return image


def visualize_image(image, key=None):
    if isinstance(image, dict):
        visual = {}
        for k, v in image.items():
            visual[k] = visualize_image(v, k)
        return visual
    if key in ['sem', 'seg']:
        if key == 'sem':
            color_box = igibson_colorbox
        else:
            color_box = np.array(sns.hls_palette(n_colors=image.max() + 1, l=.45, s=1.))
            np.random.shuffle(color_box)
        color_map = np.zeros(list(image.shape) + [3], dtype=np.uint8)
        for i_color, color in enumerate(color_box):
            color_map[image == i_color] = color * 255
        image = color_map
    if key == 'depth':
        image = image.astype(np.float)
        image = (image / image.max() * 255).astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class IGVisualizer:
    def __init__(self, scene: IGScene, gpu_id=0, debug=False):
        self.scene = scene
        self.image_io = scene.image_io
        self.mesh_io = scene.mesh_io
        self.transform = scene.transform
        self.gpu_id = gpu_id
        self._temp_folder = None
        self._renderer = None
        self.debug = debug

    def render(self, background=None, camera=None, camera_color=(29, 203, 224), layout_color=(255, 69, 80)):
        first_person = camera is None
        scene = self.scene

        # merge meshes
        scene_mesh = scene.merge_mesh(
            colorbox=igibson_colorbox * 255,
            separate=True,
            camera_color=camera_color if not first_person else None,
            layout_color=layout_color
        )

        # save temp meshes
        if self._temp_folder is None:
            self._temp_folder = 'out/tmp' if self.debug else tempfile.mktemp(dir='/dev/shm')
            os.makedirs(self._temp_folder, exist_ok=True)
        scene_mesh.save(self._temp_folder)

        # save temp background image
        if not isinstance(background, np.ndarray):
            if background is None:
                background = 255
            background = self.background(background, force_pano_height=1024)
        background_path = os.path.join(self._temp_folder, 'background.png')
        save_image(background, background_path)

        # initialize renderer
        obj_paths = glob(os.path.join(self._temp_folder, '*', '*.obj'))
        settings = MeshRendererSettings(
            env_texture_filename=os.path.join('images', 'photo_studio_01_2k.hdr'),
            env_texture_filename2=os.path.join('images', 'photo_studio_01_2k.hdr'),
            env_texture_filename3=background_path,
            msaa=True, enable_shadow=False, enable_pbr=True
        )
        height, width = self.scene['camera']['height'], self.scene['camera']['width']
        self._renderer = MeshRenderer(
            width=width if 'K' in self.scene['camera'] else height,
            height=height,
            rendering_settings=settings,
            device_idx=self.gpu_id
        )
        self._renderer.set_light_position_direction([10, 10, 10], [0, 0, 0])

        if first_person:
            # set camera to the first-person view
            camera = scene['camera']
        elif not isinstance(camera, dict):
            # set camera view automatically
            new_camera = {'width': width, 'height': height}
            scene_mesh = scene.merge_mesh(separate=False)
            target = np.mean(scene_mesh.vertices, 0)
            new_camera['target'] = target
            radius = np.linalg.norm(scene_mesh.vertices - target, axis=1).max()

            dis = 10
            fov = np.rad2deg(np.arctan2(radius, dis) * 2)
            new_camera['vertical_fov'] = fov * new_camera['height'] / new_camera['height']

            if camera == 'birds_eye':
                yaw = np.pi / 4
                camera_height = dis * np.sin(np.pi / 4)
                pos = np.array([
                    np.sin(yaw) * camera_height,
                    np.cos(yaw) * camera_height,
                    camera_height + target[-1]
                ], np.float32)
                up = np.array([0, 0, 1], np.float32)
            elif camera == 'up_down':
                pos = target.copy()
                pos[-1] += dis
                up = np.array([1, 0, 0], np.float32)
            else:
                raise NotImplementedError
            new_camera['pos'] = pos
            new_camera['up'] = up
            new_camera['K'] = None
            camera = new_camera

        # load temp meshes and render
        for i_obj, obj_path in enumerate(obj_paths):
            self._renderer.load_object(obj_path)
            self._renderer.add_instance(i_obj, class_id=i_obj)
        render = render_camera(self._renderer, camera, 'rgb')['rgb']
        self._renderer.release()
        self._renderer = None

        # clean out temp files
        if not self.debug:
            for obj_path in obj_paths:
                shutil.rmtree(os.path.dirname(obj_path))
        return render

    def background(self, color=200, channels=3, force_pano_height=None):
        camera = self.scene['camera']
        height, width = camera['height'], camera['width']
        background = np.ones(
            [
                height if force_pano_height is None else force_pano_height,
                width if force_pano_height is None else force_pano_height * 2,
                channels
            ],
            dtype=np.uint8
        ) * color
        return background

    def image(self, key='rgb'):
        image = self.image_io[key]
        image = visualize_image(image, key)
        return image

    def bdb2d(self, image, dataset=None):
        if 'objs' not in self.scene.data or not self.scene['objs'] or 'bdb2d' not in self.scene['objs'][0]:
            return image
        sample = detectron_gt_sample(self.scene)
        image = visualize_igibson_detectron_gt(sample, image, dataset)
        return image

    def bfov(self, image, thickness=2, include=('objs', )):
        for key in include:
            if key not in self.scene.data or not self.scene[key] or 'bfov' not in self.scene[key][0]:
                continue
            image = image.copy()
            objs = self.scene[key]
            for obj in objs:
                color = (igibson_colorbox[obj['label']] * 255).astype(np.uint8).tolist() \
                    if 'label' in obj else (255, 255, 0)
                bfov = obj['bfov']
                self._bfov(image, bfov, color, thickness)
        return image

    def layout(self, image, color=(255, 255, 0), thickness=2, force_pano=False, total3d=False):
        if total3d and 'K' in self.scene['camera']:
            if 'layout' not in self.scene.data or 'total3d' not in self.scene['layout']:
                return image
            image = image.copy()
            layout_bdb3d = self.scene['layout']['total3d']
            self._bdb3d(image, layout_bdb3d, color)
            return image
        else:
            H, W = image.shape[:2]

            if 'K' in self.scene['camera']:
                if 'layout' not in self.scene.data or 'manhattan_world' not in self.scene['layout']:
                    return image
                image = image.copy()
                if force_pano:
                    data = self.scene.data.copy()
                    data['camera'] = data['camera'].copy()
                    data['camera'].pop('K')
                    data['camera']['height'], data['camera']['width'] = H, W
                    scene = IGScene(data)
                    visualizer = IGVisualizer(scene)
                else:
                    visualizer = self
                layout_points = self.scene['layout']['manhattan_world']
                N = len(layout_points) // 2
                frame = 'world'
            else:
                if 'layout' not in self.scene.data or 'manhattan_pix' not in self.scene['layout']:
                    return image
                cor_id = np.array(self.scene['layout']['manhattan_pix'], np.float32)
                image = image.copy()

                N = len(cor_id) // 2
                floor_z = -1.6
                floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
                c = np.sqrt((floor_xy ** 2).sum(1))
                v = np_coory2v(cor_id[0::2, 1], H)
                ceil_z = (c * np.tan(v)).mean()

                assert N == len(floor_xy)
                layout_points = [[x, -floor_z, -y] for x, y in floor_xy] + \
                            [[x, -ceil_z, -y] for x, y in floor_xy]
                frame = 'cam3d'
                visualizer = self

            layout_lines = layout_line_segment_indexes(N)
            layout_points = np.array(layout_points)
            for point1, point2 in layout_lines:
                point1 = layout_points[point1]
                point2 = layout_points[point2]
                visualizer._line3d(image, point1, point2, color, thickness, frame=frame)

        return image

    def objs3d(self, image, bbox3d=True, axes=False, centroid=False, info=False, thickness=2):
        if 'objs' not in self.scene.data or not self.scene['objs'] or 'bdb3d' not in self.scene['objs'][0]:
            return image
        image = image.copy()
        objs = self.scene['objs']
        dis = [np.linalg.norm(self.transform.world2cam3d(o['bdb3d']['centroid'])) for o in objs]
        i_objs = sorted(range(len(dis)), key=lambda k: dis[k])
        for i_obj in reversed(i_objs):
            obj = objs[i_obj]
            color = (igibson_colorbox[obj['label']] * 255).astype(np.uint8).tolist()
            bdb3d = obj['bdb3d']
            # # test bdb3d transformation
            # bdb3d = self.transform.world2campix(bdb3d)
            # bdb3d = self.transform.campix23d(bdb3d)
            # bdb3d = self.transform.cam3d2pix(bdb3d)
            # bdb3d = self.transform.campix2world(bdb3d)
            if axes:
                self._objaxes(image, bdb3d, thickness=thickness)
            if centroid:
                self._centroid(image, bdb3d['centroid'], color, thickness=thickness)
            if bbox3d:
                self._bdb3d(image, bdb3d, color, thickness=thickness)
            if info:
                self._objinfo(image, bdb3d, color)
        return image

    def wall3d(self, image, color=100, thickness=2):
        for wall in self.scene['walls']:
            image = image.copy()
            self._bdb3d(image, wall['bdb3d'], color, thickness=thickness)
        return image

    def relation(self, image, obj_obj_tch=True, floor_ceil=True, obj_wall_tch=True, collision=False, thickness=2):
        image = image.copy()
        objs = self.scene['objs']
        walls = self.scene['walls']
        relation_color = 255
        collision_color = (255, 74, 40)
        relation_thickness = thickness
        collision_thickness = thickness

        # visualize touch relationship between objects
        if obj_obj_tch or collision:
            for i_a, obj_a in enumerate(objs):
                for i_b, obj_b in enumerate(objs):
                    if i_a == i_b or any(not obj.get('in_room', True) for obj in (obj_a, obj_b)):
                        continue
                    bdb3d_a = obj_a['bdb3d']
                    bdb3d_b = obj_b['bdb3d']
                    src = bdb3d_a['centroid']
                    dst = bdb3d_b['centroid']
                    if obj_obj_tch and self.scene['relation']['obj_obj_tch'][i_a, i_b]:
                        self._line3d(image, src, dst, relation_color, relation_thickness, frame='world')
                    if collision and self.scene['relation']['obj_obj_col'][i_a, i_b]:
                        self._line3d(image, src, dst, collision_color, collision_thickness, frame='world')

        # visualize floor/ceiling relationships of objects
        if floor_ceil or collision:
            layout_z = self.scene['layout']['manhattan_world'][:, -1]
            floor = layout_z.min()
            ceil = layout_z.max()
            for obj in objs:
                src = obj['bdb3d']['centroid']
                dst = src.copy()

                dst[-1] = floor
                if floor_ceil and obj['floor_tch']:
                    self._line3d(image, src, dst, relation_color, relation_thickness, frame='world')
                if collision and obj['floor_col']:
                    self._line3d(image, src, dst, collision_color, collision_thickness, frame='world')

                dst[-1] = ceil
                if floor_ceil and obj['ceil_tch']:
                    self._line3d(image, src, dst, relation_color, relation_thickness, frame='world')
                if collision and obj['ceil_col']:
                    self._line3d(image, src, dst, collision_color, collision_thickness, frame='world')

        # visualize touch relationship between objects and walls
        if obj_wall_tch or collision:
            for i_obj, obj in enumerate(objs):
                for i_wall, wall in enumerate(walls):
                    bdb3d_obj = obj['bdb3d']
                    bdb3d_wall = wall['bdb3d']
                    label_obj_wall = self.scene['relation']['obj_wall_tch'][i_obj, i_wall]

                    src = bdb3d_obj['centroid']
                    wall_forward = bdb3d_axis(bdb3d_wall)
                    wall_hdepth = bdb3d_wall['size'][1]
                    obj_wall_dis = np.dot(src - bdb3d_wall['centroid'], wall_forward) - wall_hdepth / 2
                    dst = src - wall_forward * obj_wall_dis

                    if obj_wall_tch and label_obj_wall:
                        self._line3d(image, src, dst, relation_color, relation_thickness, frame='world')
                    if collision and self.scene['relation']['obj_wall_col'][i_obj, i_wall]:
                        self._line3d(image, src, dst, collision_color, collision_thickness, frame='world')

        return image

    def _bfov(self, image, bfov, color, thickness=2):
        target = self.transform.camrad2world(np.stack([bfov['lon'], bfov['lat']]), 1)
        pers_trans = self.transform.copy()
        pers_trans.look_at(target)

        # coordinate of right down corner in perspective camera frame
        half_x_fov = bfov['x_fov'] / 2
        half_y_fov = bfov['y_fov'] / 2
        half_height = np.tan(half_y_fov)
        dis_right = 1 / np.cos(half_x_fov)
        right_down_y = half_height / dis_right
        right_down_x = np.sin(half_x_fov)
        right_down_z = np.cos(half_x_fov)

        corners = np.array([
            [right_down_x, right_down_y, right_down_z],
            [-right_down_x, right_down_y, right_down_z],
            [-right_down_x, -right_down_y, right_down_z],
            [right_down_x, -right_down_y, right_down_z]
        ])

        corners = pers_trans.cam3d2world(corners)
        for start, end in zip(corners, np.roll(corners, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, frame='world')

    def _objaxes(self, image, bdb3d, thickness=2):
        origin = np.zeros(3, dtype=np.float32)
        centroid = self.transform.obj2frame(origin, bdb3d)
        for axis in np.eye(3, dtype=np.float32):
            endpoint = self.transform.obj2frame(axis / 2, bdb3d)
            color = axis * 255
            self._line3d(image, centroid, endpoint, color, thickness, frame='world')

    def _centroid(self, image, centroid, color, thickness=2):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        center = self.transform.world2campix(centroid)
        cv2.circle(image, tuple(center.astype(np.int32).tolist()), 5, color, thickness=thickness, lineType=cv2.LINE_AA)

    def _bdb3d(self, image, bdb3d, color, thickness=2):
        corners = self.transform.world2cam3d(bdb3d_corners(bdb3d))
        corners_box = corners.reshape(2, 2, 2, 3)
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    self._line3d(image, corners_box[idx1], corners_box[idx2], color, thickness=thickness, frame='cam3d')
        for idx1, idx2 in [(0, 5), (1, 4)]:
            self._line3d(image, corners[idx1], corners[idx2], color, thickness=thickness, frame='cam3d')

    def _contour(self, image, contour, color, thickness=1):
        contour_pix = np.stack([contour['x'], contour['y']], -1)
        contour_3d = self.transform.campix23d(contour_pix, 1)
        for start, end in zip(contour_3d, np.roll(contour_3d, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, quality=2, frame='cam3d')

    def _bdb2d(self, image, bdb2d, color, thickness=1):
        x1, x2, y1, y2 = bdb2d['x1'], bdb2d['x2'], bdb2d['y1'], bdb2d['y2']
        corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        corners_pix = np.array(corners)
        corners_3d = self.transform.campix23d(corners_pix, 1)
        for start, end in zip(corners_3d, np.roll(corners_3d, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, frame='cam3d')

    def _objinfo(self, image, bdb3d, color):
        color = [255 - c for c in color]
        bdb3d_pix = self.transform.world2campix(bdb3d)
        bdb3d_info = [f"center: {bdb3d_pix['center'][0]:.0f}, {bdb3d_pix['center'][1]:.0f}",
                      f"dis: {bdb3d_pix['dis']:.1f}",
                      f"ori: {np.rad2deg(bdb3d_pix['ori']):.0f}"]
        bottom_left = bdb3d_pix['center'].copy().astype(np.int32)
        for info in reversed(bdb3d_info):
            bottom_left[1] -= 16
            cv2.putText(image, info, tuple(bottom_left.tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _line3d(self, image, p1, p2, color, thickness, quality=30, frame='world'):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        if frame == 'world':
            p1 = self.transform.world2cam3d(p1)
            p2 = self.transform.world2cam3d(p2)
        elif frame != 'cam3d':
            raise NotImplementedError
        points = interpolate_line(p1, p2, quality)
        pix = np.round(self.transform.cam3d2pix(points)).astype(np.int32)
        for t in range(quality - 1):
            p1, p2 = pix[t], pix[t + 1]
            if 'K' in self.scene['camera']:
                if self.transform.in_cam(points[t], frame='cam3d') \
                        or self.transform.in_cam(points[t + 1], frame='cam3d'):
                    cv2.line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)
            else:
                wrapped_line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)

    def __del__(self):
        if not self.debug:
            if self._temp_folder is not None:
                shutil.rmtree(self._temp_folder)
        if self._renderer is not None:
            self._renderer.release()
