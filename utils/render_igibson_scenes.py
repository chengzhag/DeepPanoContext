import os
import gibson2
import argparse
import numpy as np
import pybullet as p
from multiprocessing import Pool
from tqdm import tqdm
import shutil
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, Point, MultiPoint
import shapely
from glob import glob
import traceback

from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.utils.assets_utils import get_ig_scene_path, get_cubicasa_scene_path, get_3dfront_scene_path
from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz

from configs.data_config import IG56CLASSES
from utils.relation_utils import RelationOptimization
from utils.render_utils import seg2obj, render_camera, is_obj_valid, hdr_texture, hdr_texture2, background_texture
from .igibson_utils import IGScene, hash_split
from utils.image_utils import ImageIO
from .layout_utils import scene_layout_from_mesh, room_layout_from_scene_layout, \
    manhattan_pix_layout_from_room_layout, cuboid_world_layout_from_room_layout, \
    manhattan_world_layout_from_room_layout, horizon_layout_gt_from_scene_data
from .transform_utils import bdb3d_corners, IGTransform
from utils.basic_utils import write_json, read_pkl, write_pkl


def _render_scene_fail_remove(args):
    output_folder = os.path.join(args.output, args.scene_name)
    try:
        camera_paths = _render_scene(args)
    except Exception as err:
        camera_paths = []
        traceback.print_exc()
        if args.strict:
            raise err
    if not camera_paths:
        tqdm.write(f"Failed to generate {args.scene_name}")
        if os.path.exists(output_folder) \
                and len(glob(os.path.join(output_folder, '*/'))) <= len(camera_paths):
            shutil.rmtree(output_folder)
    return camera_paths


def _render_scene(args):
    # preparation
    scene_name, scene_source = args.scene_name, args.scene_source
    output_folder = os.path.join(args.output, scene_name)
    if scene_source == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_source == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)
    light_modulation_map_filename = os.path.join(
        scene_dir, 'layout', 'floor_lighttype_0.png')
    scene = InteractiveIndoorScene(
        scene_name,
        texture_randomization=False,
        object_randomization=args.random_obj is not None,
        object_randomization_idx=args.random_obj,
        scene_source=scene_source
    )
    settings = MeshRendererSettings(
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True, msaa=False, enable_pbr=True
    )

    # generate scene layout
    scene_layout = scene_layout_from_mesh(args)
    if not scene_layout:
        raise Exception('Layout not valid!')

    # set camera parameters
    perspective = args.vertical_fov is not None
    if perspective:
        vertical_fov = args.vertical_fov
        render_width = args.width
        output_width = args.width
    else:
        vertical_fov = 90
        render_width = args.height
        output_width = args.height * 2
    output_height = args.height
    render_height = args.height * args.super_sample
    render_width *= args.super_sample
    s = Simulator(mode='headless', image_width=render_width, image_height=render_height,
                  vertical_fov=vertical_fov, device_idx=args.gpu_id, rendering_settings=settings)

    # convert floor_trav pickle file to protocol-4 to avoid error
    floor_trav_path = os.path.join(scene_dir, 'layout', 'floor_trav_0.p')
    if os.path.exists(floor_trav_path):
        try:
            read_pkl(floor_trav_path)
        except ValueError:
            print(f"floor_trav pickle file {floor_trav_path} is not compatible, converting...")
            floor_trav = read_pkl(floor_trav_path, protocol=5)
            write_pkl(floor_trav, floor_trav_path)

    # import scene and run physical simulation
    try:
        s.import_ig_scene(scene)
    except Exception as err:
        s.disconnect()
        raise err
    if not args.no_physim:
        for i in range(200):
            s.step()

    # get scene object info
    is_fixed = {} # if the link type is fixed or floating
    urdf_files = {} # temp URDF files of each sub object
    obj_ids = list(scene.objects_by_id.keys())
    i_obj = 0
    obj_groups = {} # main object and the sub objects of object groups
    while i_obj < len(obj_ids):
        obj_id = obj_ids[i_obj]
        urdf_object = scene.objects_by_id[obj_id]
        for i_subobj, (fixed, urdf_file) in enumerate(zip(urdf_object.is_fixed, urdf_object.urdf_paths)):
            is_fixed[obj_id + i_subobj] = fixed
            urdf_files[obj_id + i_subobj] = urdf_file
        obj_group = urdf_object.body_ids.copy()
        if len(obj_group) > 1:
            # treat the object with the greatest mass as main object
            mass_list = []
            for i_subobj in obj_group:
                obj_tree = ET.parse(urdf_files[i_subobj])
                mass_list.append(float(obj_tree.find("link").find("inertial").find('mass').attrib['value']))
            main_object = obj_group[np.argmax(mass_list)]
            obj_group.remove(main_object)
            obj_groups[main_object] = obj_group
        i_obj += len(urdf_object.body_ids)

    # get object params
    objs = {}
    for obj_id in range(len(scene.objects_by_id)):
        # get object info
        obj = scene.objects_by_id[obj_id]
        if getattr(obj, 'bounding_box', None) is None:
            continue
        obj_dict = {
            'classname': obj.category,
            'label': IG56CLASSES.index(obj.category),
            'model_path': os.path.join(*obj.model_path.split('/')[-2:]),
            'is_fixed': is_fixed[obj_id],
        }

        # get object bdb3d
        if is_fixed[obj_id]:
            orn = p.getLinkState(obj_id, 0)[-1]
            aabb = p.getAABB(obj_id, 0)
        else:
            _, orn = p.getBasePositionAndOrientation(obj_id)
            aabb = p.getAABB(obj_id, -1)

        # use axis aligned bounding box center of first link as bounding box center
        centroid = np.mean(aabb, axis=0)
        basis = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
        obj_dict['bdb3d'] = {
            'centroid': centroid.astype(np.float32),
            'basis': basis.astype(np.float32),
            'size': obj.bounding_box.astype(np.float32)
        }
        objs[obj_id] = obj_dict

    # get object layout
    object_layout = []
    for obj in objs.values():
        corners = bdb3d_corners(obj['bdb3d'])
        corners2d = corners[(0, 1, 3, 2), :2]
        obj2d = Polygon(corners2d)
        object_layout.append(obj2d)
    object_layout = shapely.ops.cascaded_union(object_layout)
    # plot_layout(object_layout)

    # render random camera, get and save GT
    np.random.seed(args.seed)
    camera_paths = []
    i_camera = 0
    while i_camera < args.renders:
        # randomize camera position
        _, (px, py, pz) = scene.get_random_point()
        if len(args.cam_height) == 1:
            pz = args.cam_height[0]
        else:
            pz = np.random.random() * (args.cam_height[1] - args.cam_height[0]) + args.cam_height[0]
        camera_pos = np.array([px, py, pz], dtype=np.float32)

        # generate room layout by camera position
        camera_name = i_camera + args.random_obj * args.renders if args.random_obj is not None else i_camera
        camera = {
            'pos': camera_pos,
            'height': output_height,
            'width': output_width
        }
        data = {
            'name': f"{camera_name:05d}",
            'scene': scene_name,
            'room': scene.get_room_instance_by_point(camera_pos[:2]),
            'camera': camera
        }
        skip_info = f"Skipped camera {data['name']} of {data['scene']}: "
        if data['room'] is None:
            print(skip_info + "room is 'None'")
            continue
        room_layout = room_layout_from_scene_layout(camera, scene_layout)
        if room_layout is None:
            print(skip_info + "room layout generation failed")
            continue

        # randomize camera target
        if args.random_yaw:
            if perspective:
                yaw = np.random.random() * 2 * np.pi
            else:
                yaw = np.random.randint(4) * np.pi / 2
        else:
            yaw = np.pi / 2  # default align to positive direction of axis x
        if len(args.cam_pitch) == 1:
            pitch = args.cam_pitch[0]
        else:
            pitch = np.random.random() * (args.cam_pitch[1] - args.cam_pitch[0]) + args.cam_pitch[0]
        pitch = np.deg2rad(pitch)
        camera_target = np.array([px + np.sin(yaw), py + np.cos(yaw), pz + np.tan(pitch)], dtype=np.float32)
        camera["target"] = camera_target
        camera["up"] = np.array([0, 0, 1], dtype=np.float32)
        if perspective:
            camera['K'] = s.renderer.get_intrinsics().astype(np.float32) / 2

        # generate camera layout and check if the camaera is valid
        layout = {'manhattan_pix': manhattan_pix_layout_from_room_layout(camera, room_layout)}
        data['layout'] = layout
        if layout['manhattan_pix'] is None:
            print(skip_info + "manhattan pixel layout generation failed")
            continue
        if args.cuboid_lo:
            layout['cuboid_world'] = cuboid_world_layout_from_room_layout(room_layout)
        if args.world_lo:
            layout['manhattan_world'] = manhattan_world_layout_from_room_layout(room_layout)
        if args.horizon_lo:
            layout['horizon'] = horizon_layout_gt_from_scene_data(data)

        # filter out camera by object layout
        camera_point = Point(*camera['pos'][:2])
        if any(obj.contains(camera_point) for obj in object_layout):
            print(skip_info + "inside or above/below obj")
            continue
        nearest_point, _ = shapely.ops.nearest_points(object_layout.boundary, camera_point)
        distance_obj = camera_point.distance(nearest_point)
        # if distance_obj < 0.5:
        #     print(f"{skip_info}too close ({distance_obj:.3f} < 0.5) to object")
        #     continue

        # render
        render_results = render_camera(s.renderer, camera, args.render_type,
                                       perspective, obj_groups, scene.objects_by_id)

        # extract object params
        if 'seg' in args.render_type:
            data['objs'] = []
            ids = np.unique(render_results['seg']).astype(np.int).tolist()

            for obj_id in ids:
                if obj_id not in objs.keys():
                    continue
                obj_dict = objs[obj_id].copy()
                obj_dict['id'] = obj_id

                # get object bdb2d
                obj_dict.update(seg2obj(render_results['seg'], obj_id))
                if not is_obj_valid(obj_dict):
                    continue

                # rotate camera to recenter bdb3d
                recentered_trans = IGTransform.level_look_at(data, obj_dict['bdb3d']['centroid'])
                corners = recentered_trans.world2campix(bdb3d_corners(obj_dict['bdb3d']))
                full_convex = MultiPoint(corners).convex_hull
                # pyplot.plot(*full_convex.exterior.xy)
                # pyplot.axis('equal')
                # pyplot.show()

                # filter out objects by ratio of visible part
                contour = obj_dict['contour']
                contour_points = np.stack([contour['x'], contour['y']]).T
                visible_convex = MultiPoint(contour_points).convex_hull
                if visible_convex.area / full_convex.area < 0.2:
                    continue

                data['objs'].append(obj_dict)

            if not data['objs']:
                print(f"{skip_info}no object in the frame")
                continue

        # construction IGScene
        ig_scene = IGScene(data)

        # generate relation
        if args.relation:
            relation_optimization = RelationOptimization(expand_dis=args.expand_dis)
            relation_optimization.generate_relation(ig_scene)

        # save data
        ig_scene.image_io = ImageIO(render_results)
        camera_folder = os.path.join(output_folder, data['name'])
        ig_scene.image_io.save(camera_folder)
        ig_scene.to_pickle(camera_folder)
        if args.json:
            ig_scene.to_json(camera_folder)
        i_camera += 1
        camera_paths.append(os.path.join(data['scene'], data['name']))

    s.disconnect()
    return camera_paths


def main():
    parser = argparse.ArgumentParser(
        description='Render RGB panorama from iGibson scenes.')
    parser.add_argument('--scene', dest='scene_name',
                        type=str, default=None,
                        help='The name of the scene to load')
    parser.add_argument('--source', dest='scene_source',
                        type=str, default='IG',
                        help='The name of the source dataset, among [IG,CUBICASA,THREEDFRONT]')
    parser.add_argument('--output', type=str, default='data/igibson',
                        help='The path of the output folder')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating camera pose')
    parser.add_argument('--width', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of output image')
    parser.add_argument('--processes', type=int, default=0,
                        help='Number of threads')
    parser.add_argument('--renders', type=int, default=10,
                        help='Number of renders per room')
    parser.add_argument('--cam_height', type=float, default=[1.6], nargs='+',
                        help='Height of camera in meters (provide two numbers to specify range)')
    parser.add_argument('--cam_pitch', type=float, default=[0.], nargs='+',
                        help='Pitch of camera in degrees (provide two numbers to specify range)')
    parser.add_argument('--random_yaw', default=False, action='store_true',
                        help='Randomize camera yaw')
    parser.add_argument('--vertical_fov', type=float, default=None,
                        help='Fov for perspective camera in degrees')
    parser.add_argument('--render_type', type=str, default=['rgb', 'seg', 'sem', 'depth'], nargs='+',
                        help='Types of renders (rgb/normal/seg/sem/depth/3d)')
    parser.add_argument('--strict', default=False, action='store_true',
                        help='Raise exception if render fails')
    parser.add_argument('--super_sample', type=int, default=2,
                        help='Set to greater than 1 to use super_sample')
    parser.add_argument('--no_physim', default=False, action='store_true',
                        help='Do physical simulation before rendering')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Ratio of train split')
    parser.add_argument('--horizon_lo', default=False, action='store_true',
                        help='Generate Horizon format layout GT from manhattan layout')
    parser.add_argument('--json', default=False, action='store_true',
                        help='Save camera info as json too')
    parser.add_argument('--cuboid_lo', default=False, action='store_true',
                        help='Generate cuboid world frame layout from manhattan layout')
    parser.add_argument('--world_lo', default=False, action='store_true',
                        help='Generate manhatton world frame layout')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--split', default=False, action='store_true',
                        help='Split train/test dataset without rendering')
    parser.add_argument('--random_obj', default=None, action='store_true',
                        help='Use the 10 objects randomization for each scene')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing renders')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--crop_width', default=None, type=int,
                        help='Width of image cropped of ground truth 2d bounding box')
    parser.add_argument('--relation', default=False, action='store_true',
                        help='Generate relationships')
    args = parser.parse_args()

    assert args.vertical_fov is not None or args.cam_pitch != 0, \
        "cam_pitch not supported for panorama rendering"
    assert all(r in ['rgb', 'normal', 'seg', 'sem', 'depth', '3d'] for r in args.render_type), \
        "please check render type setting"
    assert args.vertical_fov is not None or not any(r in args.render_type for r in ['normal']), \
        "render type 'normal' not supported for panorama"

    # prepare arguments
    scene_names = []
    if args.scene_name is None:
        dataset_path = {'IG': gibson2.ig_dataset_path,
                        'CUBICASA': gibson2.cubicasa_dataset_path,
                        'THREEDFRONT': gibson2.threedfront_dataset_path}
        dataset_path = dataset_path[args.scene_source]
        dataset_path = os.path.join(dataset_path, "scenes")
        for n in os.listdir(dataset_path):
            if n != 'background' \
                    and os.path.isdir(os.path.join(dataset_path, n)) \
                    and n.endswith('_int'):
                scene_names.append(n)
    else:
        scene_names = [args.scene_name]

    # begin rendering
    if not args.split:
        args_list = []
        args_dict = args.__dict__.copy()
        for scene_name in scene_names:
            args_dict['scene_name'] = scene_name
            if args.resume:
                cameras = glob(os.path.join(args.output, scene_name, '*', 'data.pkl'))
                i_cameras = np.array([int(os.path.basename(os.path.dirname(c))) for c in cameras])
            if args.random_obj:
                for random_idx in range(10):
                    if args.resume and np.sum(
                            ((random_idx * args.renders) <= i_cameras) &
                            (i_cameras < (random_idx + 1) * args.renders)
                    ) >= args.renders:
                        continue
                    args_dict['seed'] = args.seed + random_idx
                    args_dict['random_obj'] = random_idx
                    args_list.append(argparse.Namespace(**args_dict))
            else:
                if args.resume and len(i_cameras) < args.renders:
                    continue
                args_list.append(argparse.Namespace(**args_dict))
        print(f"{len(args_list)} scenes to be rendered")

        if args.processes == 0:
            r = []
            for a in tqdm(args_list):
                r.append(_render_scene_fail_remove(a))
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(_render_scene_fail_remove, args_list), total=len(args_list)))

    # split dataset
    split = {'train': [], 'test': []}
    scenes = {'train': set(), 'test': set()}
    cameras = glob(os.path.join(args.output, '*', '*', 'data.pkl'))
    for camera in cameras:
        scene_name = camera.split('/')[-3]
        is_train = hash_split(args.train, scene_name)
        path = os.path.join(*camera.split('/')[-3:])
        if is_train:
            split['train'].append(path)
            scenes['train'].add(scene_name)
        else:
            split['test'].append(path)
            scenes['test'].add(scene_name)

    print(f"{len(scenes['train']) + len(scenes['test'])} scenes, "
          f"{len(scenes['train'])} train scenes, "
          f"{len(scenes['test'])} test scenes, "
          f"{len(split['train'])} train cameras, "
          f"{len(split['test'])} test cameras")

    for k, v in split.items():
        v.sort()
        write_json(v, os.path.join(args.output, k + '.json'))


if __name__ == "__main__":
    main()
