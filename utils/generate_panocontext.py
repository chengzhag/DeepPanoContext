import argparse
import os
from glob import glob
import shutil
from scipy.io import loadmat
from collections import defaultdict
from shapely.geometry import Polygon
import numpy as np
import cv2
from copy import deepcopy
from tqdm import tqdm
import trimesh
from multiprocessing import Pool

from configs.data_config import WIMR11CLASSES, PC2WIMR
from external.HorizonNet.misc.pano_lsd_align import rotatePanorama
from utils.igibson_utils import IGScene
from utils.visualize_utils import IGVisualizer
from utils.transform_utils import bdb3d_corners
from utils.image_utils import save_image, show_image
from utils.render_utils import seg2obj, is_obj_valid
from utils.basic_utils import write_json


def fill_mask(mask):
    if mask[:, 0].any():
        edge_points = np.where(mask[:, 0])[0]
        mask[edge_points[0]:edge_points[-1], 0] = 255
    if mask[:, -1].any():
        edge_points = np.where(mask[:, -1])[0]
        mask[edge_points[0]:edge_points[-1], -1] = 255
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
        # show_image(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess PanoContext dataset for 2D detector training.')
    parser.add_argument('--dataset', type=str, default='data/panoContext_data',
                        help='The path of the dataset')
    parser.add_argument('--test_data', type=str, default='data/sun360_extended/test/rgb/',
                        help='The path of the test/rgb/ split of sun360_extended')
    parser.add_argument('--output', type=str, default='data/wimr_detection',
                        help='The path of the output folder')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Show intermedia image outputs')
    args = parser.parse_args()

    test_folders = glob(os.path.join(args.test_data, 'pano_*'))
    test_scenes = [os.path.splitext(os.path.basename(test_folder))[0] for test_folder in test_folders]
    print(f"Test split: {len(test_scenes)}")
    print(test_scenes)

    for split in ['bedroom', 'living_room']:
        print(f"Processing {split} split...")
        pano_context_list_mat = os.path.join(args.dataset, split, 'IMGLIST.mat')
        pano_context_list = loadmat(pano_context_list_mat)['IMGLIST']
        pano_context_list = [str(i[0][0]) for i in pano_context_list.tolist()[0]]
        pano_context_anno_mat = os.path.join(args.dataset, split, 'ANNO_ALL.mat')
        pano_context_anno = loadmat(pano_context_anno_mat)['ANNO_ALL']

        def generate_scene(scene_folder):
            scene_name = os.path.basename(scene_folder)
            if scene_name not in pano_context_list:
                return
            image_file = glob(os.path.join(scene_folder, '*.jpg'))[0]
            gt_scene = IGScene.from_image(image_file)
            gt_vis = IGVisualizer(gt_scene)
            if args.debug:
                image = gt_vis.image('rgb')
                show_image(image)

            # generate semantic segmentation of ground truth
            gt_id = pano_context_list.index(scene_name)
            if pano_context_anno[gt_id][0]['ANNO3D'].size == 0:
                return
            anno3d = pano_context_anno[gt_id][0]['ANNO3D'][0, 0]
            objs = anno3d['objects']
            objs_list = []
            gt_sem = gt_vis.background(0, channels=1)[..., 0]

            distances = []
            for obj in objs:
                obj = obj[0]
                points = obj['points']
                if points.size == 0:
                    dis = 1000
                else:
                    centroid = np.mean(points, 0)
                    dis = np.linalg.norm(gt_scene.transform.world2cam3d(centroid))
                distances.append(dis)
            i_objs = sorted(range(len(distances)), key=lambda k: distances[k])

            for id, i_obj in enumerate(reversed(i_objs)):
                obj = objs[i_obj][0]
                if obj['name'].size == 0:
                    continue
                classname = str(obj['name'][0])
                if classname == 'room':
                    continue
                if classname not in PC2WIMR:
                    continue
                classname = PC2WIMR[classname]
                label = WIMR11CLASSES.index(classname)

                points = obj['points']
                start_points = points[None, ...].repeat(len(points), axis=0)
                end_points = start_points.transpose([1, 0, 2]).reshape(-1, 3)
                start_points = start_points.reshape(-1, 3)
                mask = gt_vis.background(0, channels=1)[..., 0]
                for start_point, end_point in zip(start_points, end_points):
                    gt_vis._line3d(mask, start_point, end_point, 255, thickness=1, frame='world')
                fill_mask(mask)
                # if args.debug:
                #     show_image(mask)
                mask = np.roll(mask, -mask.shape[1] // 4)
                id = id + 1
                gt_sem[mask > 0] = id
                objs_list.append({
                    'label': label,
                    'classname': classname,
                    'id': id
                })

            objs = []
            for obj in objs_list:
                obj_dict = seg2obj(gt_sem, obj['id'])
                if obj_dict is None:
                    continue
                if not is_obj_valid(obj_dict):
                    continue
                obj.update(obj_dict)
                objs.append(obj)
            if len(objs) == 0:
                return

            # save as pickle
            gt_scene['objs'] = objs
            camera_folder = os.path.join(args.output, scene_name)
            rgb = gt_scene.image_io['rgb'] / 255.0
            vp = pano_context_anno[gt_id][0]['vp'][2::-1, :]
            rgb = rotatePanorama(rgb, vp=vp)
            if anno3d['b_singleroom']:
                R = anno3d['Rc']
                rgb = rotatePanorama(rgb, R=R)
            rgb = (rgb * 255).astype(np.uint8)
            gt_scene.image_io['rgb'] = rgb
            gt_scene.image_io['seg'] = gt_sem
            gt_scene.image_io.save(camera_folder)
            gt_scene.to_pickle(camera_folder)

            # visualize
            rgb = gt_vis.image('rgb')
            image = gt_vis.bdb2d(rgb, dataset='wimr_detection')
            save_image(image, os.path.join(camera_folder, 'det2d.png'))
            if args.debug:
                show_image(image)
            # print(scene_name)

        scene_folders = glob(os.path.join(args.dataset, split, 'pano_*'))
        if args.processes == 0:
            r = []
            for scene_folder in tqdm(scene_folders):
                r.append(generate_scene(scene_folder))
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(generate_scene, scene_folders), total=len(scene_folders)))

    # split dataset
    split = {'train': [], 'test': []}
    scenes = {'train': set(), 'test': set()}
    cameras = glob(os.path.join(args.output, '*', 'data.pkl'))
    for camera in cameras:
        scene_name = camera.split('/')[-2]
        path = os.path.join(*camera.split('/')[-2:])
        is_train = scene_name not in test_scenes
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
        write_json(v, os.path.join(args.output, k + '.json'))

