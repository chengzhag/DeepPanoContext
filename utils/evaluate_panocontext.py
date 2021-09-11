import argparse
import os
from glob import glob
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import trimesh
from multiprocessing import Pool

from configs.data_config import WIMR11CLASSES, IG2PC
from utils.generate_panocontext import fill_mask
from utils.igibson_utils import IGScene
from utils.visualize_utils import IGVisualizer
from utils.image_utils import show_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate 2D metrics for PanoContext dataset.')
    parser.add_argument('--dataset', type=str, default='data/wimr_detection',
                        help='The path of the dataset')
    parser.add_argument('--output', type=str, default='demo/output_realpano/21032312485256',
                        help='The path of the output folder')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Show intermedia image outputs')
    parser.add_argument('--contour', default=False, action='store_true',
                        help='Test mIoU of detector contour outputs instead of bdb3d projects')
    parser.add_argument('--bdb2d', default=False, action='store_true',
                        help='Test mIoU of detector bdb2d outputs instead of bdb3d projects')
    args = parser.parse_args()

    def evaluate_scene(scene_folder):
        scene_name = os.path.basename(scene_folder)
        samples = []
        sample_points_pix = None
        for data_root in [args.dataset, args.output]:
            scene_pkl = os.path.join(data_root, scene_name, 'data.pkl')
            if not os.path.exists(scene_pkl):
                return None

            scene = IGScene.from_pickle(scene_pkl)
            objs = scene['objs']
            visulization = IGVisualizer(scene)
            if args.debug:
                image = visulization.image('rgb')
                show_image(image)
                image = visulization.bdb2d(image)
                show_image(image)

            # generate semantic segmentation of prediction from far to near
            if 'seg' in scene.image_io.image_path:
                seg = scene.image_io['seg']
                sem = np.zeros_like(seg, dtype=np.uint8)
                for obj in objs:
                    mask = seg == obj['id']
                    sem[mask] = obj['label'] + 1
            else:
                dis = [np.linalg.norm(scene.transform.world2cam3d(o['bdb3d']['centroid'])) for o in objs]
                i_objs = sorted(range(len(dis)), key=lambda k: dis[k])
                sem = visulization.background(0, channels=1)
                for i_obj in reversed(i_objs):
                    obj = objs[i_obj]
                    classname = obj['classname_wimr']
                    label = WIMR11CLASSES.index(classname)

                    mask = visulization.background(0, channels=1)
                    if args.contour:
                        contour = obj['contour']
                        visulization._contour(mask, contour, 255, thickness=1)
                    elif args.bdb2d:
                        bdb2d = obj['bdb2d']
                        visulization._bdb2d(mask, bdb2d, 255, thickness=1)
                    else:
                        bdb3d = obj['bdb3d']
                        visulization._bdb3d(mask, bdb3d, 255, thickness=1)
                    # show_image(mask)

                    # fill polygons
                    fill_mask(mask)
                    sem[mask > 0] = label + 1

                sem = sem[..., 0]

            if args.debug:
                show_image(sem)

            if sample_points_pix is None:
                # uniformly sample point on sphere surface
                sphere = trimesh.primitives.Sphere(radius=0.5, subdivisions=4)
                sample_points_world = sphere.sample(100000)
                sample_points_pix = scene.transform.world2campix(sample_points_world)
                sample_points_pix = np.round(sample_points_pix).astype(np.int)
                sample_points_pix = np.clip(sample_points_pix, 0, [sem.shape[1] - 1, sem.shape[0] - 1])
                sample_points_pix = sample_points_pix[:, (1, 0)]
            samples.append(sem[sample_points_pix[:, 0], sample_points_pix[:, 1]])

        return samples

    scene_folders = glob(os.path.join(args.output, 'pano_*'))
    if args.processes == 0:
        samples = []
        for scene_folder in tqdm(scene_folders):
            samples.append(evaluate_scene(scene_folder))
    else:
        with Pool(processes=args.processes) as p:
            samples = list(tqdm(p.imap(evaluate_scene, scene_folders), total=len(scene_folders)))

    invalid_scenes = len([s for s in samples if s is None])
    est_samples = np.concatenate([s[0] for s in samples if s is not None])
    gt_samples = np.concatenate([s[1] for s in samples if s is not None])
    class_iou = {}
    WIMR11CLASSES_void = ['void'] + WIMR11CLASSES
    for label, classname in enumerate(WIMR11CLASSES_void):
        est_sample = est_samples == label
        gt_sample = gt_samples == label
        intersection = est_sample & gt_sample
        union = est_sample | gt_sample
        iou = intersection.sum() / union.sum()
        class_iou[classname] = iou

    average = np.mean([i for i in class_iou.values() if not np.isnan(i)])
    class_iou['average'] = average
    print({k: f"{v * 100:.2f}" for k, v in class_iou.items()})
    print(f"{invalid_scenes} invalid scenes")
