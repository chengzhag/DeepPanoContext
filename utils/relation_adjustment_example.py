import argparse
import os

import cv2
import numpy as np

from models.pano3d.dataloader import collate_fn
from utils.igibson_utils import IGScene
from utils.image_utils import save_image
from utils.relation_utils import RelationOptimization, visualize_relation, relation_from_bins, compare_bdb3d
from utils.transform_utils import IGTransform


def main():
    parser = argparse.ArgumentParser(
        description='Relation optimization testing.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the iGibson dataset')
    parser.add_argument('--igibson_obj_dataset', type=str, default='data/igibson_obj',
                        help='The path of the iGibson object dataset')
    parser.add_argument('--output', type=str, default='out/tmp',
                        help='The path of the output folder')
    parser.add_argument('--scene', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--id', type=str, default=None,
                        help='The id of the camera to visualize')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip visualizing mesh GT, which is time consuming')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Show visualization results instead of saving to output')
    parser.add_argument('--toleration_dis', type=float, default=0.0,
                        help='Toleration distance when calculating optimization loss')
    parser.add_argument('--expand_dis', type=float, default=0.1,
                        help='Distance of bdb3d expansion when generating collision and touch relation '
                             'between objects, walls, floor and ceiling')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for generating camera pose')
    args = parser.parse_args()
    np.random.seed(args.seed)

    scene_folder = os.path.join(args.dataset, args.scene)
    camera_folder = os.path.join(scene_folder, args.id)
    scene = IGScene.from_pickle(camera_folder, args.igibson_obj_dataset)
    relation_optimization = RelationOptimization(
        visual_path=args.output, expand_dis=args.expand_dis, toleration_dis=args.toleration_dis,
        visual_frames=30
    )

    # inference relationships between objects from GT
    relation_optimization.generate_relation(scene)

    # visualize GT scene
    background = visualize_relation(scene, layout=True, relation=False)
    save_image(background, os.path.join(args.output, f'gt.png'))
    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    background = (background * 0.5).astype(np.uint8)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    relation_optimization.visual_background = background

    # randomize scene bdb3d
    gt_data = collate_fn(scene.data)
    relation_optimization.randomize_scene(scene)

    # visualize randomized scene
    image = visualize_relation(scene, background)
    save_image(image, os.path.join(args.output, f'diff.png'))

    # to tensor
    optim_data = collate_fn(scene.data)
    relation_label = relation_from_bins(optim_data, None)
    optim_data['objs'].update(relation_label['objs'])
    optim_data['relation'] = relation_label['relation']

    # run relation optimization with visualization
    optim_bdb3d = relation_optimization.optimize(optim_data, visual=True)
    # optim_data['objs']['bdb3d'].update(IGTransform(optim_data).campix2world(optim_bdb3d))

    # evaluate bdb3d with gt
    optim_bdb3d.update(IGTransform(optim_data).campix2world(optim_bdb3d))
    compare_bdb3d(optim_data['objs']['bdb3d'], gt_data['objs']['bdb3d'], 'to_gt_before: ')
    compare_bdb3d(optim_bdb3d, optim_data['objs']['bdb3d'], 'from_initial: ')
    compare_bdb3d(optim_bdb3d, gt_data['objs']['bdb3d'], 'to_gt: ')


if __name__ == "__main__":
    main()
