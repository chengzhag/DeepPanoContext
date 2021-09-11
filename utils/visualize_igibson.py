import os
import argparse
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from models.detector.dataset import register_igibson_detection_dataset
from utils.igibson_utils import IGScene
from utils.image_utils import save_image, show_image
from models.pano3d.dataloader import IGSceneDataset
from utils.visualize_utils import IGVisualizer


def visualize_camera(args):
    scene_folder = os.path.join(args.dataset, args.scene) if args.scene is not None else args.dataset
    camera_folder = os.path.join(scene_folder, args.id)
    scene = IGScene.from_pickle(camera_folder, args.igibson_obj_dataset)
    visualizer = IGVisualizer(scene, gpu_id=args.gpu_id, debug=args.debug)

    if not args.skip_render:
        render = visualizer.render(background=200)

    image = visualizer.image('rgb')
    image = visualizer.layout(image, total3d=True)
    image = visualizer.objs3d(image, bbox3d=True, axes=True, centroid=True, info=False)
    if not args.show:
        save_path = os.path.join(scene_folder, args.id, 'det3d.png')
        save_image(image, save_path)
    image = visualizer.bfov(image, include=('walls', 'objs'))
    image = visualizer.bdb2d(image)

    if args.show:
        if not args.skip_render:
            show_image(render)
            if 'K' in scene['camera']:
                birds_eye = visualizer.render(background=200, camera='birds_eye')
                show_image(birds_eye)
                up_down = visualizer.render(background=200, camera='up_down')
                show_image(up_down)
        show_image(image)
    else:
        if not args.skip_render:
            save_path = os.path.join(scene_folder, args.id, 'render.png')
            save_image(render, save_path)
        save_path = os.path.join(scene_folder, args.id, 'visual.png')
        save_image(image, save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson scenes.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the iGibson dataset')
    parser.add_argument('--igibson_obj_dataset', type=str, default='data/igibson_obj',
                        help='The path of the iGibson object dataset')
    parser.add_argument('--scene', type=str, default=None,
                        help='The name of the scene to visualize')
    parser.add_argument('--id', type=str, default=None,
                        help='The id of the camera to visualize')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU used for rendering')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Save temporary files to out/tmp/ when rendering')
    parser.add_argument('--skip_render', default=False, action='store_true',
                        help='Skip visualizing mesh GT, which is time consuming')
    parser.add_argument('--show', default=False, action='store_true',
                        help='Show visualization results instead of saving')
    args = parser.parse_args()
    register_igibson_detection_dataset(args.dataset)


    if args.scene is not None and args.id is not None:
        args_dict = args.__dict__.copy()
        visualize_camera(argparse.Namespace(**args_dict))
    elif args.scene is None and args.id is None:
        cameras = IGSceneDataset({'data': {'split': args.dataset, 'igibson_obj': args.igibson_obj_dataset}}).split
        args_dict = args.__dict__.copy()
        args_list = []
        for camera in cameras:
            camera_dirs = camera.split('/')
            if len(camera_dirs) == 4:
                args_dict['scene'], args_dict['id'] = None, camera_dirs[-2]
            else:
                args_dict['scene'], args_dict['id'] = camera_dirs[-3:-1]
            args_dict['id'] = os.path.splitext(args_dict['id'])[0]
            args_list.append(argparse.Namespace(**args_dict))
        if args.processes == 0:
            for a in tqdm(args_list):
                visualize_camera(a)
        else:
            with Pool(processes=args.processes) as p:
                r = list(tqdm(p.imap(visualize_camera, args_list), total=len(args_list)))
    else:
        raise Exception('Should specify both scene and id for visualizing single camera. ')

if __name__ == "__main__":
    main()

