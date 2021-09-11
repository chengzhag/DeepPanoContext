import os
from glob import glob
import shutil
from scipy.io import savemat
from collections import defaultdict

from models.detector.dataset import register_igibson_detection_dataset
from utils.igibson_utils import IGScene
from utils.visualize_igibson import IGVisualizer
from utils.image_utils import save_image


dataset = 'data/igibson'
register_igibson_detection_dataset(dataset)
igibson_obj_dataset = 'data/igibson_obj'
dataset_samples = {
    'Beechwood_0_int': '00009',
    'Beechwood_1_int': '00096',
    'Merom_1_int': '00090',
    'Benevolence_1_int': '00040',
    'Benevolence_2_int': '00060',
    'Ihlen_0_int': '00082',
}
output_folder = '/home/zhangcheng/projects/pano_3d_understanding/paper/dataset_scenes'
os.makedirs(output_folder, exist_ok=True)

for scene_name, camera_id in dataset_samples.items():
    camera_folder = os.path.join(dataset, scene_name, camera_id)
    camera_pkl = os.path.join(camera_folder, 'data.pkl')
    scene = IGScene.from_pickle(camera_pkl, igibson_obj_dataset)
    visualizer = IGVisualizer(scene)
    dst_name = '-'.join([scene_name, camera_id])

    for key in ['rgb', 'depth', 'render']:
        src_name = f"{key}.png"
        src_path = os.path.join(camera_folder, src_name)
        shutil.copy(src_path, os.path.join(output_folder, f"{dst_name}-{src_name}"))

    for key in ['sem', 'seg']:
        image = visualizer.image(key)
        save_image(image, os.path.join(output_folder, f"{dst_name}-{key}.png"))

    image = visualizer.image('rgb')
    image = visualizer.bfov(image)
    image = visualizer.bdb2d(image)
    save_image(image, os.path.join(output_folder, f"{dst_name}-det2d.png"))

    image = visualizer.image('rgb')
    image = visualizer.objs3d(image, bbox3d=True, axes=True, centroid=True, info=False)
    save_image(image, os.path.join(output_folder, f"{dst_name}-det3d.png"))
