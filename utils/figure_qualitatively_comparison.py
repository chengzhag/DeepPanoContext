import os
from glob import glob
import shutil
from scipy.io import savemat
from collections import defaultdict

from utils.igibson_utils import IGScene


image_types = ['render', 'rgb', 'det3d']


# qualitatively_comparison
qualitatively_comparison = {
    'Total3D': 'out/total3d_fov/21031718254792',
    'Im3D': 'out/im3d_fov/21031718262127',
    'Ours': 'out/relation_scene_gcn/21032113251731',
    'GT': 'data/igibson'
}

selected_scenes = {
    'Beechwood_1_int': [
        '00069', '00089', '00094', '00043', '00028', '00014', '00009'
    ],
    'Merom_0_int': [
        '00006', '00005'
    ],
    'Merom_1_int': [
        '00097', '00086', '00085', '00083', '00073', '00070', '00062', '00007'
    ]
}

output_folder = '/home/zhangcheng/projects/pano_3d_understanding/paper/qualitatively'

# # failure cases
# qualitatively_comparison = {
#     'Ours': 'out/relation_scene_gcn/21032113251731',
#     'GT': 'data/igibson'
# }
#
# selected_scenes = {
#     'Benevolence_0_int': ['00033', '00052'],
#     'Merom_0_int': ['00086'],
#     'Beechwood_1_int': ['00067', '00034'],
#     'Merom_1_int': ['00078', '00076', '00071']
# }
#
# output_folder = '/home/zhangcheng/projects/pano_3d_understanding/paper/failure'

os.makedirs(output_folder, exist_ok=True)
scene_paths = defaultdict(list)
for method, method_folder in qualitatively_comparison.items():
    for scene_name, selected_images in selected_scenes.items():
        for image in selected_images:
            if method == 'GT':
                image_folder = os.path.join(method_folder, scene_name, image)
            else:
                image_folder = os.path.join(method_folder, 'visualization', scene_name, image)
            src_files = glob(os.path.join(image_folder, '*'))
            for src_file in src_files:
                ids = src_file.split('/')[-3:]
                dst_name = '-'.join(ids)
                if src_file.endswith('.png'):
                    if os.path.splitext(os.path.basename(src_file))[0] not in image_types:
                        continue
                    if src_file.endswith('rgb.png'):
                        dst_name = f"input-{dst_name}"
                    else:
                        dst_name = f"{method}-{dst_name}"
                    dst_file = os.path.join(output_folder, dst_name)
                    shutil.copy(src_file, dst_file)
                elif src_file.endswith('.pkl'):
                    scene = IGScene.from_pickle(src_file)
                    layout = scene['layout']['manhattan_world'].T
                    bdb3ds = []
                    for obj in scene['objs']:
                        bdb3d = obj['bdb3d']
                        bdb3d = {'coeffs': bdb3d['size'], 'basis': bdb3d['basis'], 'centroid': bdb3d['centroid']}
                        bdb3d['label'] = obj['label']
                        bdb3ds.append(bdb3d)
                    dst_name = f"{method}-{os.path.splitext(dst_name)[0] + '.mat'}"
                    dst_file = os.path.join(output_folder, dst_name)
                    savemat(dst_file, mdict={'layout': layout, 'bdb3d': bdb3ds})
                    scene_paths['-'.join(ids[:-1])].append(dst_name)
savemat(os.path.join(output_folder, 'scenes.mat'), mdict={'scenes': scene_paths})


