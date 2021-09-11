import numpy as np
import seaborn as sns
import scipy.io as sio
import os


IG56CLASSES = [
    'basket', 'bathtub', 'bed', 'bench', 'bottom_cabinet',
    'bottom_cabinet_no_top', 'carpet', 'chair', 'chest',
    'coffee_machine', 'coffee_table', 'console_table',
    'cooktop', 'counter', 'crib', 'cushion', 'dishwasher',
    'door', 'dryer', 'fence', 'floor_lamp', 'fridge',
    'grandfather_clock', 'guitar', 'heater', 'laptop',
    'loudspeaker', 'microwave', 'mirror', 'monitor',
    'office_chair', 'oven', 'piano', 'picture', 'plant',
    'pool_table', 'range_hood', 'shelf', 'shower', 'sink',
    'sofa', 'sofa_chair', 'speaker_system', 'standing_tv',
    'stool', 'stove', 'table', 'table_lamp', 'toilet',
    'top_cabinet', 'towel_rack', 'trash_can', 'treadmill',
    'wall_clock', 'wall_mounted_tv', 'washer', 'window'
]

IG59CLASSES = IG56CLASSES + ['walls', 'floors', 'ceilings']

WIMR11CLASSES = [
    'bed', 'painting', 'table', 'mirror', 'window', 'chair',
    'sofa', 'door', 'cabinet', 'bedside', 'tv',
]

WIMR2PC = {
    'bed': [
        'bed',
        'bed:outside',
        'bed:outside room',
        'bed:outside room ',
        'outside room bed',
        'baby bed'
    ],
    'painting': [
        'painting',
        'paitning',
        'paint',
        'picture',
        'picture: inside',
        'outside room picture',
        'picture:outside room',
        'picture: outside',
        'picture: outside room',
        'photo',
        'poster'
    ],
    'table': [
        'table',
        'table:outside room ',
        'table: outside room',
        'table:outside room',
        'outside room table',
        'round table',
        'round table:outside',
        'dressing table',
        'desk',
        'desk:outside',
        'desk:outside room',
        'dining table',
        'dining table ',
        'dining table:outside ',
        'dining table:outside',
        'dining table: outside',
        'outside dining table',
        'console table',
        'console table ',
        'console table',
        'console table:outside',
        'coffee table',
        'coffee table:outside',
        'end table',
        'bar table',
        'bar table:outside room ',
        'kitchen table',
        'desk and chair'
    ],
    'mirror': [
        'mirror',
        'mirror:outside room',
        'outside room mirror'
    ],
    'window': [
        'window',
        'window:outside',
        'window:outside room',
        'window:outside room ',
        'window: outside room',
        'outside room window'
    ],
    'chair': [
        'chair',
        'chair:outside',
        'chair: outside',
        'chair:outside room',
        'chair:outside room ',
        'chair: outside room',
        'outside room chair',
        'deck chair',
        'deck chair:outside room',
        'chair and table'
    ],
    'sofa': [
        'sofa',
        'sofa:outside',
        'sofa:outside room',
        'sofa:outside room ',
        'outside room sofa'
    ],
    'door': [
        'door',
        'doorway',
        'door non-4pt-polygon'
    ],
    'cabinet': [
        'cabinet',
        'cabinet:outside',
        'cabinet:outside room',
        'cabinet: outside room',
        ' cabinet:outside room',
        'outside room cabinet',
        'wardrobe',
        'wardrobe:outside'
    ],
    'bedside': [
        'bedside',
        'beside',
        'outside room nightstand',
        'nightstand',
        'nightstand:outside'
    ],
    'tv': [
        'tv',
        'TV',
        'tv:outside room ',
        'TV set'
    ],
}

PC2WIMR = {s: t for t, sl in WIMR2PC.items() for s in sl}

PC12CLASSES = [
    'bed', 'painting', 'nightstand', 'window', 'mirror', 'desk',
    'wardrobe', 'tv', 'door', 'chair', 'sofa', 'cabinet'
]

IG2PC = {
    'bed': 'bed', 'crib': 'bed', 
    'picture': 'painting', 
    'chest': 'nightstand', 
    'window': 'window', 
    'mirror': 'mirror', 
    'table': 'desk', 'coffee_table': 'desk', 'console_table': 'desk', 'pool_table': 'desk',
    'shelf': 'wardrobe',
    'standing_tv': 'tv', 'monitor': 'tv', 'wall_mounted_tv': 'tv', 
    'door': 'door', 
    'chair': 'chair', 'office_chair': 'chair', 'sofa_chair': 'chair', 'stool': 'chair', 
    'sofa': 'sofa', 
    'bottom_cabinet': 'cabinet', 'bottom_cabinet_no_top': 'cabinet', # 'top_cabinet': 'cabinet'
}

PC2IG = {
    'bed': 'bed', 
    'painting': 'picture',
    'table': 'table',
    'mirror': 'mirror',
    'window': 'window',
    'chair': 'chair',
    'sofa': 'sofa',
    'door': 'door',
    'cabinet': 'bottom_cabinet',
    'bedside': 'bottom_cabinet',
    'tv': 'standing_tv',
    'shelf': 'shelf'
}

colorbox_path = 'external/cooperative_scene_parsing/evaluation/vis/igibson_colorbox.mat'
igibson_colorbox = np.array(sns.hls_palette(n_colors=len(IG59CLASSES), l=.45, s=.8))
if not os.path.exists(colorbox_path):
    sio.savemat(colorbox_path, {'igibson_colorbox': igibson_colorbox})

metadata = {}


def get_dataset_name(split):
    if split.endswith(('.json', '/')):
        name = split.split('/')[-2]
    else:
        name = os.path.basename(split)
    return name


def generate_bins(b_min, b_max, n):
    bins_width = (b_max - b_min) / n
    bins = np.arange(b_min, b_max, bins_width).astype(np.float32)
    bins = np.stack([bins, bins + bins_width]).T
    return bins


def bins_config():
    metadata['dis_bins'] = generate_bins(0, 12, 6)
    metadata['ori_bins'] = generate_bins(-np.pi, np.pi, 6)
    metadata['rot_bins'] = generate_bins(-np.pi, np.pi, 8)[:, 0]
    metadata['lat_bins'] = generate_bins(-np.pi / 2, np.pi / 2, 6)
    metadata['lon_bins'] = generate_bins(-np.pi, np.pi, 12)
    # metadata['pitch_bins'] = generate_bins(np.deg2rad(-20), np.deg2rad(60), 2)
    metadata['pitch_bins'] = generate_bins(np.deg2rad(10), np.deg2rad(30), 2)
    # metadata['roll_bins'] = generate_bins(np.deg2rad(-20), np.deg2rad(20), 2)
    metadata['roll_bins'] = generate_bins(np.deg2rad(-10), np.deg2rad(10), 2)
    metadata['layout_ori_bins'] = generate_bins(np.deg2rad(-45), np.deg2rad(45), 2)


bins_config()
