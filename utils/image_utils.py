import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import imageio


class ImageIO(dict):
    def __init__(self, images=None):
        if images is None:
            images = {}
        self.image_path = None
        super().__init__(images)

    @classmethod
    def from_file(cls, key_path_pair: (dict, list)):
        image_io = cls()
        if isinstance(key_path_pair, list):
            key_path_pair = {os.path.basename(os.path.splitext(p)[0]): p for p in key_path_pair}
        image_io.image_path = key_path_pair
        return image_io

    def __getitem__(self, item):
        if item not in super().keys():
            im = load_image(self.image_path[item])
            if item == 'depth':
                depth = (im >> 3) | (im << 13)
                im = depth.astype(np.float32) / 1000.
            super().__setitem__(item, im)
        return super().__getitem__(item)

    def save(self, folder):
        assert self
        os.makedirs(folder, exist_ok=True)
        for k, v in self.items():
            if k == 'depth':
                im = v.copy()
                im = (im * 1000).astype(np.uint16)
                v = (im << 3) | (im >> 13)
            save_image(v, os.path.join(folder, k + '.png'))

    def link(self, folder):
        assert self.image_path
        os.makedirs(folder, exist_ok=True)
        for src in self.image_path.values():
            dst = os.path.join(folder, os.path.basename(src))
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)


def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(image).save(save_path)


def load_image(path):
    return np.array(Image.open(path))


def show_image(image):
    plt.cla()
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def save_gif(frames, save_path, duration=1.):
    imageio.mimsave(save_path, frames, 'GIF', duration=duration)


class GifIO:
    def __init__(self, frames=None, duration=1.):
        self.frames = [] if frames is None else frames
        self.duration = duration

    def append(self, frame):
        self.frames.append(frame)

    def clear(self):
        self.frames = []

    def save(self, save_path):
        save_gif(self.frames, save_path, self.duration)
