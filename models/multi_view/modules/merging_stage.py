import cv2
import torch
from torch import nn
import numpy as np
from collections import defaultdict
import os

from models.registers import MODULES
from utils.basic_utils import list_of_dict_to_dict_of_array
from utils.igibson_utils import IGScene
from utils.image_utils import show_image, GifIO, save_image
from utils.relation_utils import visualize_relation
from utils.transform_utils import IGTransform, bdb3d_corners, points2bdb2d


class MergingStage:

    def __init__(self, cfg, _):
        self.cfg = cfg
        self.direct_copy = ('classname', 'label', 'score', 'cls_code')

    def __call__(self, est_data):
        raise NotImplementedError


@MODULES.register_module
class Feedforward(MergingStage):

    def __init__(self, cfg, _):
        super(Feedforward, self).__init__(cfg, _)

    def __call__(self, est_data):
        stitched = list_of_dict_to_dict_of_array(
            est_data['views'],
            force_list={
                'mesh_path': True,
                'objs': {'contour': {'x': True, 'y': True}},
            },
            force_cat=True,
            stop_recursion={'camera': True}
        )

        stitched['camera'] = [{k: v.squeeze(0) for k, v in c.items()} for c in stitched['camera']]
        for k in stitched.keys():
            if k == 'objs':
                stitched[k]['split'] = torch.tensor([[0, len(stitched[k])]])
            else:
                stitched[k] = [stitched[k]]

        return stitched


@MODULES.register_module
class View1(Feedforward):

    def __init__(self, cfg, _):
        super(View1, self).__init__(cfg, _)

    def __call__(self, est_data):
        stitched = Feedforward.__call__(self, est_data)
        stitched['objs'] = est_data['views'][0]['objs']
        return stitched


@MODULES.register_module
class View2(Feedforward):

    def __init__(self, cfg, _):
        super(View2, self).__init__(cfg, _)

    def __call__(self, est_data):
        stitched = Feedforward.__call__(self, est_data)
        stitched['objs'] = est_data['views'][1]['objs']
        return stitched


@MODULES.register_module
class AvgRandChoose(Feedforward):

    def __init__(self, cfg, _):
        super(AvgRandChoose, self).__init__(cfg, _)

    def __call__(self, est_data):
        stitched = Feedforward.__call__(self, est_data)

        affinity = est_data['affinity']
        merged_objs = defaultdict(list)

        # merge associated objs
        object_idx = []
        unmerged = set(range(sum(affinity.shape)))
        for i in range(affinity.shape[0]):
            if affinity[i].type(torch.bool).any():
                j = torch.where(affinity[i])[0].cpu().item()
                j += affinity.shape[0]
                unmerged.remove(i)
                unmerged.remove(j)
                idx = (i, j)[int(np.random.rand() > 0.5)]
                bdb3d = stitched['objs']['bdb3d']
                merged_objs['bdb3d'].append({
                    'size': bdb3d['size'][(i, j), :].mean(0),
                    'centroid': bdb3d['centroid'][(i, j), :].mean(0),
                    'basis': bdb3d['basis'][idx],
                })

                merged_objs['id'].append([i, j])
                object_idx.append(idx)
                for k in self.direct_copy:
                    merged_objs[k].append(stitched['objs'][k][idx])

        # add unmerged objs to codes
        for i in unmerged:
            bdb3d = stitched['objs']['bdb3d']
            merged_objs['bdb3d'].append({
                'size': bdb3d['size'][i],
                'centroid': bdb3d['centroid'][i],
                'basis': bdb3d['basis'][i],
            })

            merged_objs['id'].append([i, -1])
            object_idx.append(i)
            for k in self.direct_copy:
                merged_objs[k].append(stitched['objs'][k][i])

        merged_objs['mesh_extractor'] = stitched['objs']['mesh_extractor'][object_idx]
        merged_objs['bdb3d'] = list_of_dict_to_dict_of_array(merged_objs['bdb3d'])
        for k in ('label', 'score', 'cls_code'):
            merged_objs[k] = torch.stack(merged_objs[k])
        merged_objs['split'] = torch.tensor([[0, len(merged_objs['label'])]])
        stitched['objs'] = merged_objs

        return stitched


@MODULES.register_module
class OptimBdb3d(AvgRandChoose):

    def __init__(self, cfg, _):
        super(OptimBdb3d, self).__init__(cfg, _)
        model_config = cfg.config['model']['merging_stage']
        loss_weights = model_config.get('loss_weights')
        if loss_weights is None:
            loss_weights = {
                'centroid': 0.1, 'size': 1.0,
                'bdb3d_proj': 10.0
            }
        self.weights = loss_weights
        self.visualize = model_config.get('visualize', False)
        self.steps = model_config.get('steps', 100)
        self.lr = model_config.get('lr', 1.)
        self.momentum = model_config.get('momentum', 0.9)
        self.visual_frames = model_config.get('visual_frames', 10)
        self.gif_io = GifIO(duration=0.1)

    def relation_loss(self, out_bdb3d, est_data, transforms_to_bfov=None):
        return

    def __call__(self, est_data):
        # initialize optimization
        stitched = Feedforward.__call__(self, est_data)
        merged = AvgRandChoose.__call__(self, est_data)

        objs = merged['objs']
        bdb3d = objs['bdb3d']
        optim_bdb3d = {k: bdb3d[k].detach().clone() for k in ('size', 'centroid', 'basis')}
        for k, v in optim_bdb3d.items():
            if k != 'basis':
                v.requires_grad = True

        # optimization settings
        optimizer = torch.optim.SGD(list(optim_bdb3d.values()), lr=self.lr, momentum=self.momentum)
        criterion = lambda e: nn.SmoothL1Loss(reduction='mean')(e, torch.zeros_like(e, device=e.device))

        # construct transform
        transforms = []
        for view in est_data['views']:
            transforms.append(IGTransform(view, split='layout'))

        # optimization step
        loss = defaultdict(lambda: 0)

        def closure():
            optimizer.zero_grad()
            loss.clear()

            for i in range(optim_bdb3d['size'].shape[0]):
                ids = objs['id'][i]
                for id in ids:
                    if id < 0:
                        continue

                    # observation error
                    for k in ('size', 'centroid'):
                        loss[k] += torch.abs(optim_bdb3d[k][i] - stitched['objs']['bdb3d'][k][id])

                    # bdb3d projection error
                    corners = bdb3d_corners({k: optim_bdb3d[k][i] for k in optim_bdb3d})
                    corners_pix = transforms[0 if id < est_data['affinity'].shape[0] else 1].world2campix(corners)
                    proj_bdb2d_pix = points2bdb2d(corners_pix)
                    bdb2d = {k: v[id] for k, v in stitched['objs']['bdb2d'].items()}
                    for b in (proj_bdb2d_pix, bdb2d):
                        for k, v in b.items():
                            b[k] = b[k].type(torch.float32) \
                                   / (transforms[0]['width'][0] if 'x' in k else transforms[0]['height'][0])
                    loss['bdb3d_proj'] += torch.abs(torch.stack(
                        [proj_bdb2d_pix[k] - bdb2d[k] for k in bdb2d.keys()]
                    ))

            for k, v in loss.items():
                loss[k] = v / optim_bdb3d['size'].shape[0]

            loss['total'] = sum([
                self.weights[k] * criterion(loss[k]) if len(loss[k]) > 0
                else 0. for k in self.weights.keys() if k in loss
            ])
            if loss['total'] > 0:
                loss['total'].backward(retain_graph=True)
            return loss['total']

        # run optimization
        with torch.enable_grad():
            for step in range(self.steps):
                optimizer.step(closure)

                if self.visualize:
                    loss_str = ', '.join([f"{k}: {torch.mean(l):.4f}" for k, l in loss.items()])
                    print(f"({step}) {loss_str}")

                    every = self.steps // self.visual_frames
                    end = step == self.steps - 1
                    if step == 0:
                        self.gif_io.clear()
                        backgrounds = []
                        for view in est_data['views']:
                            out_scene = IGScene.from_batch(view)[0]
                            background = visualize_relation(out_scene, None, show=end, collision=False, layout=True, relation=False)
                            background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
                            background = (background * 0.5).astype(np.uint8)
                            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
                            backgrounds.append(background)

                    if (step % every == every - 1) or end:
                        images = []
                        for view, background in zip(est_data['views'], backgrounds):
                            optim_data = view.copy()
                            optim_data['objs'] = merged['objs'].copy()
                            optim_data['objs']['bdb3d'] = optim_bdb3d
                            out_scene = IGScene.from_batch(optim_data)[0]
                            image = visualize_relation(out_scene, background, show=end, collision=False, layout=True, relation=False)
                            images.append(image)
                        images.insert(1, np.ones([image.shape[0], image.shape[1] // 10, image.shape[2]], dtype=np.uint8) * 255)
                        image = np.concatenate(images, 1)
                        visual_path = os.path.join(
                            self.cfg.save_path, 'optim_bdb3d', out_scene['scene'],
                            f"{est_data['views'][0]['name'][0]}-{est_data['views'][1]['name'][0]}"
                        )
                        save_image(image, os.path.join(visual_path, f'frame_{len(self.gif_io.frames)}.png'))
                        self.gif_io.append(image)

                    if end:
                        self.gif_io.save(os.path.join(visual_path, 'optim.gif'))

        # update bdb3d
        for v in optim_bdb3d.values():
            v.requires_grad = False

        for k in ('size', 'centroid', 'basis'):
            bdb3d[k] = optim_bdb3d[k]

        return merged
