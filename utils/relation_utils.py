import os
import numpy as np
from shapely.geometry import Polygon, Point
import torch
from torch import nn
from collections import defaultdict

from utils.layout_utils import wall_bdb3d_from_manhattan_world_layout, wall_contour_from_manhattan_pix_layout, \
    manhattan_world_layout_info, manhattan_2d_from_manhattan_world_layout
from .basic_utils import list_of_dict_to_dict_of_array, recursively_to, get_any_array
from .igibson_utils import IGScene
from .image_utils import save_image, show_image, GifIO
from .visualize_utils import IGVisualizer
from .transform_utils import num2bins, label_or_num_from_cls_reg, bdb3d_corners, IGTransform, bdb3d_axis, \
    expand_bdb3d, point_polygon_dis, points2bdb2d
from configs import data_config


def relation_from_bins(data: dict, thres):
    if thres is None:
        thres = {}

    if 'obj_obj_tch' in data:
        rot_bins = data_config.metadata['rot_bins']
        sample = get_any_array(data)
        if isinstance(sample, torch.Tensor):
            rot_bins = torch.from_numpy(data_config.metadata['rot_bins']).to(sample.device)
        new_relation = {}
        for k in data.keys():
            v = data[k]
            if len(v) == 0:
                continue
            score_key = k + '_score'
            bins = rot_bins if k.endswith('rot') else None
            new_relation[k], new_relation[score_key] = label_or_num_from_cls_reg(
                v, bins=bins, return_score=True, threshold=thres.get(k, 0.5))
            # if k.startswith('obj_obj'):
            #     new_relation[k][range(len(new_relation[k])), range(len(new_relation[k]))] = 0
            #     new_relation[score_key][range(len(new_relation[k])), range(len(new_relation[k]))] = 0.
        return new_relation
    else:
        new_data = {}
        if 'relation' in data:
            relations = data['relation']
            new_relations = []
            for i_scene in range(len(relations)):
                relation = relations[i_scene]
                new_relation = relation_from_bins(relation, thres)
                new_relations.append(new_relation)
            new_data['relation'] = new_relations

        objs = data['objs']
        new_objs = {}
        for k in ('floor_tch', 'ceil_tch', 'in_room'):
            new_objs[k], new_objs[k + '_score'] = label_or_num_from_cls_reg(
                objs[k], return_score=True, threshold=thres.get('obj_' + k, 0.5))
        new_data['objs'] = new_objs

        return new_data


def test_bdb3ds(bdb3d_a, bdb3d_b=None, toleration_dis=0.):
    """
    Test if two bdb3d has collision with Separating Axis Theorem.
    If toleration_dis is positive, consider bdb3ds colliding with each other with specified distance as separate.

    Parameters
    ----------
    bdb3d_a: bdb3d dict
    bdb3d_b: bdb3d dict
    toleration_dis: distance of toleration

    Returns
    -------
    labels:
        0: no collision
        1: has collision
        2: a in b
        3: b in a
    collision_err_allaxes: collision errors for backpropagation
    touch_err_allaxes: touch errors for backpropagation
    """
    axes = [bdb3d_axis(bdb3d_a, axis=n) for n in ('x', 'y', 'z')]
    assert not isinstance(axes[0], torch.Tensor) or bdb3d_b is None
    n_bdb3d = len(axes[0])
    bdb3d_a = expand_bdb3d(bdb3d_a, - toleration_dis / 2)
    bdb3d_a_corners = bdb3d_corners(bdb3d_a)
    if bdb3d_b:
        bdb3d_b = expand_bdb3d(bdb3d_b, - toleration_dis / 2)
        bdb3d_b_corners = bdb3d_corners(bdb3d_b)
        axes.extend([bdb3d_axis(bdb3d_b, axis=n) for n in ('x', 'y', 'z')])

    shadow_collision_allaxes = None # if the shadows projected on all three axes have overlaps
    shadow_a_in_b_allaxes = None # if the shadows of a projected on all three axes of b is contained by b
    shadow_b_in_a_allaxes = None
    collision_err_allaxes = None
    touch_err_allaxes = None
    for axis in axes:
        if bdb3d_b:
            bdb3d_a_corners_proj = (axis[..., None, :] * bdb3d_a_corners).sum(axis=-1)
            shadow_a = bdb3d_a_corners_proj.min(axis=-1), bdb3d_a_corners_proj.max(axis=-1)
            bdb3d_b_corners_proj = (axis[..., None, :] * bdb3d_b_corners).sum(axis=-1)
            shadow_b = bdb3d_b_corners_proj.min(axis=-1), bdb3d_b_corners_proj.max(axis=-1)
        else:
            axis = axis.expand(n_bdb3d, -1, -1)[..., None, :].transpose(0, 1)
            bdb3d_a_corners = bdb3d_a_corners.expand(n_bdb3d, -1, -1, -1)
            bdb3d_corners_proj2a = (axis * bdb3d_a_corners).sum(axis=-1)
            shadow_b = torch.stack([bdb3d_corners_proj2a.min(axis=-1)[0], bdb3d_corners_proj2a.max(axis=-1)[0]])
            shadow_a = shadow_b[:, range(n_bdb3d), range(n_bdb3d)]
            shadow_a = shadow_a[..., None].expand(-1, -1, n_bdb3d)

        a_in_b = [(shadow_b[0] <= shadow_a_end) & (shadow_a_end <= shadow_b[1]) for shadow_a_end in shadow_a]
        b_in_a = [(shadow_a[0] <= shadow_b_end) & (shadow_b_end <= shadow_a[1]) for shadow_b_end in shadow_b]

        shadow_a_in_b = a_in_b[0] & a_in_b[1] # if the shadows of a is contained by b
        shadow_a_in_b_allaxes = shadow_a_in_b if shadow_a_in_b_allaxes is None \
            else shadow_a_in_b_allaxes & shadow_a_in_b
        shadow_b_in_a = b_in_a[0] & b_in_a[1]
        shadow_b_in_a_allaxes = shadow_b_in_a if shadow_b_in_a_allaxes is None \
            else shadow_b_in_a_allaxes & shadow_b_in_a
        shadow_collision = a_in_b[0] | a_in_b[1] | b_in_a[0] | b_in_a[1] # if shadows have overlaps
        shadow_collision_allaxes = shadow_collision if shadow_collision_allaxes is None \
            else shadow_collision_allaxes & shadow_collision

        if not bdb3d_b:
            collision_err = torch.min(torch.abs(shadow_a[1] - shadow_b[0]), torch.abs(shadow_a[0] - shadow_b[1]))
            collision_err_allaxes = collision_err \
                if collision_err_allaxes is None else (collision_err_allaxes + collision_err)
            touch_err = collision_err.clone()
            touch_err[shadow_collision] = 0.
            touch_err_allaxes = touch_err if touch_err_allaxes is None else (touch_err_allaxes + touch_err)

    if bdb3d_b:
        if shadow_b_in_a_allaxes:
            return 3
        if shadow_a_in_b_allaxes:
            return 2
        elif shadow_collision_allaxes:
            return 1
        return 0
    else:
        labels = torch.zeros_like(shadow_collision_allaxes, dtype=torch.uint8, device=shadow_collision_allaxes.device)
        labels[shadow_collision_allaxes & shadow_collision_allaxes.T] = 1
        labels[shadow_a_in_b_allaxes & shadow_a_in_b_allaxes.T] = 2
        labels[shadow_b_in_a_allaxes & shadow_b_in_a_allaxes.T] = 3
        labels[range(n_bdb3d), range(n_bdb3d)] = 0

        collision_err_allaxes = collision_err_allaxes + collision_err_allaxes.T
        collision_err_allaxes[torch.logical_not(labels)] = 0.
        collision_err_allaxes[range(n_bdb3d), range(n_bdb3d)] = 0.

        touch_err_allaxes = touch_err_allaxes + touch_err_allaxes.T
        touch_err_allaxes[labels > 0] = 0.
        touch_err_allaxes[range(n_bdb3d), range(n_bdb3d)] = 0.

        return labels, collision_err_allaxes, touch_err_allaxes


def visualize_relation(scene, background=None, wall3d=False,
                       relation=True, show=True, collision=False, layout=False):
    visualizer = IGVisualizer(scene)
    image = visualizer.background(0) if background is None else background
    if layout:
        image = visualizer.layout(image, 50, thickness=1)
    if wall3d:
        image = visualizer.wall3d(image, 50, thickness=1)
    image = visualizer.objs3d(image, bbox3d=True, axes=False, centroid=False, info=False, thickness=1)
    image = visualizer.objs3d(image, bbox3d=False, axes=False, centroid=True, info=False, thickness=2)
    if relation:
        image = visualizer.relation(image, thickness=2, collision=collision)
    if show:
        show_image(image)
    return image


class RelationOptimization:

    def __init__(self, loss_weights=None, expand_dis=None, toleration_dis=None,
                 score_weighted=False, score_thres=None,
                 visual_path=None, visual_background=None, visual_frames=10):
        if loss_weights is None:
            loss_weights = {
                'center': 0.0001, 'size': 1.0, 'dis': 0.01, 'ori': 0.001,
                'obj_obj_col': 0.1, 'obj_wall_col': 1.0,
                'obj_floor_col': 1.0, 'obj_ceil_col': 1.0,
                'obj_obj_tch': 0.1, 'obj_wall_tch': 1.0,
                'obj_floor_tch': 1.0, 'obj_ceil_tch': 1.0,
                'obj_obj_rot': 0.01, 'obj_wall_rot': 0.1,
                'obj_obj_dis': 0.01,
                'bdb3d_proj': 10.0
            }
        self.score_thres = defaultdict(lambda: 0.5)
        if score_thres is not None:
            self.score_thres.update(score_thres)
        self.weights = loss_weights
        self.visual_path = visual_path
        self.visual_background = visual_background
        self.visual_frames = visual_frames
        self.expand_dis = expand_dis
        self.toleration_dis = toleration_dis
        self.score_weighted = score_weighted
        self.gif_io = GifIO(duration=0.2)

    def generate_relation(self, scene):
        expand_dis = self.expand_dis
        objs = scene['objs']
        n_objs = len(objs)
        obj_obj_rot = np.zeros([n_objs, n_objs], dtype=np.int)  # angles of clockwise rotation from a to b
        obj_obj_dis = np.zeros_like(obj_obj_rot, dtype=np.bool)  # is a further than b
        obj_obj_tch = obj_obj_dis.copy()  # is a touching b

        # object - object relationships
        for obj in objs:
            obj['bdb3d'].update(scene.transform.world2campix(obj['bdb3d']))

        for i_a, obj_a in enumerate(objs):
            for i_b, obj_b in enumerate(objs):
                if i_a == i_b:
                    continue
                bdb3d_a = obj_a['bdb3d']
                bdb3d_b = obj_b['bdb3d']
                rot = np.mod(bdb3d_b['ori'] - bdb3d_a['ori'], np.pi * 2)
                rot = rot - np.pi * 2 if rot > np.pi else rot
                obj_obj_rot[i_a, i_b] = num2bins(data_config.metadata['rot_bins'], rot)
                obj_obj_dis[i_a, i_b] = bdb3d_a['dis'] > bdb3d_b['dis']
                obj_obj_tch[i_a, i_b] = bool(test_bdb3ds(bdb3d_a, bdb3d_b, - expand_dis))

        # object - floor/ceiling relationships
        layout = scene['layout']['manhattan_world']
        layout_info = manhattan_world_layout_info(layout)
        for obj in objs:
            bdb3d = expand_bdb3d(obj['bdb3d'], expand_dis)
            corners = bdb3d_corners(bdb3d)
            bottom = corners[:, -1].min()
            top = corners[:, -1].max()

            obj['floor_tch'] = bottom < layout_info['floor']
            obj['ceil_tch'] = top > layout_info['ceil']
            corners_2d = corners[:4, :2]
            obj['in_room'] = any(layout_info['layout_poly'].contains(Point(c)) for c in corners_2d)

        walls_bdb3d = wall_bdb3d_from_manhattan_world_layout(layout)

        # object - wall relationships
        # get contour from layout estimation
        walls = []
        for wall_bdb3d in walls_bdb3d:
            wall_bdb3d['bdb3d'].update(scene.transform.world2campix(wall_bdb3d['bdb3d']))
            walls.append(wall_bdb3d)

        obj_wall_rot = np.zeros(
            [n_objs, len(walls)], dtype=np.int)  # angles of clockwise rotation from object to wall
        obj_wall_tch = np.zeros_like(obj_wall_rot, dtype=np.bool)  # is obj touching wall
        for i_obj, obj in enumerate(objs):
            for i_wall, wall in enumerate(walls):
                bdb3d_obj = obj['bdb3d']
                bdb3d_wall = wall['bdb3d']
                rot = np.mod(bdb3d_wall['ori'] - bdb3d_obj['ori'], np.pi * 2)
                rot = rot - np.pi * 2 if rot > np.pi else rot
                obj_wall_rot[i_obj, i_wall] = num2bins(data_config.metadata['rot_bins'], rot)
                is_touching = test_bdb3ds(obj['bdb3d'], wall['bdb3d'], - expand_dis) if obj['in_room'] else 0
                obj_wall_tch[i_obj, i_wall] = is_touching

        # write to scene data
        if 'walls' in scene.data:
            for wall_old, wall_new in zip(scene['walls'], walls):
                wall_old.update(wall_new)
        else:
            scene['walls'] = walls
        scene['relation'] = {
            'obj_obj_rot': obj_obj_rot,
            'obj_obj_dis': obj_obj_dis,
            'obj_obj_tch': obj_obj_tch,
            'obj_wall_rot': obj_wall_rot,
            'obj_wall_tch': obj_wall_tch
        }

    def relation_loss(self, out_bdb3d, est_data, transforms_to_bfov=None):
        loss = {}
        relation = est_data['relation'][0]
        layout = est_data['layout']['manhattan_world'][0]
        est_bdb3d_wall = est_data['walls']['bdb3d']
        objs = est_data['objs']
        est_bdb3d_obj = objs['bdb3d']
        in_room = self.label2weight('obj_in_room', objs['in_room'], objs['in_room_score'])
        n_objs = len(in_room)
        toleration_dis = self.toleration_dis
        obj_obj_err_mask = 1 - torch.eye(n_objs, device=layout.device)

        # rel_in_room = torch.ones([n_objs, n_objs], device=device) > 0
        # rel_in_room[torch.logical_not(in_room), :] = False
        # rel_in_room[:, torch.logical_not(in_room)] = False

        transform = IGTransform(est_data)
        out_bdb3d_3d = transform.campix2world(out_bdb3d)
        corners = bdb3d_corners(out_bdb3d_3d)

        # observation error
        for k in ('center', 'size', 'dis', 'ori'):
            loss[k] = torch.abs(out_bdb3d[k] - est_bdb3d_obj[k])
            if self.score_weighted:
                if k == 'size':
                    weight = self.label2weight(key=k, score=objs['score'])[:, None]
                elif k in ('dis', 'ori'):
                    weight = self.label2weight(key=k, score=est_bdb3d_obj[k + '_score'])
                else:
                    weight = 1.
                loss[k] *= weight

        # bdb3d projection error
        if transforms_to_bfov is not None:
            corners_rad = transforms_to_bfov.world2camrad(corners)
            proj_bdb2d_rad = points2bdb2d(corners_rad)
            bfov = objs['bfov']
            det_bdb2d_rad = {
                'x1': -bfov['x_fov'] / 2, 'x2': bfov['x_fov'] / 2,
                'y1': -bfov['y_fov'] / 2, 'y2': bfov['y_fov'] / 2
            }
            loss['bdb3d_proj'] = torch.abs(torch.stack(
                [proj_bdb2d_rad[k] - det_bdb2d_rad[k] for k in det_bdb2d_rad.keys()]
            ))

        # obj_obj_col error
        all_bdb3d = {k: torch.cat([out_bdb3d_3d[k], est_bdb3d_wall[k]]) for k in out_bdb3d_3d.keys()}
        has_collision, collision_err, _ = test_bdb3ds(all_bdb3d, toleration_dis=toleration_dis)
        loss['obj_obj_col'] = collision_err[:n_objs, :n_objs] * obj_obj_err_mask

        # obj_wall_col error
        layout_2d = manhattan_2d_from_manhattan_world_layout(layout)
        corners_2d = corners[..., :2].reshape(-1, 2)
        dis = torch.relu(point_polygon_dis(corners_2d, layout_2d) - toleration_dis)
        loss['obj_wall_col'] = dis.view(n_objs, 8) * in_room[:, None]

        # obj_floor_col/obj_ceil_col error
        layout_z = layout[:, -1]
        floor = layout_z.min()
        ceil = layout_z.max()
        corners_z = corners[..., -1]
        loss['obj_floor_col'] = torch.relu(floor - corners_z - toleration_dis)
        loss['obj_ceil_col'] = torch.relu(corners_z - ceil - toleration_dis)

        # obj_obj_tch/obj_wall_tch error
        has_collision, _, touch_err = test_bdb3ds(all_bdb3d, toleration_dis=-toleration_dis)

        obj_obj_tch_err = touch_err[:n_objs, :n_objs]
        obj_obj_tch_err *= self.label2weight('obj_obj_tch', relation['obj_obj_tch'], relation['obj_obj_tch_score'])
        loss['obj_obj_tch'] = obj_obj_tch_err * obj_obj_err_mask

        obj_wall_tch_err = touch_err[:n_objs, n_objs:]
        obj_wall_tch_err *= self.label2weight('obj_wall_tch', relation['obj_wall_tch'], relation['obj_wall_tch_score'])
        loss['obj_wall_tch'] = obj_wall_tch_err * in_room[:, None]

        # obj_floor_tch/obj_ceil_tch error
        bottom = corners_z.min(-1)[0]
        top = corners_z.max(-1)[0]

        obj_floor_tch_err = torch.relu(bottom - floor - toleration_dis)
        obj_floor_tch_err *= self.label2weight('obj_floor_tch', objs['floor_tch'], objs['floor_tch_score'])
        loss['obj_floor_tch'] = obj_floor_tch_err

        obj_ceil_tch_err = torch.relu(ceil - top - toleration_dis)
        obj_ceil_tch_err *= self.label2weight('obj_ceil_tch', objs['ceil_tch'], objs['ceil_tch_score'])
        loss['obj_ceil_tch'] = obj_ceil_tch_err

        # obj_obj_dis error
        obj_obj_dis = relation['obj_obj_dis']  # is a further than b
        dis_a = out_bdb3d['dis']
        dis_b = dis_a.expand(len(dis_a), -1)
        dis_a = dis_b.T
        dis_a_minus_b = dis_a - dis_b
        mask = (dis_a_minus_b > 0) != obj_obj_dis
        obj_obj_dis_err = torch.abs(dis_a - dis_b)
        obj_obj_dis_err *= self.label2weight('obj_obj_dis', mask, relation['obj_obj_dis_score'])
        loss['obj_obj_dis'] = obj_obj_dis_err * obj_obj_err_mask

        # obj_obj_rot error
        obj_obj_rot = relation['obj_obj_rot']
        ori_a = out_bdb3d['ori']
        ori_b = ori_a.expand(len(ori_a), -1)
        ori_a = ori_b.T
        obj_obj_rot_err = torch.abs(torch.remainder(ori_a + obj_obj_rot - ori_b, np.pi * 2))
        obj_obj_rot_err[obj_obj_rot_err > np.pi] = 2 * np.pi - obj_obj_rot_err[obj_obj_rot_err > np.pi]
        obj_obj_rot_err *= self.label2weight(key='obj_obj_rot', score=relation['obj_obj_rot_score'])
        loss['obj_obj_rot'] = obj_obj_rot_err * obj_obj_err_mask

        # obj_wall_rot error
        obj_wall_rot = relation['obj_wall_rot']
        ori_wall = est_bdb3d_wall['ori']
        ori_obj = out_bdb3d['ori']
        ori_obj = ori_obj.expand(len(ori_wall), -1).T
        ori_wall = ori_wall.expand(len(ori_obj), -1)
        obj_wall_rot_err = torch.abs(torch.remainder(ori_obj + obj_wall_rot - ori_wall, np.pi * 2))
        obj_wall_rot_err[obj_wall_rot_err > np.pi] = 2 * np.pi - obj_wall_rot_err[obj_wall_rot_err > np.pi]
        if self.score_weighted:
            obj_wall_rot_err *= self.label2weight(key='obj_wall_rot', score=relation['obj_wall_rot_score'])
        loss['obj_wall_rot'] = obj_wall_rot_err

        return loss

    @staticmethod
    def randomize_scene(scene):
        # add random noise to bdb3d
        for obj in scene['objs']:
            bdb3d = obj['bdb3d']
            if 'ori' not in obj['bdb3d']:
                bdb3d.update(IGTransform(scene).world2campix(bdb3d))
            center = bdb3d['center']
            bdb3d['center'] = np.clip(
                center + (np.random.rand(2) - 0.5) * 20,
                0, (scene['camera']['width'], scene['camera']['height'])
            )
            bdb3d['size'] = bdb3d['size'] * ((np.random.rand(3) - 0.5) * 0.25 + 1)
            bdb3d['dis'] = bdb3d['dis'] * ((np.random.rand() - 0.5) * 0.5 + 1)
            bdb3d['ori'] = np.mod(bdb3d['ori'] + (np.random.rand() - 0.5) * 2, 2 * np.pi)
            obj['bdb3d'].update(scene.transform.campix2world(bdb3d))

    def label2weight(self, key=None, mask=None, score=None):
        weight = 1.
        if mask is not None:
            weight = weight * mask.type(torch.float32)
        if mask is None or (score is not None and self.score_weighted):
            weight = weight * torch.relu(score - self.score_thres[key]) / (1 - self.score_thres[key])
        return weight

    def visual_fcn(self, optim_bdb3d, est_data, loss, step, total):
        # TODO: transform to IGScene and use generate_relation with expand_dis=-0.1
        loss = ', '.join([f"{k}: {torch.mean(l):.4f}" for k, l in loss.items()])
        print(f"({step}) {loss}")

        every = total // self.visual_frames
        end = step == total - 1
        if step == 0:
            self.gif_io.clear()

        if (step % every == every - 1) or end:
            optim_data = est_data.copy()
            optim_data['objs'] = optim_data['objs'].copy()
            n_objs = len(optim_bdb3d['size'])
            optim_bdb3d_3d = IGTransform(est_data).campix2world(optim_bdb3d)
            optim_data['objs']['bdb3d'] = optim_bdb3d_3d

            all_bdb3d = {k: torch.cat([optim_bdb3d_3d[k], est_data['walls']['bdb3d'][k]])
                         for k in optim_bdb3d_3d.keys()}
            has_collision, _, _ = test_bdb3ds(all_bdb3d, toleration_dis=0.1)
            has_collision = has_collision.type(torch.bool)
            relation = optim_data['relation'][0]
            relation['obj_obj_col'] = has_collision[:n_objs, :n_objs]
            relation['obj_wall_col'] = has_collision[:n_objs, n_objs:]

            layout = optim_data['layout']['manhattan_world'][0]
            layout_z = layout[:, -1]
            bdb3d_expanded = expand_bdb3d(optim_bdb3d_3d, -0.1)
            corners = bdb3d_corners(bdb3d_expanded)
            optim_data['objs']['floor_col'] = (corners[..., -1] < layout_z.min()).any(-1)
            optim_data['objs']['ceil_col'] = (corners[..., -1] > layout_z.max()).any(-1)

            out_scene = IGScene.from_batch(optim_data)[0]
            image = visualize_relation(out_scene, self.visual_background, show=end, collision=True, layout=True)
            save_image(image, os.path.join(self.visual_path, f'frame_{len(self.gif_io.frames)}.png'))
            self.gif_io.append(image)

        if end:
            self.gif_io.save(os.path.join(self.visual_path, 'optim.gif'))

    def optimize(self, data, steps=100, visual=False, lr=1., momentum=0.9):
        # initialize optimization
        objs = data['objs']
        bdb3d = objs['bdb3d']
        if 'ori' not in bdb3d:
            bdb3d.update(IGTransform(data).world2campix(bdb3d))
        optim_bdb3d = {k: bdb3d[k].detach().clone() for k in ('center', 'dis', 'ori', 'size')}
        for v in optim_bdb3d.values():
            v.requires_grad = True

        # optimization settings
        optimizer = torch.optim.SGD(list(optim_bdb3d.values()), lr=lr, momentum=momentum)
        # optimizer = torch.optim.Adam(list(optim_bdb3d.values()), lr=lr, betas=(0.9, 0.999))
        # optimizer = torch.optim.LBFGS(list(optim_bdb3d.values()), lr=lr, max_iter=20)
        criterion = lambda e: nn.SmoothL1Loss(reduction='mean')(e, torch.zeros_like(e, device=e.device))

        # transform for bdb3d projection error
        if self.weights.get('bdb3d_proj'):
            scene = IGScene.from_batch(data)[0]
            transforms_to_bfov = IGTransform()
            transforms_to_bfov.camera = []
            for obj in scene['objs']:
                transform = scene.transform.copy()
                transform.data = None
                center_rad = np.array([obj['bfov']['lon'], obj['bfov']['lat']])
                center_world = transform.camrad2world(center_rad, 1)
                transform.look_at(center_world)
                transforms_to_bfov.camera.append(transform.camera)
            transforms_to_bfov.camera = list_of_dict_to_dict_of_array(transforms_to_bfov.camera, to_tensor=True)
            transforms_to_bfov.camera = recursively_to(transforms_to_bfov.camera, device=bdb3d['size'].device)
        else:
            transforms_to_bfov = None

        # optimization step
        loss = {}

        def closure():
            optimizer.zero_grad()
            loss.update(self.relation_loss(optim_bdb3d, data, transforms_to_bfov))
            loss['total'] = sum([
                self.weights[k] * criterion(loss[k]) if len(loss[k]) > 0
                else 0. for k in self.weights.keys() if k in loss
            ])
            if loss['total'] > 0:
                loss['total'].backward(retain_graph=True)
                if optim_bdb3d['center'].grad is not None:
                    optim_bdb3d['center'].grad *= 1000 # center in pix is around 10e2, need more grad
            return loss['total']

        # run optimization
        with torch.enable_grad():
            for step in range(steps):
                optimizer.step(closure)

                if visual:
                    self.visual_fcn(optim_bdb3d, data, loss, step, steps)

        for v in optim_bdb3d.values():
            v.requires_grad = False

        return optim_bdb3d


def compare_bdb3d(est_bdb3d, gt_bdb3d, info):
    print(info + ', '.join([f"{k}: {torch.abs(est_bdb3d[k] - gt_bdb3d[k]).mean():.4f}"
                                   for k in est_bdb3d.keys()]))


