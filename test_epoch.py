# Testing functions.
# author: ynie
# date: April, 2020
from utils.project_utils import ETA
from models.eval_metrics import ClassMeanMeter, MetricRecorder, AverageMeter
from time import time
import torch
import wandb


def test_func(cfg, tester, test_loader):
    '''
    test function.
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    metric_recorder = MetricRecorder()
    cfg.log_string('-' * 100)
    eta_calc = ETA(smooth=0.99, ignore_first=True)
    for iter, data in enumerate(test_loader):
        loss, est_data = tester.test_step(data)

        # visualize intermediate results.
        vis_step = cfg.config['log'].get('vis_step')
        if vis_step and ((iter + 1) % vis_step) == 0:
            tester.visualize_step(est_data)

        metric_recorder.add(loss)

        eta = eta_calc(len(test_loader) - iter - 1)
        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            loss_str = ', '.join([f"{k}: {v}" for k, v in loss.items()])
            cfg.log_string(f"Phase: {cfg.config['mode']}. "
                           f"{iter + 1}/{len(test_loader)}. "
                           f"ETA: {eta}. "
                           f"Current loss: {{{loss_str}}}.")
            wandb.summary['ETA'] = str(eta)

    return metric_recorder

def test(cfg, tester, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    cfg.log_string('-' * 100)
    # set mode
    tester.net.train(cfg.config['mode'] == 'train')
    start = time()
    with torch.no_grad():
        metric_recorder = test_func(cfg, tester, test_loader)
    cfg.log_string('Test time elapsed: (%f).' % (time()-start))
    cfg.log_string('\n' + str(metric_recorder))
    metric_recorder.log()
