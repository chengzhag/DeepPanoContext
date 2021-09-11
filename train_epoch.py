# Training functions.
# author: ynie
# date: Feb, 2020

from utils.project_utils import ETA
from models.eval_metrics import MetricRecorder
from torch.optim import lr_scheduler
import wandb

def train_epoch(cfg, epoch, trainer, dataloaders, step):
    '''
    train by epoch
    :param cfg: configuration file
    :param epoch: epoch id.
    :param trainer: specific trainer for networks
    :param dataloaders: dataloader for training and validation
    :return:
    '''
    for phase in ['train', 'val']:
        dataloader = dataloaders[phase]
        loss_recorder = MetricRecorder()
        trainer.net.train(phase == 'train')
        cfg.log_string('-' * 100)
        cfg.log_string('Switch Phase to %s.' % (phase))
        cfg.log_string('-' * 100)
        eta_calc = ETA(smooth=0.99, ignore_first=True)
        for iter, data in enumerate(dataloader):
            if phase == 'train':
                loss = trainer.train_step(data)
            else:
                loss = trainer.eval_step(data)

            loss_recorder.add(loss)

            eta = eta_calc(len(dataloader) - iter - 1)
            if ((iter + 1) % cfg.config['log']['print_step']) == 0:
                loss_str = ', '.join([f'{k}: {v:.3f}' for k, v in loss.items()])
                cfg.log_string(f"Phase: {phase}. "
                               f"Epoch {epoch}: "
                               f"{iter + 1}/{len(dataloader)}. "
                               f"ETA: {eta}. "
                               f"Current loss: {{{loss_str}}}.")
                wandb.summary['ETA_stage'] = str(eta)
                if phase == 'train':
                    loss = {f'train_{k}': v for k, v in loss.items()}
                    wandb.log(loss, step=step)
                    wandb.log({'epoch': epoch}, step=step)

            if phase == 'train':
                step += 1

        cfg.log_string('=' * 100)
        for loss_name, loss_value in loss_recorder.items():
            cfg.log_string(f"Currently the last {phase} loss ({loss_name}) is: {loss_value()}")
        cfg.log_string('=' * 100)

    return loss_recorder, step

def train(cfg, trainer, scheduler, checkpoint, train_loader, val_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param scheduler: scheduler for optimizer
    :param trainer: specific trainer for networks
    :param checkpoint: network weights.
    :param train_loader: dataloader for training
    :param val_loader: dataloader for validation
    :return:
    '''
    start_epoch = scheduler.last_epoch
    if isinstance(scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
        start_epoch -= 1
    total_epochs = cfg.config['train']['epochs']
    min_eval_loss = checkpoint.get('min_loss')
    step = checkpoint.get('step')

    dataloaders = {'train': train_loader, 'val': val_loader}

    eta_calc = ETA(smooth=0)
    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('-' * 100)
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        trainer.show_lr()

        eval_loss_recorder, step = train_epoch(cfg, epoch + 1, trainer, dataloaders, step)

        total_eval_loss = eval_loss_recorder['total']()
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_eval_loss)
        elif isinstance(scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
            scheduler.step()
        else:
            raise NotImplementedError
        eval_loss = {f'test_{k}': v() for k, v in eval_loss_recorder.items()}
        wandb.log(eval_loss, step=step)
        wandb.log({f'lr{i}': g['lr'] for i, g in enumerate(trainer.optimizer.param_groups)}, step=step)
        wandb.log({'epoch': epoch + 1}, step=step)

        eta = eta_calc(total_epochs - epoch - 1)
        cfg.log_string('Epoch (%d/%s) ETA: (%s).' % (epoch + 1, total_epochs, eta))
        wandb.summary['ETA'] = str(eta)

        # save checkpoint
        checkpoint.register_modules(epoch=epoch, min_loss=min_eval_loss, step=step)
        if cfg.config['log'].get('save_checkpoint', True):
            checkpoint.save('last')
        cfg.log_string('Saved the latest checkpoint.')
        if epoch==-1 or total_eval_loss<min_eval_loss:
            if cfg.config['log'].get('save_checkpoint', True):
                checkpoint.save('best')
            min_eval_loss = total_eval_loss
            cfg.log_string('Saved the best checkpoint.')
            cfg.log_string('=' * 100)
            for loss_name, loss_value in eval_loss_recorder.items():
                wandb.summary[f'best_test_{loss_name}'] = loss_value()
                cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value()))
            cfg.log_string('=' * 100)