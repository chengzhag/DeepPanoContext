from models.training import BaseTrainer


class Trainer(BaseTrainer):
    '''
    Trainer object for pano3d.
    '''
    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''computer losses'''
        loss = self.net.loss(est_data, data)
        return loss
