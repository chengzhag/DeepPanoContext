from models.training import BaseTrainer

class Trainer(BaseTrainer):

    def eval_step(self, data):
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def compute_loss(self, data):
        data = self.to_device(data)
        est_data = self.net(data)
        loss = self.net.loss(est_data, data)
        return loss
