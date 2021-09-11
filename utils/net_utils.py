
#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)

