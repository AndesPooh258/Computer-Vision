import torch.nn as nn
import torch.nn.functional as F

def EPE(output, target):
    return (((output - target) ** 2).sum(dim=1) ** 0.5).mean()

class EPELoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(EPELoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['EPE']

    def forward(self, output, target):
        target = self.div_flow * target
        assert output.shape == target.shape, (output.shape, target.shape)
        epevalue = EPE(output, target)
        return [epevalue]

class MultiscaleLoss(nn.Module):
    def __init__(self, args):
        super(MultiscaleLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Multiscale']
        self.w_init = 1
        self.w_decay = 0.5

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            epevalue += self.w_init * (self.w_decay ** i) * EPE(output_, target_)
        return [epevalue]
        