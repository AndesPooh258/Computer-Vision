import torch
import torch.nn as nn
import torch.nn.functional as F

def EPE(output, target):
    # return torch.linalg.norm(output-target, dim=1).mean()
    return (((output - target) ** 2).sum(dim=1) ** 0.5).mean()

class OursLoss(nn.Module):
    def __init__(self, args):
        super(OursLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Ours']
        
        # reference: https://github.com/NVIDIA/flownet2-pytorch/blob/master/losses.py
        self.w_init = 0.32
        self.w_decay = 0.5

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            epevalue += self.w_init * (self.w_decay ** i) * EPE(output_, target_)
        return [epevalue]
