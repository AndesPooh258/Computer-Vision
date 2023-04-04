import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-12, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        # add a small constant to avoid log(0)
        self.eps = eps

    def forward(self, x, y, **kwargs):
        # direct implementation by considering the definition of cross-entropy loss
        return -torch.mean(torch.gather(input=torch.log(x + self.eps), dim=1, index=torch.unsqueeze(y, 1)).squeeze(-1))
