import torch
import torch.nn as nn

'''
- Number of FLOPs: 2048.66M
- total_params: 3.94M
- Inference Averages EPE: 6.216
'''
class FlowNetEncoder(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20):
        super(FlowNetEncoder,self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, padding=1)

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        
        # basic FlowNet encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # predict and upscale
        x = self.conv5(x)
        flow4 = nn.functional.interpolate(x, scale_factor=16)

        if self.training:
            return flow4
        else:
            return flow4 * self.div_flow