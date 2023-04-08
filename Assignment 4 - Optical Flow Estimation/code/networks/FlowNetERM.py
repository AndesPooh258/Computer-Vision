import torch
import torch.nn as nn

'''
- Number of FLOPs: 2070.63M
- total_params: 3.95M
- Inference Averages Multiscale: 5.856
'''
class FlowNetEncoderRefineMultiscale(nn.Module):
    def __init__(self, args, input_channels = 6, batchNorm=True, div_flow=20):
        super(FlowNetEncoderRefineMultiscale, self).__init__()
        
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
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Conv2d(in_channels=514, out_channels=2, kernel_size=3, padding=1)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=514, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv7 = nn.Conv2d(in_channels=258, out_channels=2, kernel_size=3, padding=1)

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        
        # basic FlowNet encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # upscale with deconvolution and concatenate with shallow-level features
        # keep predictions at different scale
        flow4 = self.conv5(x4)
        x5 = torch.cat((x3, self.deconv1(x4), nn.functional.interpolate(flow4, scale_factor=2)), dim=1)
        flow3 = self.conv6(x5)
        x6 = torch.cat((x2, self.deconv2(x5), nn.functional.interpolate(flow3, scale_factor=2)), dim=1)

        # predict and upscale
        x7 = self.conv7(x6)
        flow2 = nn.functional.interpolate(x7, scale_factor=4)

        if self.training:
            return flow2, flow3, flow4
        else:
            return flow2 * self.div_flow
