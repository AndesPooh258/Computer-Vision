import torch
import torch.nn as nn

'''
- Number of FLOPs: 2253.23M
- total_params: 4.03M
- Inference Averages Ours: 5.394
'''
class FlowNetOurs(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20):
        super(FlowNetOurs, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=4, dilation=4),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, padding=1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv7 = nn.Conv2d(in_channels=514, out_channels=2, kernel_size=3, padding=1)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=514, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv8 = nn.Conv2d(in_channels=514, out_channels=2, kernel_size=3, padding=1)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=514, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv9 = nn.Conv2d(in_channels=258, out_channels=2, kernel_size=3, padding=1)

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        # basic FlowNet encoder with one additional block and dilated convolution
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # upscale with deconvolution and concatenate with shallow-level features
        # keep predictions at different scale
        flow5 = self.conv6(x5)
        x6 = torch.cat((x4, self.deconv1(x5), nn.functional.interpolate(flow5, scale_factor=2)), dim=1)
        flow4 = self.conv7(x6)
        x7 = torch.cat((x3, self.deconv2(x6), nn.functional.interpolate(flow4, scale_factor=2)), dim=1)
        flow3 = self.conv8(x7)
        x8 = torch.cat((x2, self.deconv3(x7), nn.functional.interpolate(flow3, scale_factor=2)), dim=1)
        x9 = self.conv9(x8)

        # predict and upscale
        flow2 = nn.functional.interpolate(x9, scale_factor=4)

        if self.training:
            return flow2, flow3, flow4, flow5
        else:
            return flow2 * self.div_flow