import torch.nn as nn

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        # nn.AdaptiveAvgPool1d automatically set the kernel and stride for the desired output size
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv1d(
                in_channels=channel,
                out_channels=channel // reduction,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=channel // reduction,
                out_channels=channel,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(RCAB, self).__init__()
        modules_body = [
            conv(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        ]
        modules_body.append(CALayer(channel=n_feat, reduction=reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv=conv, n_feat=n_feat, kernel_size=kernel_size, reduction=reduction)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(in_channels=n_feat, out_channels=n_feat, kernel_size=kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class FB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=False, act=nn.ReLU(True)):
        super(FB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res