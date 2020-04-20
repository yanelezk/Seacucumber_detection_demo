import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock2D(nn.Module):
    def __init__(self,in_channels,reduction=2,mode='embedded_gaussian'):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1,stride=1)
        self.theta = nn.Conv2d(self.in_channels,self.inter_channels,kernel_size=1,stride=1)
        self.phi = nn.Conv2d(self.in_channels,self.inter_channels,kernel_size=1,stride=1)
        self.conv_mask = nn.Conv2d(self.inter_channels,self.in_channels,kernel_size=1,stride=1)

    def embedded_gaussian(self, x):
        batch_size = x.size(0)

        g_x = F.relu(self.g(x)).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = F.relu(self.theta(x)).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = F.relu(self.phi(x)).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = F.relu(self.conv_mask(map_))
        final = mask + x
        return final

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output = self.embedded_gaussian(x)
        else:
            raise NotImplementedError("The code has not been implemented.")
        return output

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class GCN(nn.Module):

    def __init__(self, in_planes, stride = 1, scale = 0.1, groups=8, thinning=2, k = 7, dilation=1):
        super(GCN, self).__init__()
        self.scale = scale
        second_in_planes = in_planes // thinning

        #p = math.floor((k-1)/2) 
        p =3
        out_planes = 490
        self.cfem_a = list()
        self.cfem_a += [BasicConv(in_planes, in_planes, kernel_size = (1,k), stride = 1, padding = (0,p))]
        self.cfem_a += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = (k, 1), stride = 1, padding = (p, 0))]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a = nn.ModuleList(self.cfem_a)

        self.cfem_b = list()
        self.cfem_b += [BasicConv(in_planes, in_planes, kernel_size = (k,1), stride = (1,1), padding = (p,0), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 3, stride=stride, padding=dilation,dilation=dilation)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = (1, k), stride = 1, padding = (0, p))]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b = nn.ModuleList(self.cfem_b)

        self.ConvLinear = BasicConv(2 * second_in_planes, out_planes, kernel_size = 1, stride = 1, relu = False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size = 1, stride = stride, relu = False)
        self.relu = nn.ReLU(inplace = False)

    def forward(self,x):

        x1 = self.cfem_a[0](x)
        x1 = self.cfem_a[1](x1)
        x1 = self.cfem_a[2](x1)
        x1 = self.cfem_a[3](x1)
        x1 = self.cfem_a[4](x1)

        x2 = self.cfem_b[0](x)
        x2 = self.cfem_b[1](x2)
        x2 = self.cfem_b[2](x2)
        x2 = self.cfem_b[3](x2)
        x2 = self.cfem_b[4](x2)

        out = torch.cat([x1, x2], 1)

        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        #out = self.relu(out)

        return out

