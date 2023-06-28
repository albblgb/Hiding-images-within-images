import torch
import torch.nn as  nn
import torch.nn.functional as F
import functools
from utils.image import quantization

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels , kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Encoder, self).__init__()

        self.n_channels = in_channels
        self.out_channels = out_channels

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        
        self.up1 = Up(512, 512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 128)
        self.up6 = Up(256, 64)

        self.outlayer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels , kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # x = torch.cat((secret, cover), dim=1)
        x1 = self.down1(x)      # 64   * 64 *64
        x2 = self.down2(x1)     # 128  * 32 *32
        x3 = self.down3(x2)     # 256  * 16 *16
        x4 = self.down4(x3)     # 512  * 8  *8
        x5 = self.down5(x4)     # 512  * 4  *4     
        x6 = self.down6(x5)     # 512  * 2  *2
        x7 = self.down7(x6)     # 512  * 1  *1

        x = self.up1(x7, x6)    #1024  * 2  *2  
        x = self.up2(x, x5)     #1024  * 4  *4
        x = self.up3(x, x4)     #1024  * 8  *8
        x = self.up4(x, x3)     #512   * 16 *16
        x = self.up5(x, x2)     #256   * 32 *32
        x = self.up6(x, x1)     #128   * 64 *64
        stego = self.outlayer(x)# 3    * 128*128
        
        return stego


class Decoder(nn.Module):
    '''
    To hide single image, we slightly tuned the arch of the Decoder
    '''
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.dec_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, stego):
        secret_rev = self.dec_net(stego)
        return secret_rev



class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward(self, input):
        output=self.main(input)
        return output


class wengnet(nn.Module):
    def __init__(self) -> None:
        super(wengnet, self).__init__()

        self.encoder = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
        self.decoder = RevealNet()
    
    def forward(self, secret, cover, mode='train'):
        input_of_encoder = torch.cat((secret, cover), dim=1)
        stego = self.encoder(input_of_encoder)
        if mode == 'test':
            stego = quantization(stego)
        secret_rev = self.decoder(stego)
        if mode == 'test':
            secret_rev = quantization(secret_rev)
        return stego, secret_rev



