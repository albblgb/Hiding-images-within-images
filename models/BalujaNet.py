import torch
import torch.nn as  nn
import torch.nn.functional as F
from utils.image import quantization

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        self.pre_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=30, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=7, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.hid_net = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=30, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=3, kernel_size=2, stride=1),
            nn.Tanh()
        )

    def forward(self, secret, cover):
        f_secret = self.pre_net(secret)
        f_secret = F.pad(f_secret, (0, 1, 0, 1))
        input_of_HidNet = torch.cat((cover, f_secret), dim=1)
        stego = self.hid_net(input_of_HidNet)
        return stego



class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.dec_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=100, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=50, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=3, kernel_size=2, stride=1),
            nn.Tanh(),
        )

    def forward(self, stego):
        secret_rev = self.dec_net(stego)
        return secret_rev


class balujanet(nn.Module):
    def __init__(self) -> None:
        super(balujanet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, secret, cover, mode='train'):
        stego = self.encoder(secret, cover)
        if mode == 'test':
            stego = quantization(stego)
        secret_rev = self.decoder(stego)
        if mode == 'test':
            secret_rev = quantization(secret_rev)
        return stego, secret_rev




