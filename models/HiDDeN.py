import torch
import torch.nn as nn
import config as c
from utils.image import quantization

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.H = 128
        self.W = 128
        self.conv_channels = 64
        self.num_blocks = 4

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(self.num_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + 3 * 11, # The secret image is copied 11 times
                                             self.conv_channels)
        
        self.added_layers = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.conv_channels, 3, 1, 1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv_channels, self.conv_channels, 3, 1, 1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv_channels, self.conv_channels, 3, 1, 1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv_channels, self.conv_channels, 3, 1, 1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        # self.sigmoid = nn.sigmoid()
        

    def forward(self, cover, secret):

        expanded_secret = secret.repeat(1, 11, 1, 1)
        cover_features = self.conv_layers(cover)
        # concatenate expanded secret and cover
        concat = torch.cat([expanded_secret, cover_features, cover], dim=1)
        stego = self.final_layer(self.added_layers(self.after_concat_layer(concat)))
        return stego


class Decoder(nn.Module):
    """
    To hide images, we slightly tuned the output of the Decoder
    """
    def __init__(self):

        super(Decoder, self).__init__()
        self.channels = 64
        self.decoder_blocks = 8
        self.out_channels = 3

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(self.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(nn.Conv2d(self.channels, self.out_channels, 3, 1, padding=1))
        # layers.append(ConvBNRelu(self.channels, self.out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, stego):
        secret_rev = self.layers(stego)
        return secret_rev
    

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = 64
        self.num_blocks = 3

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(self.num_blocks-1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(self.channels, 1)


    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X
    

class EncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        if c.mode == 'test':
            stego = quantization(stego)
        secret_rev = self.decoder(stego)
        if c.mode == 'test':
            secret_rev = quantization(secret_rev)
        return stego, secret_rev
