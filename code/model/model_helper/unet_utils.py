import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderPart(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        return x4


class Maxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.maxpool(x)


class Upconvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2) #input 28x28 out 56x56 || put 28 in above eqn

    def forward(self, x):
        return self.upconv(x)


class DecoderPart(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_block = EncoderPart(input_channels, output_channels)

    def forward(self, x):
        return self.conv_block(x)


class OutputLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_layer(x)


def concat(tensor_1, tensor_2):
    '''
    concatenate tensor_2 to tensor_1 
    '''
    dim_1 = tensor_1.shape[2]
    dim_2 = tensor_2.shape[2]

    part_to_remove = int((dim_2 - dim_1)/2)

    cropped_tensor_2 = tensor_2[:, :, part_to_remove-1:(part_to_remove+dim_1-1), part_to_remove-1:(part_to_remove+dim_1-1)]

    after_concat = torch.cat((tensor_1, cropped_tensor_2), dim=1) # dim=1 means concat along the channels 

    return after_concat

