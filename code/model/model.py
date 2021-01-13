import time
from model_helper.unet_utils import *

class UNet(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

        self.encoder_layer1 = EncoderPart(1,64)
        self.encoder_layer2 = EncoderPart(64,128)
        self.encoder_layer3 = EncoderPart(128,256)
        self.encoder_layer4 = EncoderPart(256,512)
        self.encoder_layer5 = EncoderPart(512,1024)

        self.maxpool = Maxpool()

        self.upconvolution1 = Upconvolution(1024,512)
        self.decoder_layer1 = DecoderPart(1024,512)
        self.upconvolution2 = Upconvolution(512,256)
        self.decoder_layer2 = DecoderPart(512,256)
        self.upconvolution3 = Upconvolution(256,128)
        self.decoder_layer3 = DecoderPart(256,128)
        self.upconvolution4 = Upconvolution(128,64)
        self.decoder_layer4 = DecoderPart(128,64)

        self.output_layer = OutputLayer(64,2)

    def forward(self, x):
        encoder_out_layer_1 = self.encoder_layer1(x)
        encoder_maxpool_out_layer_1 = self.maxpool(encoder_out_layer_1)
        if self.debug:
            print("[DEBUG] Shape: ",encoder_maxpool_out_layer_1.shape)
        encoder_out_layer_2 = self.encoder_layer2(encoder_maxpool_out_layer_1)
        encoder_maxpool_out_layer_2 = self.maxpool(encoder_out_layer_2)
        if self.debug:
            print("[DEBUG] Shape: ",encoder_maxpool_out_layer_2.shape)
        encoder_out_layer_3 = self.encoder_layer3(encoder_maxpool_out_layer_2)
        encoder_maxpool_out_layer_3 = self.maxpool(encoder_out_layer_3)
        if self.debug:
            print("[DEBUG] Shape: ",encoder_maxpool_out_layer_3.shape)
        encoder_out_layer_4 = self.encoder_layer4(encoder_maxpool_out_layer_3)
        encoder_maxpool_out_layer_4 = self.maxpool(encoder_out_layer_4)
        if self.debug:
            print("[DEBUG] Shape: ",encoder_maxpool_out_layer_4.shape)
        encoder_out_layer_5 = self.encoder_layer5(encoder_maxpool_out_layer_4)
        if self.debug:
            print("[DEBUG] Shape: ",encoder_out_layer_5.shape)

        upconvolution_out_layer_1 = self.upconvolution1(encoder_out_layer_5)
        concat_out_1 = concat(upconvolution_out_layer_1, encoder_out_layer_4)
        decoder_out_layer_1 = self.decoder_layer1(concat_out_1)
        if self.debug:
            print("[DEBUG] Shape: ",decoder_out_layer_1.shape)

        upconvolution_out_layer_2 = self.upconvolution2(decoder_out_layer_1)
        concat_out_2 = concat(upconvolution_out_layer_2, encoder_out_layer_3)
        decoder_out_layer_2 = self.decoder_layer2(concat_out_2)
        if self.debug:
            print("[DEBUG] Shape: ",decoder_out_layer_2.shape)

        upconvolution_out_layer_3 = self.upconvolution3(decoder_out_layer_2)
        concat_out_3 = concat(upconvolution_out_layer_3, encoder_out_layer_2)
        decoder_out_layer_3 = self.decoder_layer3(concat_out_3)
        if self.debug:
            print("[DEBUG] Shape: ",decoder_out_layer_3.shape)

        upconvolution_out_layer_4 = self.upconvolution4(decoder_out_layer_3)
        concat_out_4 = concat(upconvolution_out_layer_4, encoder_out_layer_1)
        decoder_out_layer_4 = self.decoder_layer4(concat_out_4)
        if self.debug:
            print("[DEBUG] Shape: ",decoder_out_layer_4.shape)

        output = self.output_layer(decoder_out_layer_4)

        return output

model = UNet(debug=True)

x = torch.randn(1,1,572,572)
prev_time = time.time()
print("[DEBUG] Output shape: ",model(x).shape)
print("\n[DEBUG] FPS: ", 1/(time.time()-prev_time))