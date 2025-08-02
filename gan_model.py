import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# --- Configuration - CHANGE THIS VALUE ---
# INPUT_IMAGE_PATH = 'vada_pics/1.jpg'
# MODEL_PATH = 'models/35_net_G.pth'
# OUTPUT_IMAGE_PATH = 'result.png'
# ----------------------------------------

# ----------------------------------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------------------------------
class ResnetGenerator(nn.Module):
    # --- FIX: Changed default norm_layer from BatchNorm2d to InstanceNorm2d ---
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, n_blocks_use_dropout=False, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # If InstanceNorm, use bias. If BatchNorm, don't.
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=n_blocks_use_dropout, use_bias=use_bias)]

        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(int(ngf * mult / 2), ngf,
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
# ----------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# MAIN SCRIPT LOGIC
# ----------------------------------------------------------------------------------
# def run_inference(input_path, output_path, model_path):
#     model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
#     print(f"Loading model from {model_path}...")
#     state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict)
#     model.eval()
#     print("Model loaded successfully.")

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     try:
#         input_image = Image.open(input_path).convert('RGB')
#     except FileNotFoundError:
#         print(f"Error: Input image not found at '{input_path}'")
#         return

#     input_tensor = transform(input_image).unsqueeze(0)
#     with torch.no_grad():
#         output_tensor = model(input_tensor)

#     output_image_data = (output_tensor.squeeze().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0
#     output_image = Image.fromarray(output_image_data.astype('uint8'))

#     output_dir = os.path.dirname(output_path)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     output_image.save(output_path)
#     print(f"Successfully converted image and saved result to '{output_path}'")
# # ----------------------------------------------------------------------------------


# --- RUN THE SCRIPT ---
# run_inference(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, MODEL_PATH)