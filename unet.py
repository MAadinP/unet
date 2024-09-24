import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d, ReLU, Sigmoid
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import cv2

image = Image.open("./imgs/bench1.jpg")
new_image = image.resize((512, 512))
print(f"{classmethod(new_image)} is the class of the image")

make_tensor = transforms.PILToTensor()

image_effects = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # v2.GaussianBlur(kernel_size=5, sigma=(98, 100))
])

img_tensor = make_tensor(new_image) # Image => Tensor
print(f"{classmethod(img_tensor)} is the class of the image")

img_tensor = img_tensor.float()
img_tensor.size()
# Conv2d needs floats rather than Bytes (unit8) that is found in the tensor when you do PILToTensor

#Another fix needed is that torch.Size([3, 512, 512]) is the outout of the tensor but we need a batch dim in there
# It should look like [1, 3, 512, 512] and the tensor should contain type: Constant

img_tensor = img_tensor.unsqueeze(0)
img_tensor.size()

# img_tensor = image_effects(img_tensor) # Apply effects to the Tensor version of the original image

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_chaine1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=64, padding="same", kernel_size=3),
            ReLU(inplace=True),
            # First is we want to go from 256x256x3 and reduce H and W by 2 as well as deepen the channel
            Conv2d(in_channels=64, out_channels=64, padding="same", kernel_size=3),
            ReLU(inplace=True),
            # The second one only reduces H and W by 2 no deepening of the channel
            # Inplace means that in memory the RELU is applied to the tensor without using more memory to store the output
        )
        self.conv_chaine2 = nn.Sequential(
            Conv2d(in_channels=64, out_channels=128, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, padding="same", kernel_size=3),
            ReLU(inplace=True),
        )
        self.conv_chaine3 = nn.Sequential(
            Conv2d(in_channels=128, out_channels=256, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, padding="same", kernel_size=3),
            ReLU(inplace=True),
        )
        self.conv_chaine4 = nn.Sequential(
            Conv2d(in_channels=256, out_channels=512, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, padding="same", kernel_size=3),
            ReLU(inplace=True),
        )
        self.bottom = nn.Sequential(
            Conv2d(in_channels=512, out_channels=1024, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=1024, out_channels=1024, padding="same", kernel_size=3),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        )
        self.conv_chaind1 = nn.Sequential(
            Conv2d(in_channels=1024, out_channels=512, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, padding="same", kernel_size=3),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        )
        self.conv_chaind2 = nn.Sequential(
            Conv2d(in_channels=512, out_channels=256, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, padding="same", kernel_size=3),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        )
        self.conv_chaind3 = nn.Sequential(
            Conv2d(in_channels=256, out_channels=128, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, padding="same", kernel_size=3),
            ReLU(inplace=True),
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        )
        self.conv_chaind4 = nn.Sequential(
            Conv2d(in_channels=128, out_channels=64, padding="same", kernel_size=3),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, padding="same", kernel_size=3),
            ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            Conv2d(kernel_size=1, padding="same", in_channels=64, out_channels=3),
            Sigmoid()
        )
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)


    def forward_pass(self, input_tensor):

        print(f"\nInput: {input_tensor.size()}")
        c1 = self.conv_chaine1(input_tensor)
        mc1 = self.max_pool(c1)
        print(f"\nInput c: {c1.size()} \n\tInput mc: {mc1.size()}")
        
        c2 = self.conv_chaine2(mc1)
        mc2 = self.max_pool(c2)
        print(f"\nInput c: {c2.size()} \n\tInput mc: {mc2.size()}")

        c3 = self.conv_chaine3(mc2)
        mc3 = self.max_pool(c3)
        print(f"\nInput c: {c3.size()} \n\tInput mc: {mc3.size()}")

        c4 = self.conv_chaine4(mc3)
        mc4 = self.max_pool(c4)
        print(f"\nInput c: {c4.size()} \n\tInput mc: {mc4.size()}")
        
        b = self.bottom(mc4)
        print(f"\nbottleneck: {b.size()}")
        
        cat1 = torch.cat(tensors=[b, c4], dim=1)
        print(cat1.size())
        
        d1 = self.conv_chaind1(cat1)
        cat2 = torch.cat(tensors=[d1, c3], dim=1)

        d2 = self.conv_chaind2(cat2)
        cat3 = torch.cat(tensors=[d2, c2], dim=1)

        d3 = self.conv_chaind3(cat3)
        cat4 = torch.cat(tensors=[d3, c1], dim=1)

        d4 = self.conv_chaind4(cat4)
        output_tensor = self.output(d4)

        return output_tensor

model = Unet()
out = model.forward_pass(img_tensor)
print(f"{out} \nOutput dim: {out.size()}")