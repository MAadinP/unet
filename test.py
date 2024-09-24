# Image Classification
import torch
from torchvision.transforms import v2
import cv2
import numpy as np

H, W = 32, 32
img = torch.randint(low=0, high=256, size=(3, H, W), dtype=torch.uint8)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img)
numpy_image = img.numpy()

# Convert the numpy array to a cv2 image
cv2_image = np.transpose(numpy_image, (1, 2, 0))
cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

# Display the image using cv2
cv2.imshow("Image", cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def generate_cool_image(height, width, ):
    pass