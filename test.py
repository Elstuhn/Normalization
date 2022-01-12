import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from calculate import *

catimg = Image.open("path")
plt.imshow(catimg)
plt.show()

plt.hist(np.array(catimg).ravel(), bins=50, density=True);
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.show()

transform1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
cat = transform1(catimg)
print(cat.shape)
cat = np.array(cat)
cat = np.expand_dims(cat, 0)
cat = torch.tensor(cat)
print(cat.shape)
mean, std = calcMStdtorch(cat)
print(mean, std)
#transform2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean, std= std)])
#normalized_img = transform2(cat)
normalized_img = normalizeT(cat, mean, std)
print(normalized_img.shape)
plt.hist(normalized_img.numpy().ravel(), bins=30, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.show()
