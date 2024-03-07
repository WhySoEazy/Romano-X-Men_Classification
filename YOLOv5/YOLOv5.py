import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
import pathlib
import cv2
import os
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225

model = torch.hub.load("ultralytics/yolov5", "custom", 
                       path = "best.pt",
                       force_reload=True)

def classify_transforms(size=224):
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

imgs = "images.jpg"
image = Image.open(imgs)
transformations = classify_transforms()
convert_tensor = transformations(image)
convert_tensor = convert_tensor.unsqueeze(0)
convert_tensor = convert_tensor.to(device)

results = model(convert_tensor)
print(results)
pred = F.softmax(results, dim=1)

for i, prob in enumerate(pred):
    top5i = prob.argsort(0, descending=True)[:5].tolist()
    text = '\n'.join(f'{prob[j]:.2f} {model.names[j]}' for j in top5i)
    print(text)

img = Image.open('images.jpg')
I1 = ImageDraw.Draw(img)
myFont = ImageFont.truetype('arial.ttf' , 75)
I1.text((10, 10), text , font = myFont , fill=(255, 255 , 255))
img.show()