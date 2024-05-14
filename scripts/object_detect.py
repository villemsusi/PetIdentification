import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv

def setup_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect(img, model):
    results = model(img)
    res = results.pandas().xyxy[0]
    if len(res.name) > 0:
        img_cropped = img.crop((max(0, res.xmin[0]-15), max(0, res.ymin[0]-15), res.xmax[0]+15, res.ymax[0]+15))
        return img_cropped
    return img

def file_detect(img, model):
    source_dir = Path(__file__).resolve().parent.parent
    img = f"{source_dir}/{img}"

    results = model(img)
    res = results.pandas().xyxy[0]
    image = Image.open(img)
    
    for i in range(len(res.name)):
        if res.name[i] == "cat" or res.name[i] == "dog":
            crop_img = image.crop((res.xmin[i]-10, res.ymin[i]-10, res.xmax[i]+10, res.ymax[i]+10))
                
    #return crop_img
    return image


def main():
    model = setup_model()
    test_img = "test_image.jpg"
    imgs = file_detect(test_img, model)
    for img in imgs:
        img.show()


if __name__ == "__main__":
    main()