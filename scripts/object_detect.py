import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv


def resize(img_path):
    img = cv.imread(img_path)
    res = cv.resize(img, dsize=(640, 800), interpolation=cv.INTER_CUBIC)
    res = cv.cvtColor(res, cv.COLOR_RGB2BGR)
    return res

def setup_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def camera_detect(img, model):
    results = model(img)
    res = results.pandas().xyxy[0]

    #print(res)
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image.show()
    img_cropped = image.crop((res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0]))
    #img_cropped.show()
    return img_cropped

def detect(img, model):
    source_dir = Path(__file__).resolve().parent.parent
    img = f"{source_dir}/{img}"

    results = model(img)
    res = results.pandas().xyxy[0]

    #print(res)
    image = Image.fromarray(resize(img).astype('uint8'), 'RGB')
    #img_cropped = image.crop((res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0]))
    #image.show()
    return image


def main():
    test_img = "test_image.jpg"
    img = detect(test_img)
    img.show()


if __name__ == "__main__":
    main()