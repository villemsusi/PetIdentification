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

    images = []

    image = Image.open(img)

    if len(res.xmin) > 0:
        for ind, i in enumerate(res.name):
            if i == "cat" or i == "dog":
                crop_img = image.crop((res.xmin[ind], res.ymin[ind], res.xmax[ind], res.ymax[ind]))
                images.append(crop_img)
                
    if (len(images) == 0):
        images.append(image)
    return images


def main():
    model = setup_model()
    test_img = "test_image.jpg"
    imgs = detect(test_img, model)
    for img in imgs:
        img.show()


if __name__ == "__main__":
    main()