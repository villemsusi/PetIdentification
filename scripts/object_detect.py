import torch
from PIL import Image
from pathlib import Path


def detect():
    source_dir = Path(__file__).resolve().parent.parent

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    img = f"{source_dir}/test_image.jpg"

    results = model(img)
    res = results.pandas().xyxy[0]

    print(res)

    im1 = Image.open(img, mode="r")
    im2 = im1.crop((res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0]))
    return im1

def main():
    detect()

if __name__ == "__main__":
    main()