import torch
from PIL import Image
from pathlib import Path


def detect(img_file):
    source_dir = Path(__file__).resolve().parent.parent

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    img = f"{source_dir}/{img_file}"

    results = model(img)
    res = results.pandas().xyxy[0]

    #print(res)

    im1 = Image.open(img, mode="r")
    im2 = im1.crop((res.xmin[0], res.ymin[0], res.xmax[0], res.ymax[0]))
    return im2

def main():
    test_img = "test_image.jpg"
    img = detect(test_img)
    img.show()

if __name__ == "__main__":
    main()