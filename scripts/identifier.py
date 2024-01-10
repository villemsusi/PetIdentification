from object_detect import detect
from object_predict import inference



if __name__ == "__main__":
    image = detect()
    res = inference("2023-12-07_13-35-07\efficientnet_b0.ra_in1k.pth", image, "cat")
    print(res)