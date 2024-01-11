from object_detect import detect
from object_predict import inference



if __name__ == "__main__":
    image = detect()
    res = inference("efficientnet_b0.ra_in1k.pth", image, "cat")
    print(res)