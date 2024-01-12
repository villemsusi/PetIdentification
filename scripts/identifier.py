from object_detect import detect
from object_predict import inference



if __name__ == "__main__":
    image = detect("test_image.jpg")
    res = inference("efficientnet_base.pth", image, "cat")
    print(res)