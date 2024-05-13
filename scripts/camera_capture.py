import cv2 as cv
from PIL import Image

def camera_setup():
    return cv.VideoCapture(0)

def capture(cam):
    result, image = cam.read()
    print(image)
    if result:
        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image
    else:
        print("No image captured")
        return None

cam = camera_setup()


capture(cam).show()