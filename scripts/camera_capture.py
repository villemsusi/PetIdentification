import cv2 as cv

def camera_setup():
    return cv.VideoCapture(0)

def capture(cam):
    result, image = cam.read()

    if result:
        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img
    else:
        print("No image captured")
        return None