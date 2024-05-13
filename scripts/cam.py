import time
from picamera2 import Picamera2
from libcamera import controls
import cv2 as cv

from PIL import Image


def camera_setup():
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"format": "RGB888", "size": (720, 720)})
    cam.configure(config)
    return cam

def capture(cam : Picamera2):
    cam.start()
    img = cam.capture_image("main")
    #img.show()
    cam.stop()
    return img

if __name__ == "__main__":
    cam = camera_setup(0)
    capture(cam).show()
    cam.close()