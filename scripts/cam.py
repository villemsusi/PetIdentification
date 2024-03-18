import time
from picamera2 import Picamera2, Preview

def camera_setup():
    cam = Picamera2()

    config = cam.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    cam.configure(config)
    cam.start()

    return cam

def capture(cam : Picamera2):

    time.sleep(2)
    img = cam.capture_image("main")

    return img