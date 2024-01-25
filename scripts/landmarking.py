import cv2 as cv
import os
import urllib.request as urlreq
import matplotlib.pyplot as plt
from pylab import rcParams
from PIL import Image

img_path = "seb3.jpg"

img = cv.imread(img_path)


image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)

image_template = image_rgb.copy()

image_gray = cv.cvtColor(image_template, cv.COLOR_BGR2GRAY)

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv.CascadeClassifier(haarcascade)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(image_gray)

# Print coordinates of detected faces
print("Faces:\n", faces)

for face in faces:
#     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2 
    cv.rectangle(image_template,(x,y),(x+w, y+d),(255, 255, 255), 2)

image_test = Image.fromarray(image_template.astype('uint8'), 'RGB')

image_test.show()
