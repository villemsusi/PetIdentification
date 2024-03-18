from object_detect import detect, camera_detect, setup_model
from object_predict import inference
from cam import camera_setup, capture

import os




if __name__ == "__main__":

    model = setup_model()
    cam = camera_setup()

    mode = input("c for camera; f for file")
    

    if mode == "c":
        while True:
            input("Press ENTER to analyse")
            image = camera_detect(capture(cam), model)
            res = inference("efficientnet_6_3_2.pth", image, "Correct")
            print(res)
    if mode == "f":
        for i in ["Incorrect", "Correct"]:
            score = 0
            correct = 0
            counter = 0
            for f in os.listdir("Test_"+i):
                images = detect("Test_"+i+"/"+f, model)
                for image in images:
                    res = inference("efficientnet_6_3_2.pth", image, i)
                    print(res)
                    if res["Target"][0]==res["Predicted"][0]:
                        correct+=1
                    score += res["Confidence Score"]
                    counter+=1
            print(correct, counter)
            print(score/counter)