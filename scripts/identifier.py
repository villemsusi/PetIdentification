from object_detect import detect, camera_detect, setup_model
from object_predict import inference
from camera_capture import camera_setup, capture

import os




if __name__ == "__main__":

    model = setup_model()
    cam = camera_setup()

    mode = input("c for camera; f for file")
    

    if mode == "c":
        while True:
            input("Press ENTER to analyse")
            image = camera_detect(capture(cam), model)
            res = inference("efficientnet_base.pth", image, "cat")
            print(res)
    if mode == "f":
        score = 0
        correct = 0
        counter = 0
        for f in os.listdir("test_incorrect"):
            image = detect("test_incorrect/"+f, model)
            res = inference("efficientnet_b0.ra_in1k.pth", image, "Incorrect")
            print(res)
            if res["Target"][0]==res["Predicted"][0]:
                correct+=1
            score += res["Confidence Score"]
            counter+=1
        print(correct, counter)
        print(score/counter)
        
        score = 0
        correct = 0
        counter = 0
        for f in os.listdir("test_correct"):
            image = detect("test_correct/"+f, model)
            res = inference("efficientnet_b0.ra_in1k.pth", image, "Correct")
            print(f, res)
            if res["Target"][0]==res["Predicted"][0]:
                correct+=1
            score += res["Confidence Score"]
            counter+=1
        print(correct, counter)
        print(score/counter)