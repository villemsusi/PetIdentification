import streamlit as st
import os
import time


from train import model_train
from cam import camera_setup, capture
from detection import detect, setup_model
from prediction import inference
from helper.funcs import read_logs

@st.cache_resource
def start_cam():
    picam = camera_setup()
    return picam

@st.cache_resource
def detection_model():
    detection = setup_model()
    return detection



picam = start_cam()
detection = detection_model()


st.title("Pet Recognition")

tab_titles = ["Monitor", "Logs", "Train"]
tab_monitor, tab_logs, tab_train = st.tabs(tab_titles)


with tab_logs:

    amount = st.number_input("Insert Amount", value=5)
    update = st.button("Update Logs", key="log_update")
    
    nr_cols = 4
    cols = st.columns(nr_cols, gap="small")

    if update:
        logs = read_logs(amount)
        for index, line in enumerate(logs):
            line = line.strip()
            with cols[index % nr_cols]:
                st.write(line)
                st.image(f"logs/images/{line}.jpg")
    
with tab_train:
    images = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    nr_cols = 4
    cols = st.columns(nr_cols, gap="small")
    for ind, image in enumerate(images):
        with cols[ind % nr_cols]:
            cols[ind % nr_cols].image(image, use_column_width=True)

    if st.button("Upload / Train", key="upload"):
        if len(images) != 0:
            for image in images:
                with open(os.path.join("images/train/Correct", image.name), "wb") as f:
                    f.write(image.getbuffer())
            model_train(1)
        else:
            st.markdown(''':red[Add Images!]''')
with tab_monitor:
    
    STREAM = st.empty()

    while True:
        time.sleep(2)

        image = capture(picam)
        STREAM.image(image)

        res_detect = detect(image, detection)

        if res_detect[0]:
            cnt = 0
            for i in range(10):
                res_predict = inference("efficientnet_6_3_2.pth", res_detect[1], "Correct")
                if res_predict["Target"][0]==res_predict["Predicted"][0]:
                    cnt += 1
                image = capture(picam)
                res_detect = detect(image, detection)
                    
            if cnt >= 7:
                print("Do something") # Signal to be sent
            print("COUNT: "+str(cnt))
            time.sleep(8) # Total 10 seconds
        else:
            time.sleep(3) # Total 5 seconds


picam.close()

            

        
    

    
    

 
