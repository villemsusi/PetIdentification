import streamlit as st
import os
from datetime import datetime
from PIL import Image

from train import model_train
from cam import camera_setup, capture
from object_detect import detect, setup_model

@st.cache_resource
def start_cam():
    picam = camera_setup()
    return picam

@st.cache_resource
def detection_model():
    detection = setup_model()
    return detection



def read_logs(amount):
    with open("logs/capture_log.txt") as f:
        log_list = f.readlines()
    return log_list[-amount:][::-1]


def write_logs(data):
    with open("logs/capture_log.txt", "a") as f:
        f.write(str(data) + "\n")



picam = start_cam()
detection = detection_model()


st.title("Pet Recognition Interface")

tab_titles = ["Monitor", "Logs", "Train"]
tab_monitor, tab_logs, tab_train = st.tabs(tab_titles)


with tab_logs:
    st.header("LOG HISTORY")

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
                st.image(f"logs/{line}.jpg")
    
with tab_train:
    st.header("Training Mode")
    images = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    nr_cols = 4
    cols = st.columns(nr_cols, gap="small")
    for ind, image in enumerate(images):
        with cols[ind % nr_cols]:
            cols[ind % nr_cols].image(image, use_column_width=True)

    if st.button("UPLOAD", key="upload"):
        if len(images) != 0:
            for image in images:
                with open(os.path.join("images/train/Correct", image.name), "wb") as f:
                    f.write(image.getbuffer())
            model_train(1)
        else:
            st.markdown(''':red[Add Images!]''')
with tab_monitor:
    st.header("Monitoring Mode")

    if st.button("Detect", key="detect"):
        image = capture(picam)
        res = detect(image, detection)

        _time = str(datetime.now())
        res.save(f"logs/{_time}.jpg")
        write_logs(_time)


    nr_cols = 2
    cols = st.columns(nr_cols, gap="small")
    with cols[0]:
        STREAM = st.empty()
        while True:
            image = capture(picam)
            STREAM.image(image)

picam.close()

            

        
    

    
    

 
