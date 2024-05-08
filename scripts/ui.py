import streamlit as st
import os

from train import model_train

st.title("Pet Recognition Interface")

tab_titles = ["Train", "Monitor"]
tab_train, tab_monitor = st.tabs(tab_titles)

with tab_train:
    st.header("Training Mode")
    images = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    nr_cols = 4
    cols = st.columns(nr_cols, gap="small")
    for ind, image in enumerate(images):
        with cols[ind % nr_cols]:
            cols[ind % nr_cols].image(image, use_column_width=True)

    if st.button("UPLOAD"):
        if len(images) != 0:
            for image in images:
                with open(os.path.join("images/train/Correct", image.name), "wb") as f:
                    f.write(image.getbuffer())
            model_train(1)
        else:
            st.markdown(''':red[Add Images!]''')
with tab_monitor:
    st.header("Monitoring Mode")
    nr_cols = 3
    cols = st.columns(nr_cols, gap="small")
    with cols[0]:
        cols[0].image()
    

 
