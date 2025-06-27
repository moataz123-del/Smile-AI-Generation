import face as Face
import message as Message
from PIL import Image
import streamlit as st
import os

# Global file path
filePath = None

def browsefiles():
    global filePath
    uploaded_file = st.file_uploader("Open file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        filePath = file_path
        return file_path
    return None


def start(file_path):
    try:
        Face.mouthDetection(file_path)
    except Exception as e:
        Message.error("Couldn't detect mouth")
        return None
    return file_path


def plotImage(imagePath, caption=None):
    st.image(imagePath, caption=caption, use_column_width=True) 