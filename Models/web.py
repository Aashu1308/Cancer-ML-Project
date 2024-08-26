from ultralytics import YOLO
import streamlit as st
import os
from datetime import datetime

# Define the models
model_dir = "Models/model"
mtype = "best.pt"
models = {
    "Brain Tumor Detection": os.path.join(model_dir, "braintumor", mtype),
    "Cervical Cancer Detection": os.path.join(model_dir, "cervicalcancer", mtype),
    "Skin Cancer Detection": os.path.join(model_dir, "skincancer", mtype),
}

st.header('Select Detection Model')
model_choice = st.selectbox("Choose a model", options=list(models.keys()))
selected_model_path = models[model_choice]

model = YOLO(selected_model_path)

st.header('Upload Image for Detection')
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    st.subheader("Uploaded Image")
    st.image(upload_file)
    save_dir = "Streamlit/Upload"
    os.makedirs(save_dir, exist_ok=True)
    unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"predict_{unique_name}"
    file_path = os.path.join(save_dir, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    results = model(
        file_path, save=True, project="Streamlit/Results", name=output_name, conf=0.5
    )
    result_image_path = os.path.join("Streamlit/Results", output_name, upload_file.name)
    for result in results:
        if result.boxes.conf.shape[0] > 0:
            st.subheader("Result Image")
            st.image(result_image_path)
            print(result.boxes)
        else:
            st.error("No detections were made.")
