from ultralytics import YOLO
import streamlit as st
import os
from datetime import datetime
import utils as u

# Define the detection models
model_dir_d = "Models/model"
mtype = "best.pt"
models_d = {
    "Brain Tumor Detection": os.path.join(model_dir_d, "braintumor", mtype),
    "Cervical Cancer Detection": os.path.join(model_dir_d, "cervicalcancer", mtype),
    "Skin Cancer Detection": os.path.join(model_dir_d, "skincancer", mtype),
}

# Define the classification models
model_dir_c = "Models/model/cls"
models_c = {
    "Brain Tumor Detection": os.path.join(model_dir_c, "braintumor", mtype),
    "Cervical Cancer Detection": os.path.join(model_dir_c, "cervicalcancer", mtype),
    "Skin Cancer Detection": os.path.join(model_dir_c, "skincancer", mtype),
}

st.header('Select Detection and Classification Model')
model_choice = st.selectbox("Choose a model", options=list(models_d.keys()))
selected_model_d_path = models_d[model_choice]
selected_model_c_path = models_c[model_choice]

model_d = YOLO(selected_model_d_path)
model_c = YOLO(selected_model_c_path)

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
    results_d = model_d(
        file_path, save=True, project="Streamlit/Results", name=output_name, conf=0.6
    )
    results_c = model_c(file_path)
    output_dir_c = "Streamlit/Results/Classify"
    os.makedirs(output_dir_c, exist_ok=True)
    result_image_path_c = os.path.join(output_dir_c, f"{unique_name}.svg")
    u.classification_graph(results_c, result_image_path_c, model_choice)
    result_image_path_d = os.path.join(
        "Streamlit/Results", output_name, upload_file.name
    )
    for result in results_d:
        if result.boxes.conf.shape[0] > 0:
            st.subheader("Result Image")
            st.image(result_image_path_d)
            if model_choice != 'Brain Tumor Detection':
                st.image(result_image_path_c)
            print(result.boxes)
        else:
            if model_choice != 'Brain Tumor Detection':
                st.subheader("Classification")
                st.image(result_image_path_c)
