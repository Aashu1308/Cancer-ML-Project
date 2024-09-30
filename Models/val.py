from ultralytics import YOLO
import os

# Define the detection models
model_dir_d = "Models/model"
mtype = "best.pt"
models_d = {
    1: os.path.join(model_dir_d, "braintumor", mtype),
    2: os.path.join(model_dir_d, "cervicalcancer", mtype),
    3: os.path.join(model_dir_d, "skincancer", mtype),
}

# Define the classification models
model_dir_c = "Models/model/cls"
models_c = {
    1: os.path.join(model_dir_c, "braintumor", mtype),
    2: os.path.join(model_dir_c, "cervicalcancer", mtype),
    3: os.path.join(model_dir_c, "skincancer", mtype),
}

choices_d = models_d.keys()

# for i in choices_d:
#     selected_model_d_path = models_d[i]
#     selected_model_c_path = models_c[i]

#     model_d = YOLO(selected_model_d_path)
#     model_c = YOLO(selected_model_c_path)

#     metrics_d = model_d.val()
#     metrics_c = model_c.val()

# selected_model_d_path = models_d[3]
# model_d = YOLO(selected_model_d_path)
# metrics_d = model_d.val()

selected_model_c_path = models_c[1]
model_c = YOLO(selected_model_c_path)
metrics_c = model_c.val()
