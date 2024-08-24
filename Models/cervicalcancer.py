from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms
import utils as u

dir = 'Data/CervicalCancer'

# call u.organize_data(dir, dir) here
# class_list = [
#     'High squamous intra-epithelial lesion',
#     'Negative for Intraepithelial malignancy',
#     'Squamous cell carcinoma',
# ]
# call u.augment_data(dir, class_list, n=500, crop=512, enable=True) here

tr_df = u.train_df('Data/CervicalCancer/Training')
ts_df = u.train_df('Data/CervicalCancer/Testing')

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

train_ds = u.CustomDataset(tr_df, transform)
test_ds = u.CustomDataset(ts_df, transform)

train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)
class_dict = {
    0: 'High squamous intra-epithelial lesion',
    1: 'Low squamous intra-epithelial lesion',
    2: 'Negative for Intraepithelial malignancy',
    3: 'Squamous cell carcinoma',
}

# call u.view_with_class(train_loader, class_dict) here

model = YOLO('yolov8n.yaml')
results = model.train(data="cervicalcancer.yaml", epochs=30, imgsz=512, batch=10)
