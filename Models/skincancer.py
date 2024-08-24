from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms
import utils as u

dir = 'Data/SkinLesion'

# call u.create_dirs(dir) here
# call u.filter_data(dir) here
# class_list = ['mel', 'bcc', 'akiec']
# call u.augment_data(dir,class_list,n=950) here

tr_df = u.train_df('Data/SkinLesion/Training')
ts_df = u.train_df('Data/SkinLesion/Testing')

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_ds = u.CustomDataset(tr_df, transform)
test_ds = u.CustomDataset(ts_df, transform)

train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)
class_dict = {0: 'akiec', 1: 'bcc', 2: 'mel'}

# call u.view_with_class(train_loader, class_dict) here

model = YOLO('yolov8n.yaml')
results = model.train(data="skincancer.yaml", epochs=30, imgsz=224, batch=10)
