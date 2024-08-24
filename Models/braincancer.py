from ultralytics import YOLO
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def train_df(tr_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(tr_path, label, image))
            for label in os.listdir(tr_path)
            if os.path.isdir(os.path.join(tr_path, label))
            for image in os.listdir(os.path.join(tr_path, label))
        ]
    )

    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df


def test_df(ts_path):
    classes, class_paths = zip(
        *[
            (label, os.path.join(ts_path, label, image))
            for label in os.listdir(ts_path)
            if os.path.isdir(os.path.join(ts_path, label))
            for image in os.listdir(os.path.join(ts_path, label))
        ]
    )

    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df


def visualise(tr_df):
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(data=tr_df, y=tr_df['Class'])
    plt.title('Count of images in each class', fontsize=20)
    ax.bar_label(ax.containers[0])
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def view_with_class(train_loader):
    images, labels = next(iter(train_loader))

    class_dict = {0: 'notumor', 1: 'glioma', 2: 'meningioma'}
    classes = list(class_dict.values())

    images = images.permute(0, 2, 3, 1).numpy()

    num_images_to_display = min(len(images), 16)
    plt.figure(figsize=(20, 20))

    for i, (image, label) in enumerate(
        zip(images[:num_images_to_display], labels[:num_images_to_display])
    ):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        class_name = label
        plt.title(class_name, color='k', fontsize=15)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


transform = transforms.Compose([transforms.Resize((399, 399)), transforms.ToTensor()])

tr_df = train_df('Data/BrainTumor/Training')
ts_df = train_df('Data/BrainTumor/Testing')

train_ds = CustomDataset(tr_df, transform)
test_ds = CustomDataset(ts_df, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = YOLO('yolov8n.yaml')
results = model.train(data="config.yaml", epochs=30, imgsz=399, batch=16)
