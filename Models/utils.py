import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numpy.random import seed
import numpy as np
import shutil
from glob import glob


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


def view_with_class(train_loader, class_dict):
    images, labels = next(iter(train_loader))

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


def create_dirs_sk(dir):
    tr_dr = os.path.join(dir, 'Training')
    os.mkdir(tr_dr)
    ts_dr = os.path.join(dir, 'Testing')
    os.mkdir(ts_dr)

    mel = os.path.join(tr_dr, 'mel')
    os.mkdir(mel)
    bcc = os.path.join(tr_dr, 'bcc')
    os.mkdir(bcc)
    akiec = os.path.join(tr_dr, 'akiec')
    os.mkdir(akiec)
    mel = os.path.join(ts_dr, 'mel')
    os.mkdir(mel)
    bcc = os.path.join(ts_dr, 'bcc')
    os.mkdir(bcc)
    akiec = os.path.join(ts_dr, 'akiec')
    os.mkdir(akiec)


def filter_data(dir):
    seed(101)
    tr_dr = os.path.join(dir, 'Training')
    ts_dr = os.path.join(dir, 'Testing')

    def identify_duplicates(x):
        unique_list = list(df['lesion_id'])

        if x in unique_list:
            return 'no_duplicates'
        else:
            return 'has_duplicates'

    md_df = pd.read_csv('Data/SkinLesion/HAM10000_metadata.csv')
    df = md_df.groupby('lesion_id').count()
    df = df[df['image_id'] == 1]
    df.reset_index(inplace=True)
    md_df['duplicates'] = md_df['lesion_id']
    md_df['duplicates'] = md_df['duplicates'].apply(identify_duplicates)
    df = md_df[md_df['duplicates'] == 'no_duplicates']
    wanted_classes = ['bcc', 'akiec', 'mel']
    df = md_df[md_df['dx'].isin(wanted_classes)]
    md_df = md_df[md_df['dx'].isin(wanted_classes)]
    y = df['dx']
    test_size = 0.17
    unique_classes, class_counts = np.unique(y, return_counts=True)
    val_counts = (class_counts * test_size).astype(int)
    val_indices = []
    for cls, val_count in zip(unique_classes, val_counts):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        val_indices.extend(cls_indices[:val_count])
    val_indices = np.array(val_indices)
    df_val = df.iloc[val_indices]

    def identify_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    md_df['train_or_val'] = md_df['image_id']
    md_df['train_or_val'] = md_df['train_or_val'].apply(identify_val_rows)
    df_train = md_df[md_df['train_or_val'] == 'train']
    md_df.set_index('image_id', inplace=True)
    p1 = os.path.join(dir, 'ham10000_images_part_1')
    p2 = os.path.join(dir, 'ham10000_images_part_2')
    folder_1 = os.listdir(p1)
    folder_2 = os.listdir(p2)
    train_list = list(df_train['image_id'])
    val_list = list(df_val['image_id'])
    for image in train_list:
        fname = image + '.jpg'
        label = md_df.loc[image, 'dx']
        if fname in folder_1:
            src = os.path.join(p1, fname)
            dst = os.path.join(tr_dr, label, fname)
            shutil.copyfile(src, dst)

        if fname in folder_2:
            src = os.path.join(p2, fname)
            dst = os.path.join(tr_dr, label, fname)
            shutil.copyfile(src, dst)

    # Transfer the val images

    for image in val_list:

        fname = image + '.jpg'
        label = md_df.loc[image, 'dx']

        if fname in folder_1:
            src = os.path.join(p1, fname)
            dst = os.path.join(ts_dr, label, fname)
            shutil.copyfile(src, dst)

        if fname in folder_2:
            src = os.path.join(p2, fname)
            dst = os.path.join(ts_dr, label, fname)
            shutil.copyfile(src, dst)


def print_size(dir):
    tr_dr = os.path.join(dir, 'Training')
    ts_dr = os.path.join(dir, 'Testing')

    mel = os.path.join(tr_dr, 'mel')
    print(len(os.listdir(mel)))
    bcc = os.path.join(tr_dr, 'bcc')
    print(len(os.listdir(bcc)))
    akiec = os.path.join(tr_dr, 'akiec')
    print(len(os.listdir(akiec)))
    mel = os.path.join(ts_dr, 'mel')
    print(len(os.listdir(mel)))
    bcc = os.path.join(ts_dr, 'bcc')
    print(len(os.listdir(bcc)))
    akiec = os.path.join(ts_dr, 'akiec')
    print(len(os.listdir(akiec)))


def augment_data(dir, class_list, n, crop=224, enable=False):
    tr_dr = os.path.join(dir, 'Training')

    for img_class in class_list:
        # Temporary directory to hold images before augmentation
        aug_dir = os.path.join(dir, 'aug_dir')
        os.makedirs(aug_dir, exist_ok=True)
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.makedirs(img_dir, exist_ok=True)

        img_list = os.listdir(os.path.join(tr_dr, img_class))
        for fname in img_list:
            src = os.path.join(tr_dr, img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)

        # Define transformations
        transform = transforms.Compose(
            [
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(crop, scale=(0.9, 1.1)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ]
        )

        if enable:
            # Create a DataFrame for CustomDataset
            img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
            labels = [img_class] * len(img_paths)
            df = pd.DataFrame({'img_path': img_paths, 'label': labels})

            # Custom dataset with transforms
            dataset = CustomDataset(dataframe=df, transform=transform)
            # Custom dataset with transforms
        else:
            dataset = CustomDataset(img_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

        # Calculate number of augmented images to generate
        num_aug_images_wanted = n
        num_files = len(os.listdir(img_dir))
        num_batches = int((num_aug_images_wanted - num_files) / 50)

        save_path = os.path.join(tr_dr, img_class)

        # Generate augmented images
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            for j, img in enumerate(batch[0]):
                # Convert tensor to PIL image for saving
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(os.path.join(save_path, f"aug_{i}_{j}.jpg"))

        # Clean up temporary directory
        shutil.rmtree(aug_dir)


def organize_data(root, output_dir, split=[0.8, 0.2]):
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    for sub_dir in ['Training', 'Testing']:
        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
        for class_name in sorted(
            [
                'High squamous intra-epithelial lesion',
                'Low squamous intra-epithelial lesion',
                'Negative for Intraepithelial malignancy',
                'Squamous cell carcinoma',
            ]
        ):
            os.makedirs(os.path.join(output_dir, sub_dir, class_name), exist_ok=True)

    # Get all image paths and labels
    image_paths = sorted(glob(f"{root}/*/*.jpg"))
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

    # Create a dataframe
    df = pd.DataFrame({'img_path': image_paths, 'label': labels})

    # Shuffle the dataframe using numpy
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    df = df.iloc[indices].reset_index(drop=True)

    # Calculate splits
    total_len = len(df)
    tr_len = int(total_len * split[0])
    ts_len = total_len - tr_len

    # Split dataframe
    df_train = df[:tr_len]
    df_test = df[tr_len:]

    # Move images
    for _, row in df_train.iterrows():
        im_path = row['img_path']
        class_name = row['label']
        shutil.copy(
            im_path,
            os.path.join(output_dir, 'Training', class_name, os.path.basename(im_path)),
        )

    for _, row in df_test.iterrows():
        im_path = row['img_path']
        class_name = row['label']
        shutil.copy(
            im_path,
            os.path.join(output_dir, 'Testing', class_name, os.path.basename(im_path)),
        )


def print_size_cc(dir):
    tr_dr = os.path.join(dir, 'Training')
    ts_dr = os.path.join(dir, 'Testing')
    a1 = []
    a2 = []
    for i in os.listdir(tr_dr):
        j = os.path.join(tr_dr, i)
        a1.append(len(os.listdir(j)))
    for i in os.listdir(ts_dr):
        j = os.path.join(ts_dr, i)
        a2.append(len(os.listdir(j)))
    print(f"Training: {a1}")
    print(f"Testing: {a2}")
