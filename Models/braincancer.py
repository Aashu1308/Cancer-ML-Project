from ultralytics import YOLO

# from torch.utils.data import DataLoader
# from torchvision import transforms
# import utils as u


def main():
    # tr_df = u.train_df('Data/BrainTumor/Training')
    # ts_df = u.train_df('Data/BrainTumor/Testing')

    # # call u.visualise(tr_df) here
    # # call u.visualise(ts_df) here

    # transform = transforms.Compose(
    #     [transforms.Resize((399, 399)), transforms.ToTensor()]
    # )

    # train_ds = u.CustomDataset(tr_df, transform)
    # test_ds = u.CustomDataset(ts_df, transform)

    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    # class_dict = {0: 'notumor', 1: 'glioma', 2: 'meningioma'}

    # call u.view_with_class(train_loader,class_dict) here
    save_dir = 'Models/model/braintumor'
    model = YOLO('yolov8n.yaml')
    model.to('cuda')
    results = model.train(
        data="Models/braincancer.yaml",
        epochs=30,
        imgsz=399,
        batch=150,
        save_dir=save_dir,
    )


if __name__ == '__main__':
    main()
