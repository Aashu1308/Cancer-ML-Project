# Cancer Project for ML

## Kaggle Dataset Links

### Download these inside Data Folder

- [Brain Tumor MRI Dataset - save as BrainTumor](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Mendeley LBC Cervical Cancer - save as CervicalCancer](https://www.kaggle.com/datasets/blank1508/mendeley-lbc-cervical-cancer)
- [Skin Cancer MNIST: HAM10000 - save as SkinLesion](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Steps:

1. Clone the project or copy source code

```sh
  git clone https://github.com/Aashu1308/Cancer-ML-Project
  cd Cancer-ML-Project
```

2. Run make file to setup environment

```sh
   make
```

If `make` doesnt work, run `choco install make` and then try again

3. Enter the conda environment

```sh
   conda activate cancerML
```

4. Train the respective models - You may need to reset the settings.yaml for ultralytics to modify pathing
5. `cd Models` and then `streamlit run web.py`
