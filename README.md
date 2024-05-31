# Anomaly_Detection_3D
An implementation of Anomaly Detection in 3D Point Clouds using Deep Geometric Descriptors by Paul Bergmann and David Sattlegger

## Getting Started

```bash
git clone https://github.com/yashmewada9618/Anomaly_Detection_3D.git
cd Anomaly_Detection_3D
pip install -r requirements.txt
```

## Prepare directory structure for synthetic dataset generation

```bash
cd Anomaly_Detection_3D/
mkdir -p datasets/pretrained_dataset/FPS_PCD datasets/pretrained_dataset/Original_PCD datasets/pretrained_dataset/Scenes datasets pretrained_dataset/train pretrained_dataset/val
mkdir -p datasets/mvtec_point_clouds
cd datasets
curl -S curl -S -O https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
```

## Prepare directory structure for mvtec dataset generation

```bash
cd Anomaly_Detection_3D/
mkdir -p datasets/MvTec_3D
curl -S -O https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b961b/download/428824485-1643285832/mvtec_3d_anomaly_detection.tar.xz
tar -xvf mvtec_3d_anomaly_detection.tar.xz -C MvTec_3D/
```


## Run python code to generate synthetic dataset

```bash
cd Anomaly_Detection_3D/
python3 dataset_generation/synthetic_data_generation.py
```

## Run python code to generate mvtec dataset (Optional as the dataloader already loads the .tiff files and converts to pcd)

```bash
cd Anomaly_Detection_3D/
python3 dataset_generation/mvtec_dataset_generation
```

## Teacher loss plots
![Train and Validation Loss](runs/exp3/Teacher_Loss.png)

## Student loss plots
![Student Loss](runs/exp3/studentloss.png)