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
cd Anomaly_Detection_3D/dataset_generation
curl -S curl -S -O https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
mkdir -p pretrained_dataset/FPS_PCD pretrained_dataset/Original_PCD pretrained_dataset/Scenes pretrained_dataset/train pretrained_dataset/val
```
