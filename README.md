
# RAS-MortDB: RAS Fish Mortality Detection Dataset and Model Weights



## Overview

RAS-MortDB is a publicly available RAS fish mortality detection dataset and model repository:

> Ranjan, R. et al. Does YOLO26 truly offer advantages over its predecessors for Edge-Deployed Object Detection? A Benchmark Case Study in Aquaculture (Paper under peer-review)



The dataset comprises 2,000 annotated images of dead and live fish collected from a semi-commercial RAS over a 90-day deployment period under ambient and supplemental lighting conditions. Trained model weights for twelve Ultralytics YOLO architectures across three size tiers are provided in PyTorch (.pt) and ONNX (.onnx) formats.





## Repository Structure

- dataset/        : 2,000 annotated images (train/valid/test split 70:20:10)

- weights/pytorch : 12 x PyTorch model weights (.pt)

- weights/onnx    : 12 x ONNX model weights (.onnx)

- inference/      : Example inference scripts



## Dataset

- Total images    : 2,000

- Classes         : Dead, Live

- Split           : 70:20:10 (train:valid:test)

- Annotation      : YOLO format (.txt)

- Lighting        : Ambient and supplemental

- Mortality levels: Zero, Low (<3 fish), High (>=3 fish)

- Collection      : 90-day commercial RAS deployment


## Sample Detection Results



### YOLO26n Detections



<p align="center">

  <img src="docs/assets/yolo26n_image20.jpg" width="45%">

  <img src="docs/assets/yolo26n_image82.jpg" width="45%">

</p>

<p align="center">

  <img src="docs/assets/yolo26n_image171.jpg" width="45%">

  <img src="docs/assets/yolo26n_image182.jpg" width="45%">

</p>



### YOLOv8n Detections



<p align="center">

  <img src="docs/assets/yolov8n_image20.jpg" width="45%">

  <img src="docs/assets/yolov8n_image82.jpg" width="45%">

</p>

<p align="center">

  <img src="docs/assets/yolov8n_image171.jpg" width="45%">

  <img src="docs/assets/yolov8n_image182.jpg" width="45%">

</p>




## Quick Start

### PyTorch inference

```python

from ultralytics import YOLO

model = YOLO("weights/pytorch/yolo26n_best.pt")

results = model.predict("your_image.jpg", conf=0.25)

results[0].show()

```

### ONNX inference (Raspberry Pi / CPU)

```python

from ultralytics import YOLO

model = YOLO("weights/onnx/yolo26n.onnx")

results = model.predict("your_image.jpg", conf=0.25)

results[0].show()

```



## Citation


If you use RAS-MortDB, please cite:

**Dataset:**

    Ranjan, R. (2026). RAS-MortDB: Fish Mortality Detection Dataset and Model
    Weights for Recirculating Aquaculture Systems (v1.0.0) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.20631781



## License

- Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)

- Model weights: AGPL-3.0 (consistent with Ultralytics YOLO license)



## Contact

Rakesh Ranjan

The Conservation Fund Freshwater Institute

Email: rranjan@conservationfund.org

