# Aneurysm Detection

## Introduction

![Results](https://user-images.githubusercontent.com/53757856/193028982-989709bd-1e00-4b02-905d-01c06327805f.png)

The final implementation for our Machine Learning for Medical Image Processing Project 2022 at TU Berlin. 

Aneurysms are localized, abnormal weak spots on the blood vessel walls of the brain. With increased pressure, they pose multiple threats such as the formation of blood clots, strokes, or hemorrhage. Diagnosis of aneurysms relies on angiographic imaging
using MRI, CT, or X-ray rotation angiography. The goal of this project was to apply image analysis and machine learning methods to detect cerebral aneurysms, provide a segmentation mask of the area and predict the rupture risk of the aneurysm.

The first task of the project was to accurately detect all aneurysms and identify each of them by proving a seed point and object-oriented bounding box. The key difficulty of this task was to orient the bounding box along the shape of the aneurysm and not the cartesian coordinates, to make the box as minimal as possible. Additionally, the box coordinates needed to be provided in world coordinates.
The second task consisted of generating a segmentation mask for all aneurysms detected in a given image. 

## Code
The base model for our solution is a 3D U-Net implemented in [unet3d.py](unet3d.py) which is then loaded into our SegmentationModel, found in  [experiment.py](experiment.py). To train the segmentation model we implemented a variety of metrics [metrics.py](metrics.py). Furthermore, we implemented a minimal object-oriented bounding box [obb.py](obb.py) to solve task 2. The step-by-step process of training and generating the solutions can be seen in the Jupyter Notebook [Segmentation.ipynb](Segmentation.ipynb).
