# Towards Earlier Detection of Oral Diseases On Smartphones Using Oral and Dental RGB Images
### Ayush Garg, Julia Lu, and Anika Maji

Oral diseases such as periodontal (gum) diseases and dental caries (cavities) affect billions of people across the world today. However, previous state-of-the-art models have relied on X-ray images to detect oral diseases, making them inaccessible to remote monitoring, developing countries, and telemedicine. To combat this overuse of X-ray imagery, we propose a lightweight machine learning model capable of detecting calculus (also known as hardened plaque or tartar) in RGB images while running efficiently on low-end devices. The model, a modified MobileNetV3-Small neural network transfer learned from ImageNet, achieved an accuracy of 72.73% (which is comparable to state-of-the-art solutions) while still being able to run on mobile devices due to its reduced memory requirements and processing times. Because our model can be accessed through smartphones, it has the potential to limit the number of serious oral disease cases as its predictions can help patients schedule appointments earlier without the need to go to the clinic. 

The current draft of the paper can be found in `./paper.pdf`. We are currently in the process of publishing the paper.


## Folder Structure
The folder structure is as follows (click on each link to go to the header with more information on that folder):
* [`./python/`](#running-the-code) - Python code used to create the models, train them, and save them
* [`./DentalClassificationApp/`](#running-the-mobile-app) - The proof of concept mobile app created to test the models

## Running the Code
The Python code is in the `python` folder and contain all of the code for the models. Use `main.py` for an interactive version of testing each model, edit `models/models.py` to change some hyperparameters for each model or add more models, and use files in `figures/` to generate some of the figures for the paper.

**Note:** Before running the code, make sure to add all images into `./python/dataset`. The images are available from https://github.com/PKNU-PR-ML-Lab/calculus ([Park et al.](https://doi.org/10.3390/electronics12071518)) and were processed using [Roboflow](https://roboflow.com/) before being used - see the paper for more information. Make sure that there are three folders, `dataset/train`, `dataset/valid`, and `dataset/test`, each with the processed train, validation, and test images respectively. 

## Running the Mobile App
The mobile app can be found in the `DentalClassificationApp` folder, and can be opened with [Android Studio](https://developer.android.com/studio) (tested with version `2022.2.1 Patch 2`).

**Note:** Make sure to replace the files in the `DentalClassificationApp/app/src/main/assets` folder before running the program (if running custom trained models):
1. `mobilenet.ptl`, which is the PyTorch Lite version of the MobileNet model (use the Python code to generate this).
2. `resnet34`, which is the PyTorch Lite version of the ResNet34 model (again, see the Python code).
3. `tensors.txt`, the text file containing the raw representations of each image as a tensor (which can be generated using `imgs_to_tensors.py` in the Python code)

**Note 2:** Make sure that you are running `torch==1.9.0` and `torchvision==1.10.1` (Python `3.7.9`) when generating the PyTorch Lite files, as otherwise the models might not work with the version of PyTorch used in the Android app.

## AUTHOR CONTRIBUTION STATEMENT 
A.G., J.L., and A.M. conceived the research topic. A.G. built and trained the models, and constructed the mobile app. J.L. constructed the graphs. A.M. created the figures. All authors contributed to the manuscript.
