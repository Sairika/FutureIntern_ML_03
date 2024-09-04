# CIFAR-10 Image Classification with CNN

## Introduction

This project implements Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset is a popular benchmark in computer vision, containing 60,000 32x32 color images categorized into 10 classes, including airplanes, automobiles, birds, and more. The primary objective of this project is to train and evaluate different CNN models to accurately classify these images into their respective categories.

## Dataset Overview

The CIFAR-10 dataset consists of:
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images

Each image in the dataset belongs to one of the following 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this project, the test set was further split into validation and test sets to fine-tune the model's hyperparameters.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Model Architectures](#model-architectures)
4. [Training Process](#training-process)
5. [Evaluation and Results](#evaluation-and-results)
6. [Visualizations](#visualizations)
7. [Conclusion](#conclusion)

## Model Architecture

### 1. Original Custom CNN Model
The first CNN model was built using TensorFlow and Keras, consisting of the following layers:

- **Conv2D + BatchNormalization + ReLU**
- **Conv2D + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Conv2D + BatchNormalization + ReLU**
- **Conv2D + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Conv2D + BatchNormalization + ReLU**
- **Conv2D + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Flatten**
- **Dense (256 units) + BatchNormalization + ReLU**
- **Dropout**
- **Dense (10 units) + Softmax**

### 2. Improved Custom CNN Model
The improved model includes additional layers, more filters, and data augmentation, along with an increased number of epochs:

- **Conv2D (64 filters) + BatchNormalization + ReLU**
- **Conv2D (64 filters) + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Conv2D (128 filters) + BatchNormalization + ReLU**
- **Conv2D (128 filters) + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Conv2D (256 filters) + BatchNormalization + ReLU**
- **Conv2D (256 filters) + BatchNormalization + ReLU**
- **MaxPooling2D + Dropout**
- **Flatten**
- **Dense (512 units) + BatchNormalization + ReLU**
- **Dropout**
- **Dense (10 units) + Softmax**

### 3. ResNet34 Model
A ResNet34 model was also implemented to leverage the advantages of residual learning for deeper networks.

## Training Process

### Original Custom CNN Model
The original custom model was trained on the CIFAR-10 training set for 50 epochs with a batch size of 64. The Adam optimizer was used, and the loss function was sparse categorical cross-entropy. Regularization techniques such as L2 regularization and dropout were employed to prevent overfitting. Early stopping and learning rate scheduling were also implemented.

### Improved Custom CNN Model
The improved custom model was trained for 100 epochs with data augmentation applied to the training set. The same optimization strategies were used as in the original model, with additional dropout layers to enhance regularization.

### ResNet34 Model
The ResNet34 model was trained using the same setup but with a focus on exploiting residual connections to improve the depth of the network without sacrificing performance.

## Evaluation and Results

The performance of each model was evaluated on the CIFAR-10 test set, achieving the following accuracies:
- **Original Custom CNN Model**: 88.54%
- **Improved Custom CNN Model**: 91.94%
- **ResNet34 Model**: 83.00%

The models' performances were evaluated using various metrics such as the confusion matrix, classification report, ROC curve, and AUC score.

## Visualizations

Training and validation metrics such as accuracy and loss were plotted to visualize the learning curves for each model. Additionally, sample images from the dataset were displayed along with their predicted and true labels.

## Conclusion

In this project, we explored different CNN architectures for image classification on the CIFAR-10 dataset. The improved custom CNN model achieved the highest accuracy at 91.94%, demonstrating the effectiveness of deeper networks and data augmentation. The ResNet34 model, while powerful, did not outperform the custom models in this particular case. Regularization techniques and careful hyperparameter tuning were crucial in achieving high performance.

The CNN model effectively classified images from the CIFAR-10 dataset with high accuracy. Regularization techniques like L2 regularization and dropout played a crucial role in preventing overfitting, while early stopping and learning rate scheduling optimized the training process. The project demonstrates the power of deep learning techniques in image classification tasks.
