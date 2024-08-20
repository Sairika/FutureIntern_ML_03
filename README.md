# CIFAR-10 Image Classification with CNN

## Introduction

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset is a popular benchmark in computer vision, containing 60,000 32x32 color images categorized into 10 classes, including airplanes, automobiles, birds, and more. The primary objective of this project is to train a CNN model that can accurately classify these images into their respective categories.

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
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Evaluation and Results](#evaluation-and-results)
6. [Visualizations](#visualizations)
7. [Conclusion](#conclusion)

## Model Architecture

The CNN model was built using TensorFlow and Keras, consisting of the following layers:

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

Regularization techniques such as L2 regularization and dropout were employed to prevent overfitting.

## Training Process

The model was trained on the CIFAR-10 training set for 50 epochs with a batch size of 64. The Adam optimizer was used, and the loss function was sparse categorical cross-entropy. To optimize the training process, early stopping and learning rate scheduling were implemented.

## Evaluation and Results

The trained CNN model achieved the following performance:
- **Test Accuracy**: 88.56%

The model's performance was evaluated using various metrics such as the confusion matrix, classification report, ROC curve, and AUC score.

## Visualizations

Training and validation metrics such as accuracy and loss were plotted to visualize the learning curves. Additionally, sample images from the dataset were displayed along with their predicted and true labels.

## Conclusion

The CNN model effectively classified images from the CIFAR-10 dataset with high accuracy. Regularization techniques like L2 regularization and dropout played a crucial role in preventing overfitting, while early stopping and learning rate scheduling optimized the training process. The project demonstrates the power of deep learning techniques in image classification tasks.