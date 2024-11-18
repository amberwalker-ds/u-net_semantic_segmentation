# Semantic Segmentation of Building Footprints Using U-Net

## Introduction
This project involves performing semantic segmentation to extract building footprints from aerial images. The goal is to create a model capable of predicting whether each pixel in an image belongs to a building or not, using a U-Net convolutional neural network architecture.

## U-Net Model
U-Net is a type of convolutional neural network (CNN) designed specifically for image segmentation tasks. It has a symmetric architecture with an encoder (contracting path) and a decoder (expanding path), forming a U-shaped structure.

- **Encoder:** Multiple convolutional layers followed by max-pooling layers, which reduce spatial dimensions and help capture general patterns.
- **Decoder:** Upsampling layers that increase spatial dimensions, followed by convolutional layers to refine features and produce a detailed segmentation map.
- **Skip Connections:** Link corresponding layers in the encoder and decoder, providing high-resolution features from the encoder to the decoder.

## Data Preparation
The dataset consists of 3347 color raster images (256x256x3 pixels), each representing an area of 300 square meters in Massachusetts. Corresponding to each image is a binary mask indicating building footprints derived from OpenStreetMap data.

### Steps:
1. **Normalization:** Pixel values are normalized to the range [0, 1] by dividing by 255.
2. **Dimension Check:** Ensured correspondence between images and labels by checking dimensions.
3. **Visualization:** Displayed several matching image and label pairs.

## Class Distribution
The dataset is imbalanced, with negative pixels outweighing positive pixels, as non-building areas make up more space than buildings.

## Simplified U-Net Model
Using the Keras functional API, a simplified U-Net model was defined and trained. Hyperparameters were optimized using Keras Tuner.

## Training
The model was trained using a batch size of 8 and 50 epochs, with early stopping to minimize validation error.

## Results
Evaluating the model on the test set, I achieved the following metrics:
Accuracy: 94.7% (Overall correctness of predictions)
Precision: 75.2% (Percentage of predicted buildings that were actual buildings)
Recall: 78.9% (Percentage of actual buildings correctly identified by the model)
Dice Metric: 76.7% (Measures the overlap between predicted and ground truth masks)
