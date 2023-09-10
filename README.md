# Age and Gender Detection

This is a machine learning project that uses computer vision techniques to detect the age and gender of a person in an image.

## Overview
In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face.
We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer).
It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression

## Dataset
For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification.
This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. 
The models we will use have been trained on this dataset.

## Model Architecture

The CNN Architecture
The convolutional neural network for this python project has 3 convolutional layers:

Convolutional layer; 96 nodes, kernel size 7
Convolutional layer; 256 nodes, kernel size 5
Convolutional layer; 384 nodes, kernel size 3
It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it

## Results

The model achieves an accuracy of over 90% on the validation set for both age and gender predictions. The model is able to predict the correct age and gender for most images in the dataset, but there are some cases where the predictions are incorrect or ambiguous.
![02](https://github.com/Bouchnak-Maher/age-and-gender-detection/assets/94197705/0b58d06e-2f32-40a5-bf7d-542061a245e1)   ![38](https://github.com/Bouchnak-Maher/age-and-gender-detection/assets/94197705/3c41d9f5-058b-419c-ab46-f37826528bca)

![female](# Age and Gender Detection

This is a machine learning project that uses computer vision techniques to detect the age and gender of a person in an image.

## Overview
In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face.
We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer).
It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression

## Dataset
For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification.
This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. 
The models we will use have been trained on this dataset.

## Model Architecture

The CNN Architecture
The convolutional neural network for this python project has 3 convolutional layers:

Convolutional layer; 96 nodes, kernel size 7
Convolutional layer; 256 nodes, kernel size 5
Convolutional layer; 384 nodes, kernel size 3
It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it

## Results

The model achieves an accuracy of over 90% on the validation set for both age and gender predictions. The model is able to predict the correct age and gender for most images in the dataset, but there are some cases where the predictions are incorrect or ambiguous.
![02](https://github.com/hachichaeya/gender-and-age-detection/blob/main/Image4'.png)   ![38](https://github.com/hachichaeya/gender-and-age-detection/blob/main/Image5'.png)

![female](https://github.com/hachichaeya/gender-and-age-detection/blob/main/Image2'.png)
![male](https://github.com/hachichaeya/gender-and-age-detection/blob/main/Image3'.png)
## Usage

To use the model, simply provide an image of a person's face as input to the model, and the model will output the predicted age and gender of the person in the image.
