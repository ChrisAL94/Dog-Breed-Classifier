[//]: # (Image References)


[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Dog Breed Classifier based on CNN (Framework: PyTorch)
*Repository of the Capstone Project of Udacity's Machine Learning Nanodegree - Dog Breed Classifier*

# Project Overview

The goal of the project is to build a machine learning model that can be used within web app to process real-world, user-supplied images. The algorithm has to perform two tasks:

   1. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.
   2. If supplied an image of a human, the code will identify the resembling dog breed.

For performing this multiclass classification, we can use Convolutional Neural Network to solve the problem.The solution involves three steps. First, to detect human images, we can use existing algorithm like OpenCV’s implementation of Haar feature based cascade classifiers. Second, to detect dog-images we will use a pretrained VGG16 model. Finally, after the image is identified as dog/human, we can pass this image to an CNN model which will process the image and predict the breed that matches the best out of 133 breeds. 

### CNN model created from scratch
<p align="justify">I have built a CNN model from scratch to solve the problem. The model has 3
convolutional layers. All convolutional layers have kernel size of 3 and stride 1. The
first conv layer (conv1) takes the 224*224 input image and the final conv layer
(conv3) produces an output size of 128. ReLU activation function is used here. The
pooling layer of (2,2) is used which will reduce the input size by 2. We have two
fully connected layers that finally produces 133-dimensional output. A dropout of
0.25 is added to avoid over overfitting.</p>

### Refinement - CNN model created with transfer learning
<p align="justify">The CNN created from scratch have accuracy of 13%, Though it meets the
benchmarking, the model can be significantly improved by using transfer learning.
To create <b>CNN with transfer learning</b>, I have selected the <b>Resnet101 architecture</b>
which is pre-trained on ImageNet dataset, the architecture is 101 layers deep. The
last convolutional output of Resnet101 is fed as input to our model. We only need
to add a fully connected layer to produce 133-dimensional output (one for each
dog category). The model performed extremely well when compared to CNN from


### Model Evaluation
<p align="justify">The CNN model created using transfer learning with
ResNet101 architecture was trained for 5 epochs, and the final model produced an
accuracy of 82% on test data. The model correctly predicted breeds for 692 images out of 836 total images.</p>

**Test Accuracy: 82% (692/836))**

## Software and Libraries

This project uses the following software and Python libraries:

* [Python](https://www.python.org/downloads/release/python-364/)
* [NumPy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [PyTorch](https://pytorch.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)


