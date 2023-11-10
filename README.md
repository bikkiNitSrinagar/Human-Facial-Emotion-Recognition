# Human-Facial-Emotion-Recognition
A tensorflow/keras implementation of a facial emotion recognition model based on a convolutional neural network architecture and trained on the FER2013 dataset with FERPlus labels.
## Built With
* Keras
* Tensorflow
* OpenCV

# Getting Started
## Prerequisites
* python >= 3.7.9
* keras >= 2.4.3
* tensorflow >= 2.3.1
* opencv >= 4.4
* sklearn >= 0.23
* numpy >= 1.18.5
* pandas >= 1.1.2
* matplotlib >= 3.3.1
## Installation
 1.Clone the repo
  [https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition]
 # Here are some test examples:
![282093994-f5dcbe5a-4dc2-4f8c-8d72-c77362ae4df4](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/f0c474c4-ad07-4c22-88f5-e8dea00456f7)

![282093639-514fefcd-93b8-4795-835f-c8516521e86c](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/4954ea49-b7a1-4fd8-a135-e5869d142d1e)

![282093944-746d68f9-f4f6-48f9-85f1-ad92115c945b](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/2c6ed466-bc60-4dba-993a-9976d16d2f92)

# Improving Model Performance
## Baseline Model
Used [neha01](https://github.com/neha01/Realtime-Emotion-Detectio) model as baseline model which is based on a 3 block convolutional neural network architecture. It achieved ~57.5% test accuracy on FER2013 dataset.
## Data Cleaning
Because of alot of mislabeled images in FER2013 dataset, we found that using FERPlus' labels is a better option to train the model for better performance.

Here are some examples of the FER vs FER+ labels extracted from the mentioned paper in FER+ repo (FER top, FER+ bottom):

![FER+vsFER](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/d4af0edf-3b76-4b85-a1bf-19c18a98e67e)

We also added 2 more blocks to the baseline model without regularization thus overall accuracy increased by ~14.
# Regularization
## 1. Data Augmentation
Data augmentation is used to artifically create images, these images are added to the original training images to increase the total training set size.
We implemented data augmentation with keras [ImageDataGenerator](https://keras.io/api/data_loading/image/) class and tuned its parameters. By doing so, we were able to raise the test accuracy by ~7%.
The trick was not to overuse it so that the model could still learn from the training images.

## 2. Batch Normalization and Dropout Layers
Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 which makes training faster and more stable.
Dropout layers randomly chooses percentage of input neurons to drop while training such that it has a regularization effect.
Both layers are added to our model improving performance by ~5%

# Performance Analysis
Plotting the accuracy and loss of the trained model is always the first step to anaylze how the the model is performing. Here are two pictures illustrating the difference in performance between one of the initial architectures and the final architecture.
![96019814-5d913480-0e4d-11eb-8679-b278ab47840d](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/3f99d114-2c8c-4165-ad7f-74cd6204c424) ![96056745-aebe1a00-0e87-11eb-9198-ceb4e274b50b](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/4758c8aa-93bf-46aa-b187-6db7cc8b28b5)

The plot on the left is for our initial architecture, we can see that the model started to overfit in the early epochs which meant that either that model wasn't the best fit for the dataset or that the dataset itself wasn't sufficient for the model to learn enough features to be able to predict with high accuracy.
On the other hand, the plot on the right shows that the cross-validation accuracy was keeping up with the training accuracy up to the 80s which is a good sign and it's certainly an improved performance from the one on the left.
Our final architecture had a test accuracy of ~84%. The architecture is a combination of these 3 blocks:
![96025592-9df4b080-0e55-11eb-917f-19b17820c4e0](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/8c756645-7e23-4b60-ac80-e25510583329) ![96025536-8caba400-0e55-11eb-8f27-29e9182459ac](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/8d523be8-398c-4dc2-b21a-16a38a515da6) ![96025489-7aca0100-0e55-11eb-8b08-ed17fcf30ba7](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/c312a42d-b09e-48c6-ad8d-04deda7cb1c3)
However, depending on only the accuracy and loss of the trained model doesn't always give a full understanding of the model's performance.
There are more advanced metrics that can be used like the F1 score which we decided to use. The F1 score is calculated using two pre-calculated metrics: precision and recall. These two metrics utilize the true positive, false positive and false negative predicted examples which are best visualised using the confusion matrix.
You can checkout [](https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd) for a full and clear explanation.
Since we designed our model to recognise the 7 universal facial emotions and the FERPlus dataset had an 8th class for 'contempt' emotions, we decided to add all contempt class' examples to the 'neutral' class rather than throwing this data away.
Here's how our confusion matrix for the 7 classes looks like, the X-axis is for predicted labels and the Y-axis is for the true ones.
F1 score = 0.8.

![96011743-9a582e00-0e43-11eb-9b95-eba91f99aa6f](https://github.com/bikkiNitSrinagar/Human-Facial-Emotion-Recognition/assets/66418501/1486361a-2377-4e5a-a7fa-39e63f97d7c8)










