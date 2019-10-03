
# Image Classification - Lab

## Introduction

Now that you have a working knowledge of CNNs and have practiced implementing associated techniques in Keras, its time to put all of those skills together. In this lab, you'll work to complete a Kaggle competition on classifying dog breeds.

https://www.kaggle.com/c/dog-breed-identification

## Objectives

You will be able to:
* Independently design and build a CNN for image classifcation tasks
* Compare and apply multiple techniques for tuning a model including data augmentation and adapting pretrained models

## Download and Load the Data

Start by downloading the data locally and loading it into a Pandas DataFrame. Be forewarned that this dataset is fairly large and it is advisable to close other memory intensive applications.

The data can be found here:

https://www.kaggle.com/c/dog-breed-identification/data

It's easiest if you download the data into this directory on your local computer. From there, be sure to uncompress the folder and subfolders. If you download the data elsewhere, be sure to modify the file path when importing the file below.


```python
# No code persay, but download and decompress the data.
```

## Preprocessing

Now that you've downloaded the data, its time to prepare it for some model building! You'll notice that the current structure provided is not the same as our lovely preprocessed folders that you've been given to date. Instead, you have one large training folder with images and a csv file with labels associated with each of these file types. 

Use this to create a directory substructure for a train-validation-test split as we have done previously. Also recall that you'll also want to use one-hot encoding as you are now presented with a multi-class problem as opposed to simple binary classification.


```python
# Your code here; open the labels.csv file stored in the zip file
```


```python
ls dog_breeds/train/ | head -5
```

    000bec180eb18c7604dcecc8fe0dba07.jpg
    001513dfcb2ffafc82cccf4d8bbaba97.jpg
    001cdf01b096e06d78e9e5112d419397.jpg
    00214f311d5d2247d5dfe4fe24b2303d.jpg
    0021f9ceb3235effd7fcde7f7538ed62.jpg



In order to input the data into our standard pipeline, you'll need to organize the image files into a nested folder structure. At the top level will be a folder for the training data, a folder for the validation data, and a folder for the testing data. Within these top directory folders, you'll then need to create a folder for each of the categorical classes (in this case, dog breeds). Finally, within these category folders you'll then place each of the associated image files. To save time, do this for just 3 of the dog breeds such as 'boston_bull', 'toy_poodle', and 'scottish_deerhound'.

You're nested file structure should look like this:
* train
    * category1
    * category2
    * category3
    ...
* val
    * category1
    * category2
    * category3
    ...
* test 
    * category1
    * category2
    * category3
    ...  

> **Hint**: To do this, you can use the `os` module which will you to use execute many common bash commands straight from your python interpreter. For example, here's how you could make a new folder:
```python
import os
os.mkdir('New_Folder_Name')
```
Start by creating top level folders for the train, validation and test sets. Then, use your pandas dataframe to split the example images for each breed of dog into a 80% train set, and 10% validation and test sets. Use `os.path.join()` with the information from the dataframe to construct the relevant file path. With this, place the relevant images using the `shutil.copy()` into the appropriate directory. 

>> **Note**: It is worthwhile to try this exercise on your own, but you can also use the images stored under the `'data_org_subset/'` folder of this repository, in which the Kaggle dataset has already been subset and preprocessed.


```python
# Your code here; transform the image files and then load them into Keras as tensors 
# (be sure to perform a train-val-test split)
```

## Optional: Build a Baseline CNN

This is an optional step. Adapting a pretrained model will produce better results, but it may be interesting to create a CNN from scratch as a baseline. If you wish to, do so here.


```python
# Create a baseline CNN model
```

## Loading a Pretrained CNN

## Feature Engineering with the Pretrained Model

As you may well have guessed, adapting a pretrained model will undoubtedly produce better results then a fresh CNN due to the limited size of training data. Import a pretrained model such as VGG19 to use a convolutional base. Use this to transform the dataset into a rich feature space and add a few fully connected layers on top of the pretrained layers to build a classification model. (Be sure to leave the pretrained model frozen!)


```python
# Your code here; add fully connected layers on top of the convolutional base
```

## Visualize History

Now fit the model and visualize the training and validation accuracy/loss functions over successive epochs.


```python
# Your code here; visualize the training / validation history associated with fitting the model.
```


```python
# Save model
```

## Final Model Evaluation

Now that you've trained and validated the model, perform a final evaluation of the model on the test set.


```python
# Your code here
```

## Summary

Congratulations! In this lab, you brought all of your prior deep learning skills together from preprocessing including one-hot encoding, to adapting a pretrained model. There are always ongoing advancements in CNN architectures and best practices, but you have a solid foundation and understanding at this point.
