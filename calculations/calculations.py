import numpy as np
import torch
import typing as t
from typing import Union

def calculateMS(image : np.ndarray):
    mean = []
    std = []
    channels = image.shape[-1]
    population = image.shape[0] * image.shape[1]
    if channels == 3:
        sum_ = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
    else:
        sum_ = [image]
        
    for i in sum_:
        total = np.sum(i)
        meanI = total/population
        mean.append(meanI)
        flattened = np.reshape(i, population)
        stdI = np.sqrt((np.sum([np.square(value - meanI) for value in flattened]))/len(flattened))
        std.append(stdI)
    return mean, std


def calculateTorch(image : torch.tensor):
    image = np.array(image)
    mean = []
    std = []
    channels = image.shape[0]
    population = image.shape[1] * image.shape[2]
    if channels == 3:
        sum_ = [image[0, :, :], image[1, :, :], image[2, :, :]]
    else:
        sum_ = [image[0, :, :]]
        
    for i in sum_:
        total = np.sum(i)
        meanI = total/population
        mean.append(meanI)
        flattened = np.reshape(i, population)
        stdI = np.sqrt((np.sum([np.square(value - meanI) for value in flattened]))/len(flattened))
        std.append(stdI)
    return mean, std

def calcMStd(dataset : np.ndarray):
    dsShape = dataset.shape
    if not len(dsShape) in [4, 3]:
        raise Exception(f"A valid dataset has to be given! Dataset of shape {len(dataset.shape)} given.")
    elif not dsShape[-1] in [1, 3]:
        raise Exception("Only 1 or 3 channel images are supported! Image shape should be in [height, width, channels]")

    if dsShape[-1] == 3: 
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([0.0, 0.0, 0.0])
    else: 
        mean = np.array([0.0])
        std = np.array([0.0])
    
    for img in dataset:
        meanimg, stdimg = calculateMS(img)
        mean += meanimg
        std += stdimg
    mean /= dsShape[0]
    std /= dsShape[0]
    return mean, std
    
    
def calcMStdtorch(dataset : Union[torch.tensor, np.ndarray]):
    dsShape = dataset.shape
    if len(dsShape) != 4:
        raise Exception(f"A valid dataset has to be given! Dataset of shape {len(dataset.shape)} given.")
    elif not dsShape[1] in [1, 3]:
        raise Exception("Only 1 or 3 channel images are supported! Image shape should be in [channels, height, width]")
    
    if dsShape[1] == 3: 
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([0.0, 0.0, 0.0])
    else: 
        mean = np.array([0.0])
        std = np.array([0.0])
    
    for img in dataset:
        meanimg, stdimg = calculateTorch(img)
        mean += meanimg
        std += stdimg
    mean /= dsShape[0]
    std /= dsShape[0]
    return mean, std


def normalizeimgN(img : np.ndarray, mean : np.ndarray, std : np.ndarray):
    shape = img.shape 
    if img.shape[-1] == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        channels = [r, g, b]
    else:
        channels = [img]
    chInd = 0
    for channel in channels:
        for row in range(shape[0]):
            for column in range(shape[1]):
                channel[row][column] = (channel[row][column] - mean[chInd]) / std[chInd]
        chInd += 1
    img = np.dstack(channels)
    return img         
    
def normalizeimgT(img : np.ndarray, mean : np.ndarray, std : np.ndarray):
    shape = img.shape 
    if img.shape[0] == 3:
        r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
        channels = [r, g, b]
    else:
        channels = [img]
    chInd = 0
    for channel in channels:
        for row in range(shape[1]):
            for column in range(shape[2]):
                channel[row][column] = (channel[row][column] - mean[chInd]) / std[chInd]
        chInd += 1
    img = np.dstack(channels)
    img = np.moveaxis(img, -1, 0)
    return img         
    

def normalizeN(dataset : np.ndarray, mean : np.ndarray, std : np.ndarray):
    dsShape = dataset.shape
    if not dsShape[-1] in [1, 3]:
        raise Exception("Only 1 or 3 channel images are supported! Image shape should be [height, width, channel]")
    
    for imgCount in range(dsShape[0]):  
        dataset[imgCount] = normalizeimgN(dataset[imgCount], mean, std)
        
    return dataset
        
    
def normalizeT(dataset : Union[np.ndarray, torch.tensor], mean : np.ndarray, std : np.ndarray):
    dataset = np.array(dataset)
    dsShape = dataset.shape
    if not dsShape[1] in [1, 3]:
        raise Exception("Only 1 or 3 channel images are supported! Image shape should be [channel, height, width]")
    
    for imgCount in range(dsShape[0]):
        dataset[imgCount] = normalizeimgT(dataset[imgCount], mean, std)
        
    return torch.tensor(dataset)
