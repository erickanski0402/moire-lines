import os
import cv2 as cv
import numpy as np
from Constants import *


def loadImageData(dataPath):
    print(f'Loading image data with shape: {(0, RESIZED_HEIGHT, RESIZED_WIDTH)}')
    data = np.empty(shape=(0, RESIZED_HEIGHT, RESIZED_WIDTH))
    labels = []
    for i, filename in enumerate(os.listdir(dataPath)):
        if "jpg" in filename or "jpeg" in filename:
            if i % 50 == 0:
                print(f'Loading {filename} as #{i}')

            img = cv.imread(f"{dataPath}/{filename}", cv.IMREAD_GRAYSCALE)
            img = normalizeData(img)
            img = np.reshape(img, (1, RESIZED_HEIGHT, RESIZED_WIDTH))
            label = encodeLabel(filename.split('-')[0])
            data = np.append(data, img, axis=0)
            labels.append(label)

    num, length, width = data.shape
    data = data.reshape((num, length, width, 1))
    # data = makeDataMultiChannel(data)
    labels = np.array(labels)
    return data, labels


def loadArrayData(dataPath):
    data = np.empty(shape=(0, RESIZED_HEIGHT * RESIZED_WIDTH))
    labels = []
    for i, filename in enumerate(os.listdir(dataPath)):
        if i % 50 == 0:
            print(f'Loading {filename} as #{i}')

        img = cv.imread(f"{dataPath}/{filename}", cv.IMREAD_GRAYSCALE)
        img = normalizeData(img)
        img = np.reshape(img, (1, RESIZED_HEIGHT * RESIZED_WIDTH))
        label = encodeLabel(filename.split('-')[0])
        data = np.append(data, img, axis=0)
        labels.append(label)

    # data = makeDataMultiChannel(data)
    labels = np.array(labels)
    return data, labels


def makeDataMultiChannel(img):
    return np.repeat(img, 3, -1)


def normalizeData(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def encodeLabel(label):
    return NEG if label == 'neg' else POS
#     return [1, 0] if label == 'neg' else [0, 1]


if __name__ == '__main__':
    # x, y = loadImageData(WAVELET_TRANSFORMED_DATA_PATH)
    x, y = loadArrayData(BB_WAVELET_TRANSFORMED_DATA_PATH)

    print('x')
    print(x)
    print('x.shape')
    print(x.shape)
    print('y.shape')
    print(y.shape)
