from os import listdir, makedirs, remove
from os.path import exists
from PIL import Image
from Constants import IMAGE_HEIGHT, SB_SCALED_IMAGE_DATA_PATH, BB_NEGATIVE_EXAMPLES_PATH, \
    BB_WAVELET_TRANSFORMED_DATA_PATH, \
    BB_FOURIER_WAVELET_TRANSFORMED_DATA_PATH, BB_POSITIVE_EXAMPLES_PATH, \
    IMAGE_WIDTH, WAVELET_TRANSFORM, FOURIER_TRANSFORM, FOURIER_AND_WAVELET_TRANSFORM, UNTRANSFORMED, DATA_TYPE_DICT, \
    RESIZED_WIDTH, RESIZED_HEIGHT, SB_POSITIVE_EXAMPLES_PATH, SB_NEGATIVE_EXAMPLES_PATH, SB_DATA_TYPE_DICT, \
    BB_DATA_TYPE_DICT, SMALL_BUCKET, BIG_BUCKET
import cv2 as cv
import numpy as np
import pywt
import pywt.data


def clearDataDirectory(dataPath):
    print('dataPath')
    print(dataPath)
    print('exists(dataPath)')
    print(exists(dataPath))
    if exists(dataPath):
        print(f"{dataPath} exists, clearing out contents")
        for filename in listdir(dataPath):
            path = f"{dataPath}/{filename}"
            remove(path)
    else:
        print(f"{dataPath} does not exist, creating directory")
        makedirs(dataPath)
    return


def populateDataDir(dataPath, path, dataType, prefix):
    print(f"populating data dir with prefix: {prefix}")
    for filePath in listdir(path):
        srcPath = f"{path}/{filePath}"
        filename = filePath.split('.')[0]
        destPath = f"{dataPath}/{prefix}-{filename}.jpg"

        imgArr = cv.imread(srcPath, cv.IMREAD_GRAYSCALE)
        if imgArr is not None:
            imgArr = resizeAndOrientImage(
                imgArr,
                IMAGE_HEIGHT,
                IMAGE_WIDTH
            )
            imgArr = transformImage(imgArr, dataType)
            imgArr = resizeAndOrientImage(imgArr, RESIZED_HEIGHT, RESIZED_WIDTH)
            newImg = arrToImg(imgArr)
            newImg.save(destPath)
    return


def transformImage(imgArr, dataType):
    if WAVELET_TRANSFORM == dataType:
        imgArr = waveletTransform(imgArr)
        imgArr = scalePixelValues(imgArr)
    elif FOURIER_TRANSFORM == dataType:
        imgArr = fourierTransform(imgArr)
        imgArr = scalePixelValues(imgArr)
    elif FOURIER_AND_WAVELET_TRANSFORM == dataType:
        imgArr = waveletTransform(imgArr)
        imgArr = fourierTransform(imgArr)
        imgArr = scalePixelValues(imgArr)
        pass

    return imgArr


def waveletTransform(imgArr):
    coeffs2 = pywt.dwt2(imgArr, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    return LL


def fourierTransform(imgArr):
    f = np.fft.fft2(imgArr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


def resizeAndOrientImage(imgArr, longestSide, shortestSide):
    h, w = imgArr.shape
    if h > w:
        imgArr = np.transpose(imgArr)
    resizedImg = Image.fromarray(imgArr).resize((shortestSide, longestSide))
    return imgToArr(resizedImg)


def imgToArr(img):
    return np.array(img)


def arrToImg(imgArr):
    return Image.fromarray(imgArr)


def scalePixelValues(img):
    # ensures individual pixel values fall in range of 0-255 for simple computation
    #   in both grayscale and rgb
    return (255 * (img - img.min()) / np.ptp(img)).astype(np.uint8)


def resolvePaths(bucket, dataType):
    if bucket == 'small':
        return SB_DATA_TYPE_DICT.get(dataType), SB_POSITIVE_EXAMPLES_PATH, SB_NEGATIVE_EXAMPLES_PATH
    else:
        return BB_DATA_TYPE_DICT.get(dataType), BB_POSITIVE_EXAMPLES_PATH, BB_NEGATIVE_EXAMPLES_PATH


if __name__ == '__main__':
    bucket = SMALL_BUCKET
    dataType = UNTRANSFORMED
    currentDataPath, positiveExamplesPath, negativeExamplesPath = resolvePaths(bucket, dataType)

    dataTypePath = DATA_TYPE_DICT.get(currentDataPath)
    print(f'   Bucket: {bucket},\n' + 
          f'   dataType: {dataType},\n' + 
          f'   currentDataPath: {currentDataPath},\n' + 
          f'   dataTypePath: {dataTypePath},\n' + 
          f'   positiveExamples: {positiveExamplesPath},\n' + 
          f'   negativeExamples: {negativeExamplesPath}\n')
    clearDataDirectory(currentDataPath)
    populateDataDir(currentDataPath, positiveExamplesPath, dataTypePath, "pos")
    populateDataDir(currentDataPath, negativeExamplesPath, dataTypePath, "neg")
