from os import listdir, makedirs, remove
from os.path import exists
from shutil import move
            
def separateExistingData(dataPathPrefix, dataPath):
    for i, filename in enumerate(listdir(dataPath)):
        srcPath = f'{dataPath}/{filename}'
        newFilename = filename.split('_')[0]
        if 'moire' in filename:
            destPath = f'{dataPathPrefix}/positive-examples/{newFilename}.jpg'
#             print(f'moving {filename} from [srcPath]: {srcPath} to [destPath]: {destPath}')
            move(srcPath, destPath)
        else:
            destPath = f'{dataPathPrefix}/negative-examples/{newFilename}.jpg'
#             print(f'moving {filename} from [srcPath]: {srcPath} to [destPath]: {destPath}')
            move(srcPath, destPath)
        
def moveExistingData(dataPath):
    for i, dirname in enumerate(listdir(dataPath)):
        print(f'[dirname]: {dirname}')
        moveExistingDataFromDir(dataPath, dirname, i)
        pass
    pass

def moveExistingDataFromDir(dataPath, dirname, dirNum):
    dirDataPath = f'{dataPath}/{dirname}'
    for filename in listdir(dirDataPath):
        srcPath = f'{dataPath}/{dirname}/{filename}'
        destPath = f'{dataPath}/{dirNum}{filename}'
#         print(f'moving {filename} from [srcPath]: {srcPath} to [destPath]: {destPath}')
        move(srcPath, destPath)

if __name__ == '__main__':
    print('Moving data')
    dataPathPrefix = './data/big-bucket'
    dataPath = f'{dataPathPrefix}/examples/train'
    separateExistingData(dataPathPrefix, dataPath)
#     moveExistingData(dataPath)