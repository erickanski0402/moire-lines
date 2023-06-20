import numpy as np
# from keras.applications import MobileNetV3Small
from keras import losses, activations, optimizers
from keras.applications.resnet import ResNet50
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization, Add, \
    Input, AveragePooling2D, ZeroPadding2D, DepthwiseConv2D, ReLU, AvgPool2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2

from Constants import *


def fitModel(model, epochs, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model


def createSigmoidModel():
    print('creating sigmoid CNN model')
    input = Input(shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 1))
    # creating convolution layers
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = mobileNetBlock(x, filters=64, strides=1)
    x = mobileNetBlock(x, filters=128, strides=2)
    x = mobileNetBlock(x, filters=128, strides=1)
    x = mobileNetBlock(x, filters=256, strides=2)
    x = mobileNetBlock(x, filters=256, strides=1)
    x = mobileNetBlock(x, filters=512, strides=2)
    x = Dropout(0.5)(x)

    for _ in range(5):
        x = mobileNetBlock(x, filters=512, strides=1)
    x = Dropout(0.5)(x)
    
    x = mobileNetBlock(x, filters=1024, strides=2)
    x = mobileNetBlock(x, filters=1024, strides=1)
    x = AvgPool2D(pool_size=5, strides=1, data_format='channels_last')(x)

    # creating dense layers
    x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dense(256, activation='relu')(x)
    
    # creating output layer
    output = Dense(units=1, activation='sigmoid')(x)

    # compiling model
    model = Model(inputs=input, outputs=output)
    opt = optimizers.SGD(learning_rate=0.01)
    # opt = 'adam'
    loss = losses.BinaryCrossentropy(from_logits=False)
    # loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def createSoftmaxModel():
    # instantiating model
    model = Sequential()
    model.add(Input(shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 1)))
    # creating convolution layers
    model.add(Convolution2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(8, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    # try experimenting with larger numbers (0.5 is standard)
    model.add(Dropout(0.5))

    # creating dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compiling model
    opt = 'adam'
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    # training model against training set and validating accuracy
    return model


# def premadeMobileNetModel():
#     return MobileNetV3Small(
#         input_shape=((RESIZED_HEIGHT, RESIZED_WIDTH, 1)),
#         alpha=1.0,
#         minimalistic=False,
#         include_top=False,
#         weights="imagenet",
#         input_tensor=None,
#         classes=2,
#         pooling=None,
#         dropout_rate=0.5,
#         classifier_activation="softmax",
#         include_preprocessing=True,
#     )


def createMobileNetModel():
    # notes on current implementation
    #       - does pretty well with accuracy on training data
    #       - does very poorly generalizing on testing data
    #       - more filters needed to generalize better??
    #       - more dense layers causes network to take exponentially longer to train
    #       - fewer mobileNetBlocks causes network to take exponentially longer to train
    input = Input(shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 1))
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = mobileNetBlock(x, filters=64, strides=1)
    x = mobileNetBlock(x, filters=128, strides=2)
    x = mobileNetBlock(x, filters=128, strides=1)
    x = mobileNetBlock(x, filters=256, strides=2)
    x = mobileNetBlock(x, filters=256, strides=1)
    x = mobileNetBlock(x, filters=512, strides=2)

    for _ in range(5):
        x = mobileNetBlock(x, filters=512, strides=1)
    x = mobileNetBlock(x, filters=1024, strides=2)
    x = mobileNetBlock(x, filters=1024, strides=1)
    x = AvgPool2D(pool_size=7, strides=1, data_format='channels_last')(x)
    x = Flatten()(x)
    # x = Dense(units=256, activation='relu')(x)

    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)

    opt = 'adam'
#     opt = Adam(
#         learning_rate=0.00001
#     )
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def mobileNetBlock(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    return x


def premadeResNetModel(X_train, y_train, X_test, y_test):
    dataShape = X_train.shape

    model = ResNet50(
        include_top=False,
        input_shape=(dataShape[1], dataShape[2], dataShape[3]),
        classes=2
    )

    model.fit(X_train, y_train, X_test, y_test)
    return model


def createResNetModel(X_train, y_train, X_test, y_test):
    input_im = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))  # cifar 10 images size
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Convolution2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # # 5th stage
    #
    # x = res_conv(x, s=2, filters=(512, 2048))
    # x = res_identity(x, filters=(512, 2048))
    # x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)  # multi-class

    # define the model
    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    opt = 'adam'
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def res_identity(x, filters):
    # renet block where dimension doesnot change.
    # The skip connection is just simple identity conncection
    # we will have 3 blocks and then input will be added

    x_skip = x  # this will be used for addition with the residual block
    f1, f2 = filters

    # first block
    x = Convolution2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block # bottleneck (but size kept same with padding)
    x = Convolution2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Convolution2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters

    # first block
    x = Convolution2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Convolution2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Convolution2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Convolution2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(
        x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x
