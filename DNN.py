from keras import losses, optimizers
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential
from Constants import *


def fitModel(model, epochs, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model


def createSigmoidModel():
    print(f'Creating model with input shape: {(RESIZED_HEIGHT, RESIZED_WIDTH, 1)}')
    model = Sequential()
    model.add(Input(shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compiling model
    opt = optimizers.SGD(learning_rate=0.01)
    loss = losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def createSoftmaxModel():
    print(f'Creating model with input shape: {(RESIZED_HEIGHT, RESIZED_WIDTH, 1)}')
    # instantiating model
    model = Sequential()
    model.add(Input(shape=(RESIZED_HEIGHT, RESIZED_WIDTH, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compiling model
#     opt = 'adam'
    opt = optimizers.Adam(
        learning_rate=0.01
    )
    loss = losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    return model
