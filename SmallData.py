'''
The model is designed to practice transfer learning and data augment.
The reason why I imported 'os' is about a mistake of the package 'h5py'.
Although I had searched and tried tons of ways to fix it, it was still there.

After reading passages from the Internet, I think there should be two ways to
imply transfer learning.

The first way is you utilize a well-defined model like VGG16 to get an initial output.
Then taking the output as an input, you will define your own model and train this model.

The second way is to connect a well-defined model and self-defined model. Then freeze the
parameters of well-defined model and only train on the residual parts. And if you like, you
can train the whole model without any frozen layers.

The greatest merit of transfer learning, obviously, is effortless and energy-saving.
Especially when you do not have a powerful computer, like me now at home, it really helps a lot.

References:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg
'''

import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input
from keras.optimizers import SGD
from keras import applications
from keras.utils import to_categorical

# This function is uesed to data. And the data used here is cifar-10, which can be downloaded
# from the public site easily.
def readData(path):
    files = os.listdir(path)
    data = np.empty([len(files), 10000, 3072])
    labels = np.empty([len(files), 10000])
    for i, file in enumerate(files):
        with open(path + '/' + file, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            data[i] = dict[b'data']
            labels[i] = dict[b'labels']
            # print(data[i].shape)
            # print(labels[i].shape)
    print('read success')
    return data, labels


# This function can transfer the raw data into our desire format.
# np.moveaxis() can help to switch between channel-first mode and channel-last.
def dataPreprocess(train_path, validation_path, test_path):
    train_data, train_labels = readData(train_path)
    validation_data, validation_labels = readData(validation_path)
    test_data, test_labels = readData(test_path)

    print('read Success')

    print("train_data: ", train_data.shape)
    print("train_labels: ", train_labels.shape)

    print("validation_data: ", validation_data.shape)
    print("validation_labels: ", validation_labels.shape)

    print("test_data: ", test_data.shape)
    print("test_labels: ", test_labels.shape)

    train_data = train_data.reshape((-1, 3, 32, 32))
    train_data = np.moveaxis(train_data, 1, -1)
    train_labels = train_labels.reshape((-1, 1))

    validation_data = validation_data.reshape((-1, 3, 32, 32))
    validation_data = np.moveaxis(validation_data, 1, -1)
    validation_labels = validation_labels.reshape((-1, 1))

    test_data = test_data.reshape((-1, 3, 32, 32))
    test_data = np.moveaxis(test_data, 1, -1)
    test_labels = test_labels.reshape((-1, 1))

    print('after reshape: ')

    print("train_data: ", train_data.shape)
    print("train_labels: ", train_labels.shape)

    print("validation_data: ", validation_data.shape)
    print("validation_labels: ", validation_labels.shape)

    print("test_data: ", test_data.shape)
    print("test_labels: ", test_labels.shape)

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


# This function works to use VGG net to pre-process data and save the output as my model's input.
# Here the data augment has been applied. The generator works in loops to generate 'batch_size'
# pieces of data and, because of it, we have to set "steps" in the predict or fit function to
# avoid endless loops.
def transVGGModel(train_data, train_labels, batch_size, file_name):
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True, vertical_flip=True, rescale=1.0 / 255)
    generator = datagen.flow(x=train_data, y=train_labels, batch_size=batch_size)
    model = applications.VGG16(include_top=False, weights='imagenet')
    step_num = train_data.shape[0] // batch_size
    bottlenect_feature = model.predict_generator(generator, steps=step_num)
    np.save(file_name, bottlenect_feature)

# Here is my model, which places like the top of a bottle.
# And what you should pay attention to is that if the Model module applied,
# then the first layer should be the Input layer, serving for indicates the
# dimension of input data. This problem does not show in the Sequential module.
def myTopModel(input_shape):
    input_layer = Input(shape=input_shape, name='top_layer0')
    x = Flatten(name='top_layer1')(input_layer)
    x = Dense(512, activation='relu', name='top_layer2')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='top_layer3')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(10, activation='softmax', name='top_layer4')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load the output from well-defined model and train my model.
def preTrainMyTopModel(bottle_train_file_name):
    vgg_train_data = np.load(bottle_train_file_name)
    print("train_data: ", vgg_train_data.shape)
    input_shape = vgg_train_data.shape[1:]
    model = myTopModel(input_shape)
    train_history = model.fit(x=vgg_train_data, y=train_labels, batch_size=batch_size, verbose=1, epochs=1)
    model.save_weights(top_model_weights_path)

# This function first catenate the well-defined model with my model. Then freeze the former part
# and train the latter one.
# Here is a question I have spend a lot of time on it: when applying well-defined models, it will
# return a Model-like object. So, it will be a good habit to set a Input layer for it, if you want
# to change its architecture.
# Besides, now I understand  what  the tensor means deeper. When adding a new layer, the function will
# return a tensor instead of a layer-object or somethings else. Unlike vectors' shape, the tensor must
# have more attribute, acting as a class and is more powerful than a simple shape. I think the dictionary
# of all the layers must be one of them.
# Therefor, when you add an old layer to a new one, the dictionary will be updated and passed. That
# will be helpful when the keras does forward and backward propagation.

def smallDataModel(top_model_weights_path, input_shape):
    load_model = applications.VGG16(weights='imagenet', include_top=False)
    old_layer_num = len(load_model.layers)
    print('Model loaded.')
    input_layer = Input(shape=input_shape)
    load_model = load_model(input_layer)

    x = Flatten(name='top_layer1')(load_model)
    x = Dense(512, activation='relu', name='top_layer2')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='top_layer3')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(10, activation='softmax', name='top_layer4')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    for layer in model.layers[:old_layer_num]:
        layer.trainable = True

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# setup the orginal file path and images' properties.
train_path = 'C:/Users/YueYuHou/Desktop/cifar-10-python/train_data'
validation_path = 'C:/Users/YueYuHou/Desktop/cifar-10-python/validation_data'
test_path = 'C:/Users/YueYuHou/Desktop/cifar-10-python/test_data'
top_model_weights_path = 'C:/Users/YueYuHou/Desktop/cifar-10-python/top_model_weights_path.h5'

img_height = 32
img_weight = 32
img_channel = 3
batch_size = 5000

# data preparation
train_data, train_labels, validation_data, validation_labels, test_data, \
test_labels = dataPreprocess(train_path, validation_path, test_path)

# change labels from integer to one-hot code.
train_labels = to_categorical(train_labels, num_classes=10)
validation_labels = to_categorical(validation_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# using VGG to pre-train model and save results
bottle_train_file_name = 'bottle_train.npy'
# transVGGModel(train_data, train_labels, batch_size, bottle_train_file_name)

# training my model.
preTrainMyTopModel(bottle_train_file_name)


##### This works as the second way for applying transfer learning. #######
# I don't want to write two files to talk about the same topic. So I write them in one file.
input_shape = train_data.shape[1:]
mySmallDataModel = smallDataModel(top_model_weights_path, input_shape)

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, vertical_flip=True, rescale=1.0 / 255)
train_generator = datagen.flow(x=train_data, y=train_labels, batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow(x=validation_data, y=validation_labels, batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow(x=test_data, y=test_labels, batch_size=batch_size)

train_steps_per_epoch = train_data.shape[0] // batch_size
vali_steps_per_epoch = validation_data.shape[0] // batch_size
test_steps_per_epoch = test_data.shape[0] // batch_size
myFitHistory = mySmallDataModel.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=10,
                                              validation_data=validation_generator,
                                              validation_steps=vali_steps_per_epoch)

myPredicHis = mySmallDataModel.predict_generator(test_generator, steps=test_steps_per_epoch, verbose=1)
