from cProfile import label
import datetime
import glob
import io
import json
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from  tensorflow.keras.models import load_model
from tqdm import tqdm
from confusion_matrix import ConfusionMatrix
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from model import efficientnet_b0 as create_model
from mixupgenerator import MixupGenerator


norm_size = 224
datapath = 'data/train'
EPOCHS = 5
INIT_LR = 3e-4
labelList = []
classnum = 6
batch_size = 8
tree_indict = {
                0: 'blackcherry',
                1: 'butternut',
                2: 'chestnut',
                3: 'redoak',
                4: 'walnut ',
                5: 'whiteoak',
                }
mislabeled_image_path = r"mislabeled.jpg"

#image loader
def loadImageData():
 
    imageList = []
    listClasses = os.listdir(datapath)  # class folder
    for class_name in listClasses:
        class_path = os.path.join(datapath, class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_full_path = os.path.join(class_path, image_name)
            labelList.append(class_name)
            image = cv2.imdecode(np.fromfile(image_full_path, dtype=np.uint8), -1)
            image = cv2.resize(image, (norm_size, norm_size), interpolation=cv2.INTER_LANCZOS4)
            if image.shape[2] > 3:
                image = image[:, :, :3]
                print(image.shape)
            image = img_to_array(image)
            imageList.append(image)
    imageList = np.array(imageList) / 255.0
    return imageList



print("loading data...")
imageArr = loadImageData()
print("Done")
print(labelList)
lb = LabelBinarizer()
labelList = lb.fit_transform(labelList)
print(labelList)
print(lb.classes_)
f = open('label_bin.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()

#split the imageList into training set and validation set
trainX, valX, trainY, valY = train_test_split(imageArr, labelList, test_size=0.3, random_state=42)
"""
trainX : images in training set.
valX : true labels of images in training set
trainY : images in validation set.
valY : true labels of images in validation set
"""
# train_len = len(trainX)
# trainXGray = []
# for image in trainX:
#     image.convert('L')



def display_examples(class_names, images, labels, ture_labels, title='Some examples of images of the dataset'):
    """DISPLAY 100 EXAMPLE IMAGES"""
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    num = len(images) - 1
    a = num//6 + 1
    for i in range(num):
        plt.subplot(6, a, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        x = "ture: {name}, mislabeled :{name2}".format(name=class_names[ture_labels[i]] , name2=class_names[labels[i]])
        plt.xlabel(x)
        # plt.xlabel(class_names[labels[i]])
    plt.savefig(mislabeled_image_path)
    plt.show()



def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):
    """display 100 mislabeled images"""
    BOO = (test_labels == pred_labels)
    mislabeled_indices = np.where(BOO==0) # indices that differ from true labels
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]
    true_labels = test_labels[mislabeled_indices]
    title = 'Mislabeled images by the classifier'
    display_examples(class_names, mislabeled_images, mislabeled_labels, true_labels, title)
    

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    )
val_datagen = ImageDataGenerator(
)  

training_generator_mix = MixupGenerator(trainX, trainY, batch_size=batch_size, alpha=0.2, datagen=train_datagen)()
val_generator = val_datagen.flow(valX, valY, batch_size=batch_size, shuffle=True)
checkpointer = ModelCheckpoint(filepath='best_model.hdf5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=10,
                           verbose=1,
                           factor=0.5,
                           min_lr=1e-6)

model = Sequential()
model.add(EfficientNetB0(include_top=False, pooling='avg', weights='imagenet'))
model.add(Dense(classnum, activation='softmax'))
model.summary()
optimizer = Adam(learning_rate=INIT_LR)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_generator_mix,
                    steps_per_epoch=trainX.shape[0] / batch_size,
                    validation_data=val_generator,
                    epochs=EPOCHS,
                    validation_steps=valX.shape[0] / batch_size,
                    callbacks=[checkpointer, reduce])
model.save('my_model.h5')


print("[INFO] evaluating network...")

#plot confusion matrix
total_val = val_generator.n

model = load_model("best_model.hdf5")
lb = pickle.loads(open("label_bin.pickle", "rb").read())

labels = [label for _, label in tree_indict.items()]
confusion = ConfusionMatrix(num_classes=6, labels=labels)

for step in tqdm(range(math.ceil(total_val / batch_size))):
        val_images, val_labels = next(val_generator)
        results = model.predict_on_batch(val_images)
        results = tf.keras.layers.Softmax()(results).numpy()
        results = np.argmax(results, axis=-1)
        labels = np.argmax(val_labels, axis=-1)
        confusion.update(results, labels)
confusion.plot()
confusion.summary()


print("mislabeled data")
predictions = model.predict(x=valX)
print_mislabeled_images(tree_indict, valX, valY.argmax(axis=1), predictions.argmax(axis=1))


loss_trend_graph_path = r"WW_loss.jpg"
acc_trend_graph_path = r"WW_acc.jpg"


print("Now,we start drawing the loss and acc trends graph...")
# summarize history of accuracy
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)
# summarize history of loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)
print("We are done, everything seems OK...")

