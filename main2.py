#getting list of image files inside 'dog-breed-identification' folder and under the 'train' subfolder
import os
import pandas as pd
import seaborn as sns
import pylab as plt
import keras_tuner
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import layers, models
from scikeras.wrappers import KerasClassifier
import keras

base_dir = r"T:\PycharmProjects\120_Dog_Breed_Classification\dog-breed-identification"
label_list = pd.read_csv(os.path.join(base_dir, "labels.csv"), index_col=0)

#3. Data Preprocessing
# ● In this section, prepare the data you have, for training the model.
# ● Create a dataframe that includes pixel values of images and the labels

#getting random 100 images from label_list and define dataset
dataset = label_list.sample(n=1000, random_state=1)
dataset["filename"] = dataset.index + ".jpg"
dataset.reset_index(drop=True, inplace=True)

refactor_size = 128
resized_image_list = []
all_paths = []

for i in range(len(dataset)):
    image_path = os.path.join(base_dir, "train", dataset["filename"][i])
    img = tf.keras.utils.load_img(image_path, target_size=(refactor_size, refactor_size))
    imgarr = np.array(img)
    resized_image_list.append(imgarr)
    all_paths.append(image_path)


#adding resized images pixel values to dataset
dataset["resized_images"] = resized_image_list

# ● Use Label Encoding or One-Hot Encoding techniques to deal with categorical targets.
le = LabelEncoder()
dataset["breed"] = le.fit_transform(dataset["breed"])

# ● Split your dataset into X_train,X_test, X_val, y_train, y_test and y_val

X_train, X_test, y_train, y_test = train_test_split(dataset["resized_images"], dataset["breed"], test_size=0.2, random_state=1)
# converting x_train and x_test to numpy array
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

print(X_train.shape, y_train.shape)

# ● Normalize the pixel values.
X_train = X_train / 255
X_test = X_test / 255

# 4. Building a Model
# ● Build a model using Tensorflow or Pytorch
# ● Your model should include Conv2D, MaxPooling2D, Flatten, Dense and Dropout.(Number of layers is up to you)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(refactor_size, refactor_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Dropout layer after MaxPooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Another Dropout layer after MaxPooling
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Dropout(0.25),  # Dropout layer after Conv2D
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dropout(0.5),  # Dropout layer before the final Dense layer
    layers.Dense(120, activation='softmax')
])

# ● Compile your model and print the summary of the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ● Train your model using train and validation subsets
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
# ● Plot model’s accuracy, validation accuracy, loss and validation loss
def plot_accuracy(model, before_after_flag):
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title(f'Model Accuracy {before_after_flag}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"model_accuracy_{before_after_flag}.png")
    plt.show(block=True)

def plot_loss(model, before_after_flag):
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title(f'Model Loss {before_after_flag}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"model_loss_{before_after_flag}.png")
    plt.show(block=True)

plot_accuracy(model, 'before')
plot_loss(model, 'before')

# 5. Hyper-parameter Optimization
# ● Optimize the hyper-parameters of the model.

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(refactor_size, refactor_size, 3)))
    model.add(keras.layers.Dense(units=hp.Int('units',
                                              min_value=32,
                                              max_value=64,
                                              step=32),
                                 activation='relu'))
    model.add(keras.layers.Dense(120, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

tuner = keras_tuner.tuners.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Dog_Breed_Classification')

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
tuner.results_summary()
best_model = tuner.get_best_models()[0]

#  ● Train your model with the opitimized parameters and show the results.
best_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
plot_accuracy(best_model, 'after')
plot_loss(best_model, 'after')













