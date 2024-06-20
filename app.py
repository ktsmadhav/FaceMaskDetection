from matplotlib import pyplot as plt
import glob, os, joblib
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import time



from pathlib import Path
import imghdr

TRAINING_DIR = "DataSet\\MP2_FaceMask_Dataset\\train/"
VALIDATION_DIR = "DataSet\\MP2_FaceMask_Dataset\\test/"
input_shape = (256, 256, 1)

def clean_data():
  incompatible_images = []

  folders = ['with_mask', 'partial_mask', 'without_mask']
  image_extensions = ['jfif', 'webp', 'jpeg', 'png', 'gif', 'jpg']  # add there all your images file extensions

  img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
  for folder in folders:
      for filepath in Path(TRAINING_DIR + folder).rglob("*"):
          # print(filepath)
          # if filepath.suffix.lower() in image_extensions:
          img_type = imghdr.what(filepath)
          # print(img_type)
          if img_type is None:
              incompatible_images.append(filepath.absolute().as_uri().replace('file://', ''))
              print(f"{filepath} is not an image")
          elif img_type not in img_type_accepted_by_tf:
              incompatible_images.append(filepath.absolute().as_uri().replace('file://', ''))
              print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
  [os.remove(x) for x in incompatible_images]


# https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
def load_data():
  train_ds = keras.utils.image_dataset_from_directory(TRAINING_DIR,
      labels="inferred",
      label_mode="int",
      class_names=['partial_mask', 'with_mask', 'without_mask'],
      color_mode="grayscale",
      batch_size=64,
      image_size=(256, 256),
      shuffle=True,
      seed=42,
      interpolation="bicubic",
      follow_links=False,
      # validation_split=0.2,
      # subset="training"
      # crop_to_aspect_ratio=False,
      # pad_to_aspect_ratio=False,
      # data_format=None,
      # verbose=True
  )

  val_ds = keras.utils.image_dataset_from_directory(VALIDATION_DIR,
      labels="inferred",
      label_mode="int",
      class_names=['partial_mask', 'with_mask', 'without_mask'],
      color_mode="grayscale",
      batch_size=64,
      image_size=(256, 256),
      shuffle=True,
      seed=42,
      interpolation="bicubic",
      follow_links=False,
      # validation_split=0.2,
      # subset="validation"
      # crop_to_aspect_ratio=False,
      # pad_to_aspect_ratio=False,
      # data_format=None,
      # verbose=True
  )

  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  return train_ds, val_ds

def create_model():
  # https://www.tensorflow.org/tutorials/images/classification
  model = keras.models.Sequential([
      keras.layers.Rescaling(1./255, input_shape = input_shape),
      keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
      keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
      keras.layers.MaxPooling2D(2),
      keras.layers.Conv2D(128, 5, activation="relu", padding="same"),
      keras.layers.Conv2D(128, 5,  activation="relu", padding="same"),
      keras.layers.MaxPooling2D(2),
      keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
      keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
      keras.layers.MaxPooling2D(2),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation="relu"),
      # keras.layers.Dropout(0.4),
      keras.layers.Dense(32, activation="relu"),
      # keras.layers.Dropout(0.5), dropout on last layer not usually done
      keras.layers.Dense(3, activation="softmax")
  ])
  model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  return model


def create_vgg16_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                     weights='imagenet',
                     input_shape=input_shape)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(256, activation='relu')(top_model)
    top_model = Dropout(0.4)(top_model)
    top_model = Dense(128, activation='relu')(top_model)
    # top_model = Dropout(0.5)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def get_class_weights(train_ds):
  img_labels = []
  for images, labels in  train_ds.take(-1):
    for label in labels:
        img_labels.append(label.numpy())
  from collections import Counter
  class_counts = Counter(img_labels)
  total_train_data_size = np.array(list(class_counts.values())).sum()
  # fraction_sum = (total_train_data_size/class_counts[0]) + (total_train_data_size/class_counts[1])\
  #      + (total_train_data_size/class_counts[2])
  return {\
          0: total_train_data_size/(class_counts[0] * 3), \
          1: total_train_data_size/(class_counts[1] * 3), \
          2: total_train_data_size/(class_counts[2]* 3)\
        }

def train_and_save_cnn_model(train_ds, val_ds):
    cnn_model_name = 'cnn_model_grayscale'
    model = create_model()
    cnn_checkpoint_filepath = f'Models/checkpoints/checkpoint.{cnn_model_name}_{round(time.time())}.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=cnn_checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_freq = 'epoch',
        save_best_only=True
    )
    model.summary()
    epochs = 10
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      class_weight = get_class_weights(train_ds),
      callbacks=[model_checkpoint_callback]
    )
    save_time = round(time.time())
    with open(f'Models/history/{cnn_model_name}_{epochs}_epochs_{save_time}.json', 'wb') as file_pi:
      joblib.dump(history.history, file_pi)
    model.save(f'Models/{cnn_model_name}_{epochs}_epochs_{save_time}.keras')

def train_and_save_vgg_model(train_ds, val_ds):
    model_name = 'vgg16_pretrained'
    checkpoint_filepath = f'Models/checkpoints/checkpoint.{model_name}_{round(time.time())}.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      monitor='val_accuracy',
      mode='max',
      save_freq = 'epoch',
      save_best_only=True)
    # vgg16_model = create_vgg16_model(input_shape= input_shape, n_classes= 3, optimizer= 'adam', fine_tune = 18)
    vgg16_model = keras.models.load_model('Models\\vgg16_pretrained_2_epochs_1718540002.keras')
    vgg16_model.summary()
    vgg_epochs = 8
    vgg16_history = vgg16_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=vgg_epochs,
    class_weight = get_class_weights(train_ds),
    callbacks=[model_checkpoint_callback]
    )
  
    save_time = round(time.time())
    with open(f'Models/history/{model_name}_{vgg_epochs}_epochs_{save_time}.json', 'wb') as file_pi:
      joblib.dump(vgg16_history.history, file_pi)
    vgg16_model.save(f'Models/{model_name}_{vgg_epochs}_epochs_{save_time}.keras')

def main():
  # clean_data()
  tf.random.set_seed(42)
  keras.utils.set_random_seed(42)
  train_ds, val_ds = load_data()

  train_and_save_cnn_model(train_ds, val_ds)
  # train_and_save_vgg_model(train_ds, val_ds)


main()
