from matplotlib import pyplot as plt
import glob, os
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

model = keras.models.load_model('Models/vgg16_pretrained_3_epochs.keras')
test_ds = keras.utils.image_dataset_from_directory("DataSet\\FaceMask_Kaggle_test\\FaceMask_Kaggle_test/",
      labels=None,
      class_names=None,
      color_mode="rgb",
      batch_size=64,
      image_size=(256, 256),
      shuffle=True,
      seed=42,
      interpolation="bilinear",
      follow_links=False,
      # validation_split=0.2,
      # subset="validation"
      # crop_to_aspect_ratio=False,
      # pad_to_aspect_ratio=False,
      # data_format=None,
      # verbose=True
  )
vgg_pred = model.predict(test_ds)
vgg_pred_classes = np.argmax(vgg_pred, axis=1)

# vgg_pred_classes[vgg_pred_classes == 0] = 'partial_mask'
# vgg_pred_classes[vgg_pred_classes == 1] = 'with_mask'
# vgg_pred_classes[vgg_pred_classes == 2] = 'without_mask'
labels_map = {0: 'partial_mask', 1: 'with_mask', 2: 'without_mask'}
import pandas as pd
vgg_pred_classes_names = [labels_map[x] for x in vgg_pred_classes]
incorrect_paths_map = {
  'FaceMask_Kaggle_test/112.jpeg': 'FaceMask_Kaggle_test/112jpeg',
  'FaceMask_Kaggle_test/130.jpeg': 'FaceMask_Kaggle_test/130jpeg',
  'FaceMask_Kaggle_test/136.jpeg': 'FaceMask_Kaggle_test/136jpeg',
  'FaceMask_Kaggle_test/142.jpeg': 'FaceMask_Kaggle_test/142jpeg',
  'FaceMask_Kaggle_test/266.jpeg': 'FaceMask_Kaggle_test/266jpeg',
  'FaceMask_Kaggle_test/342.jpeg': 'FaceMask_Kaggle_test/342jpeg',
  'FaceMask_Kaggle_test/383.jpeg': 'FaceMask_Kaggle_test/383jpeg',
  'FaceMask_Kaggle_test/463.jpeg': 'FaceMask_Kaggle_test/463jpeg',
  'FaceMask_Kaggle_test/470.jpeg': 'FaceMask_Kaggle_test/470jpeg',
  'FaceMask_Kaggle_test/552.jpeg': 'FaceMask_Kaggle_test/552jpeg',
  'FaceMask_Kaggle_test/623.jpeg': 'FaceMask_Kaggle_test/623jpeg'
  }

img_paths = [x.replace("DataSet\\FaceMask_Kaggle_test\\", "") for x in test_ds.file_paths]
img_paths_corrected = []
for x in img_paths:
  if x in incorrect_paths_map:
    img_paths_corrected.append(incorrect_paths_map[x])
  else:
    img_paths_corrected.append(x)

df = pd.DataFrame({'img_path': img_paths_corrected, 'label':vgg_pred_classes_names })
df.to_csv(f'Kaggle_Submissions/vgg_16_3_epochs_{round(time.time())}.csv', index=False)

