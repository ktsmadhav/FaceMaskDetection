
import glob, os
from turtle import mode
from sklearn.utils import resample
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
import pandas as pd
import PIL

model_name = 'cnn_model_grayscale_10_epochs_1718648603'#'checkpoint.cnn_model_grayscale_1718616912'#'checkpoint.cnn_model_1718604669'#'checkpoint.vgg16_pretrained_1718541013' #'vgg16_pretrained_2_epochs_1718524487'#'vgg16_pretrained_2_epochs_1718521396'#'vgg16_pretrained_2_epochs_1718517160'
model = keras.models.load_model('Models/' + model_name + '.keras')
labels_map = {0: 'partial_mask', 1: 'with_mask', 2: 'without_mask'}
img_paths = []
labels = []
for filepath in glob.glob('DataSet/FaceMask_Kaggle_test/FaceMask_Kaggle_test/**'):
    img = PIL.Image.open(filepath)
    img_resize = img.resize((256, 256), resample = PIL.Image.Resampling.BICUBIC)
    img_resize_gray = img_resize.convert(mode = 'L')
    img_paths.append(filepath)
    img_arr = np.asarray(img_resize_gray)
    # https://keras.io/examples/vision/image_classification_from_scratch/
    img_arr_exp = keras.ops.expand_dims(img_arr, 0)  # Create batch axis
    label = np.argmax(model.predict(img_arr_exp))
    labels.append(label)

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

img_paths_1 = [x.replace("DataSet/FaceMask_Kaggle_test/", "") for x in img_paths]
img_paths_2 = [x.replace("\\", "/") for x in img_paths_1]
img_paths_corrected = []
for x in img_paths_2:
  if x in incorrect_paths_map:
    img_paths_corrected.append(incorrect_paths_map[x])
  else:
    img_paths_corrected.append(x)

labels_class_names = [labels_map[x] for x in labels]
df = pd.DataFrame({'img_path': img_paths_corrected, 'label':labels_class_names })
df.to_csv(f'Kaggle_Submissions/{model_name}_{round(time.time())}.csv', index=False)

# https://www.analyticsvidhya.com/blog/2023/12/grad-cam-in-deep-learning/