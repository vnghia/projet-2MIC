from keras.datasets import mnist
from keras.layers import Input, Activation
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add
from keras.models import Model
from keras import optimizers, losses

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def GetData():
  (x_train, _), (x_test, _) = mnist.load_data()
  sigma_noise=30

  y_train = x_train + sigma_noise*np.random.randn(x_train.shape[0],x_train.shape[1],x_train.shape[2])
  y_test = x_test + sigma_noise*np.random.randn(x_test.shape[0],x_test.shape[1],x_test.shape[2])
  return x_train, x_test, y_train, y_test

def PreProcess(x_train, x_test, y_train, y_test):
  x_train_ext = np.expand_dims(x_train,3)
  y_train_ext = np.expand_dims(y_train,3)
  x_test_ext = np.expand_dims(x_test,3)
  y_test_ext = np.expand_dims(y_test,3)

  return x_train_ext, y_train_ext, x_test_ext, y_test_ext

def model_simple():
  init = Input(shape=(None, None,1))

  x = Convolution2D(16, (3, 3), activation='relu', padding='same')(init)
  x = MaxPooling2D((2, 2))(x)
  x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2))(x)
  x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
  x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D()(x)
  x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D()(x)
  x = Convolution2D(1, (3, 3), activation='relu', padding='same')(x)

  NN = Model(init, x)
  return NN

def TrainModel(model, loss, optim, epochs, batch_size, input_data, target_data, input_test_data, target_test_data):
  model.summary()

  model.compile(loss=loss, optimizer=optim, metrics=['mse'])

  out_train = model.fit(input_data, target_data, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(input_test_data, target_test_data))

  score = model.evaluate(input_test_data, target_test_data, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  
  loss_train = out_train.history['loss']
  loss_test = out_train.history['val_loss']

  plt.figure(2)
  plt.plot(loss_train,label='training')
  plt.plot(loss_test,label='validation')
  plt.title('Loss')
  plt.legend()
  plt.show()

  return model

def DisplayResult(number_image, target_data, input_data, result_data):
  fig, axs = plt.subplots(3, number_image)
  fig.suptitle("Afficher quelques images")

  for i in range(number_image):
    axs[0,i].imshow(target_data[i], interpolation = "nearest")
    axs[1,i].imshow(input_data[i], interpolation = "nearest")
    axs[2,i].imshow(result_data[i].reshape(28, 28), interpolation = "nearest")

  axs.flat[0].set(ylabel = "Originales")
  axs.flat[number_image].set(ylabel = "Avec du bruit")
  axs.flat[2*number_image].set(ylabel = "Résultats")

  for ax in fig.get_axes():
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.label_outer()

  plt.show()

def SNR(x_ref,x):
  x_ref_vect = x_ref.flatten()
  x_vect = x.flatten()
  res=-20*np.log10(np.linalg.norm(x_ref_vect-x_vect)/np.linalg.norm(x_vect)+1e-15)
  return res

if __name__ == "__main__":
  x_train, x_test, y_train, y_test = GetData()
  x_train_ext, y_train_ext, x_test_ext, y_test_ext = PreProcess(x_train, x_test, y_train, y_test)
  model = TrainModel(model_simple(), losses.mse, optimizers.Adam(), 10, 128, y_train_ext, x_train_ext, y_test_ext, x_test_ext)
  # model = load_model('model.h5')
  y_result_ext = model.predict(y_test_ext, verbose = 0)
  DisplayResult(10, x_test, y_test, y_result_ext)
  print("Signal to Noise Ratio:", SNR(x_test_ext, y_result_ext))