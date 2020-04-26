from datetime import datetime
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D, BatchNormalization, Subtract
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses

import numpy as np
rng = np.random.default_rng()
indexs = []
number_image = 10
epochs = 10 # 10
batch_size = 128 # 128

import matplotlib.pyplot as plt

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

def model_lineaire():
  init = Input(shape=(None, None,1))

  x = Convolution2D(16, (3, 3), padding='same')(init)
  x = Convolution2D(32, (3, 3), padding='same')(x)
  x = Convolution2D(64, (3, 3), padding='same')(x)
  x = Convolution2D(32, (3, 3), padding='same')(x)
  x = Convolution2D(16, (3, 3), padding='same')(x)
  x = Convolution2D(1, (3, 3), padding='same')(x)

  NN = Model(init, x)
  return NN

def model_dncnn():
  init = Input(shape=(None, None,1))

  x = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(init)
  x = Activation('relu')(x)
  for i in range(15):
    x = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
    x = Activation('relu')(x)

  x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
  x = Subtract()([init, x])

  NN = Model(init, x)
  return NN

def TrainModel(model, loss, optim, epochs, batch_size, input_data, target_data, input_test_data, target_test_data, model_name = "model_simple"):
  model.summary()

  model.compile(loss=loss, optimizer=optim, metrics=['mse'])

  # Define the Keras TensorBoard callback.
  logdir="logs/fit/" + model_name + "/epochs_" + str(epochs) + "_batch_" + str(batch_size) 
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

  out_train = model.fit(input_data, target_data, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(input_test_data, target_test_data), callbacks=[tensorboard_callback])

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
  plt.savefig('train_loss.png')
  plt.show()

  return model


def DisplayResult(number_image, target_data, input_data, result_data, filename = "result.png"):
  fig, axs = plt.subplots(3, number_image)

  for i in range(number_image):
    axs[0,i].imshow(target_data[indexs[i]], interpolation = "nearest", cmap='gray_r')
    axs[1,i].imshow(input_data[indexs[i]], interpolation = "nearest", cmap='gray_r')
    axs[2,i].imshow(result_data[indexs[i]], interpolation = "nearest", cmap='gray_r')

  axs.flat[0].set(ylabel = "Originales")
  axs.flat[number_image].set(ylabel = "Avec du bruit")
  axs.flat[2*number_image].set(ylabel = "Résultats")

  for ax in fig.get_axes():
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.label_outer()

  plt.savefig(filename)
  plt.show()

def DisplayWithoutResult(number_image, target_data, input_data):
  fig, axs = plt.subplots(2, number_image)

  for i in range(number_image):
    axs[0,i].imshow(target_data[indexs[i]], interpolation = "nearest", cmap='gray_r')
    axs[1,i].imshow(input_data[indexs[i]], interpolation = "nearest", cmap='gray_r')

  axs.flat[0].set(ylabel = "Originales")
  axs.flat[number_image].set(ylabel = "Avec du bruit")

  for ax in fig.get_axes():
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.label_outer()

  plt.savefig('without_result.png')
  plt.show()

def SNR(x_ref,x):
  x_ref_vect = x_ref.flatten()
  x_vect = x.flatten()
  res=-20*np.log10(np.linalg.norm(x_ref_vect-x_vect)/np.linalg.norm(x_vect)+1e-15)
  return res

if __name__ == "__main__":
  x_train, x_test, y_train, y_test = GetData()
  indexs = rng.choice(x_test.shape[0], size=number_image, replace=False)
  x_train_ext, y_train_ext, x_test_ext, y_test_ext = PreProcess(x_train, x_test, y_train, y_test)
  DisplayWithoutResult(number_image, x_test, y_test)

  # Model Simple
  #model = model_simple()
  #model = TrainModel(model, losses.mse, optimizers.Adam(), epochs, batch_size, y_train_ext, x_train_ext, y_test_ext, x_test_ext)
  #y_result = np.squeeze(model.predict(y_test_ext, verbose = 0)) 
  #DisplayResult(number_image, x_test, y_test, y_result, "model_simple.png")
  #print("Signal to Noise Ratio:", SNR(x_test, y_result))

  # Model Lineare
  #model = model_lineaire()
  #model = TrainModel(model, losses.mse, optimizers.Adam(), epochs, batch_size, y_train_ext, x_train_ext, y_test_ext, x_test_ext, "model_lineaire")
  #y_result = np.squeeze(model.predict(y_test_ext, verbose = 0)) 
  #DisplayResult(number_image, x_test, y_test, y_result, "model_lineaire.png")
  #print("Signal to Noise Ratio:", SNR(x_test, y_result))

  # Model DnCNN
  model = model_dncnn()
  model = TrainModel(model, losses.mse, optimizers.Adam(), epochs, batch_size, y_train_ext, x_train_ext, y_test_ext, x_test_ext, "model_dncnn")
  y_result = np.squeeze(model.predict(y_test_ext, verbose = 0)) 
  DisplayResult(number_image, x_test, y_test, y_result, "model_dncnn.png")
  print("Signal to Noise Ratio:", SNR(x_test, y_result))