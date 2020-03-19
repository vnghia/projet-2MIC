# Lancer cette cette commande dans le terminal avant de lancer vos scripts python,
# cela permets d'utiliser le GPU (carte graphique):
# source activate GPU
#
# Lancer ensuite vos code avec: python main_incomplet.py
#
# Ou alors lancer un terminal Python avec: ipython
# puis lancer votre script avec: run main_incomplet.py
# Cela vous permettra de debeuger et de taper vos commandes sans avoir
# a relancer tout le script.


# Librairies pour réseau de neurones
from keras.datasets import mnist # hand written digits
from keras.models import Model, load_model

import numpy as np
import matplotlib.pyplot as plt          

# Commenter à l'INSA
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Load data
# On recupere les images
# x: Matrices de taille 28x28 avec des valeurs de 0 à 255.
# y: chiffre auquel correspond l'image. On utilise pas y ici.
(x_train, _), (x_test, _) = mnist.load_data()


# Degradation
sigma_noise=30
sigma_flou=2 # Le flou va s'etaler sur environ 3*sigma_flou pixels

# ### Choisir entre l'ajout de bruit et l'ajout de flou
# # On ajoute un flou stationaire et du bruit gaussien
# n1,n2=x_train.shape[1],x_train.shape[2]
# lin1=np.linspace(-n1/2,n1/2,n1)
# lin2=np.linspace(-n2/2,n2/2,n2)
# XX,YY=np.meshgrid(lin1,lin2)
# G=np.exp(-(XX**2+YY**2)/(sigma_flou**2))
# G/=np.sum(G)
# Ghat=np.fft.fft2(np.fft.fftshift(G))

# y_train = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2]))
# for k in range(x_train.shape[0]):
#   # Convolution en u et v = F^{-1}(F(u)*F(v)), ou * est le produit terme à terme et F est l'opérateur de tranformée de fourier
#     y_train[k]=np.fft.ifft2(np.fft.fft2(x_train[k])*Ghat)+sigma_noise*np.random.randn(n1,n2) 
# y_test = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2]))
# for k in range(x_test.shape[0]):
#     y_test[k]=np.fft.ifft2(np.fft.fft2(x_test[k])*Ghat)+sigma_noise*np.random.randn(n1,n2) 

# # Inpainting
# tmp = x_train.copy()
# tmp[:,12:18,:]=0
# y_train = tmp + sigma_noise*np.random.randn(x_train.shape[0],x_train.shape[1],x_train.shape[2]) 
# tmp = x_test.copy()
# tmp[:,12:18,:]=0
# y_test = tmp + sigma_noise*np.random.randn(x_test.shape[0],x_test.shape[1],x_test.shape[2]) 

# On ajoute du bruit gaussien
y_train = x_train + sigma_noise*np.random.randn(x_train.shape[0],x_train.shape[1],x_train.shape[2]) 
y_test = x_test + sigma_noise*np.random.randn(x_test.shape[0],x_test.shape[1],x_test.shape[2]) 

# Mettre donnees en forme pour passer dans le réseau
x_train_ext = np.expand_dims(x_train,3) # ajoute une dimension a x_train à la position 3
y_train_ext = np.expand_dims(y_train,3)  
x_test_ext = np.expand_dims(x_test,3)  
y_test_ext = np.expand_dims(y_test,3) 

model = load_model('model.h5') # Pour charger le réseau model

score = model.evaluate(y_test_ext, x_test_ext, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Afficher quelques images
# TODO: evaluer le réseau sur la partie test du jeu de données (pourquoi jamais sur le train?), 
# utiliser la fonction 'predict' pour faire obtenir la sortie du réseau de neurone (model.predict)
# TODO: afficher les resultats sur la meme figure comme au debut
result = model.predict(y_test_ext, verbose = 0)
numberImage = 10
fig, axs = plt.subplots(3, numberImage)
fig.suptitle("Afficher quelques images")

for i in range(numberImage):
  axs[0,i].imshow(x_test[i], interpolation = "nearest")
  axs[1,i].imshow(y_test[i], interpolation = "nearest")
  axs[2,i].imshow(result[i].reshape(28, 28), interpolation = "nearest")

axs.flat[0].set(ylabel = "Originales")
axs.flat[numberImage].set(ylabel = "Avec du bruit")
axs.flat[2*numberImage].set(ylabel = "Résultats")

for ax in fig.get_axes():
  ax.tick_params(axis=u'both', which=u'both',length=0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  ax.label_outer()

plt.show()