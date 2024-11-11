#Sağlıkta Derin Öğrenme Uygulamaları
#X Ray Görüntüleri ile Akciğer Kanseri Tespiti

#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D ,Flatten,Dropout,BatchNormalization
#Convolutional neural network ne için kullanılır?
#Örneğin, yinelemeli sinir ağları genellikle doğal dil işleme ve konuşma tanıma için kullanılırken, evrişimli sinir ağları (ConvNets veya CNN'ler) daha çok sınıflandırma ve bilgisayarlı görme görevleri için kullanılır.
#Dense,Conv2D,Convolutional neural network da kullandığımız classification layerları.
#Flatten ile Convolutional çıktılarını düzleştirip vektör haline getiricez.
#Overfitting engellemek için Dropout Layer kullanıcaz.
#Drop out Neural networkde bulunan bağlantıların bazılarını random şekilde açıp kapatmamıza yarıyor.
#BatchNormalizasyon layerlar arasında yapılan normalizasyon işlemi
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 #open cv 
import os
#veri yüklemek için

from tqdm import tqdm
#for döngüsüne sayaç ekliyor.

# %% load data
#labelsleri tanımlayalım

labels = ["PNEUMONIA","NORMAL"]
#Görüntüleri yükledikten sonra boyutlarını ayarlayalım.
img_size = 150 #150 *150

def get_training_data(data_dir):
    data = []
    for label in labels : #NORMAL
        path = os.path.join(data_dir,label)
        #data_dir ve label olan iki stringi birleştirecek.
        class_num = labels.index(label)
        #PNEUMONIA -> 0 ,NORMAL ->1
        #Normalin içinde dolaşıp her bir görseli import edip boyutunu değiştirip datanın içerisine kaydedelim.
        for img in tqdm(os.listdir(path)): #["IM-0003-0001.jpeg",....]
             try :
                 #goruntuyu oku ve işle
                 img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) #grayscale siyah beyaz okumasını sağlıyor.
                 if img_arr is not None:
                     print(f"Error: {path} dosyası okunamadı.")
                     continue
                 #goruntuyu yyeniden boyutlandır
                 resized_arr = cv2.resize(img_arr,(img_size,img_size))
                 #bu işlemleri 250*250 boyutu içinde de dene ,başarım değişecek derin öğrenme modeli daha yavaş çalışacak.
                 
                 #veriyi listeye ekle.
                 data.append([resized_arr,class_num])
             except Exception as e:
                 print("Error: ",e)
    return np.array(data,dtype = object)

train = get_training_data("LungCancerDetectionwithX-RayImages/akciger_kanseri_tespiti_data/chest_xray/chest_xray/train")
test = get_training_data("akciger_kanseri_tespiti_data/chest_xray/chest_xray/test")
val= get_training_data("akciger_kanseri_tespiti_data/chest_xray/chest_xray/val")

# %% data visualization ve preprocessing
#target değişkenlerin sayısını tespit edelim.
l = []
for i in train:
    if(i[1] == 0):
        l.append("PNEUMONIA")
    else:
        l.append("NORMAL")

sns.countplot(x=l)

x_train = []
y_train = [] #target

x_test = []
y_test = []

x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature) 
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)


plt.figure()
plt.imshow(train[0][0], cmap = "gray")
plt.title(labels[train[0][1]])

plt.figure()
plt.imshow(train[-1][0], cmap = "gray")
plt.title(labels[train[-1][1]])

# normalization: [0, 1]
# [0, 255] / 255 = [0, 1]
x_train = np.array(x_train)/255
x_test = np.array(x_test)/255
x_val = np.array(x_val)/255

# (5216, 150, 150) -> (5216, 150, 150, 1)
#Formatı düzelttik.
x_train = x_train.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)

y_train = np.array(y_train)
y_test= np.array(y_test)
y_val = np.array(y_val)


# %% data aaugmentation
#Veri artırımı eğitim verisini yapay olarak artırmak için çeşitli dönüşümler yaparak mevcut veri sayısını artırıp overfittingi engeliiyoruz.
datagen = ImageDataGenerator(
    featurewise_center = False, # veri setinin genel ortalamasini 0 yapar
    samplewise_center = False, # her bir ornegin ortalamasini 0 yapar
    featurewise_std_normalization = False, # veriyi verinin std bolme
    samplewise_std_normalization = False, # her bir ornegi kendi std sapmasina bolme islemi
    zca_whitening=False, # zca beyazlatma yontemi, korelasyonu azaltma,model daha hızlı öğrenir.
    rotation_range = 30, # resimleri x derece rastgele dondurur
    zoom_range = 0.2, # rasgele yakinlastirma islemi
    width_shift_range = 0.1, # resimleri yatay olarak rastgele kaydirma
    height_shift_range = 0.1, # resimleri dikey olarak rastgele kaydirir
    horizontal_flip = True, # resimleri rastgele yatay olarak cevirir
    vertical_flip = True # resimleri rastgele dikey olarak cevirir
    )
datagen.fit(x_train)


# %% create deep learning model and train
#Akciğer kanseri var mı yok mu buna bakıyoruz.
#Convolutional neural network uygulanacak plan:
"""
Layerları yazalım.
Feature Extraction Blok:
   * Genellikle bu yöntemi izliyoruz.
    (con2d - Normalizasyon - MaxPooling)
   * Model başarımını arttırmak için derinleşitiriyoruz.
    (con2d - dropout - Normalizasyon - MaxPooling)
    (con2d - Normalizasyon - MaxPooling)
    (con2d - dropout - Normalizasyon - MaxPooling)
    (con2d - dropout - Normalizasyon - MaxPooling)
Classification Blok:
    flatten(düzleştirme) - Dense - Dropout - Dense (output)
Compiler: optimizer (rmsprop), loss (binary cross ent.), metric (accuracy)
"""
#ilk katman olduğu için input shape i belirliyoruz.
#filtre sayısı gittikçe artmalı.
#droput:0.1 yüzde 10 luk bir kapama olacak.
#Aşırı ezberleme durumu varsa dropout eklenebilir.
#strides Layerde bulunan 3*3lük filtrede image üzerinde kaç adım atlayarak gidecek?
#Her adımda 1 piksel kayıyor.
#MaxPool: Görüntü üzerinde 2*2 lik bir pooling uygulayacak.Yani görüntü üzerinde 2 piksel kayacak.
#Padding:Giriş verisinde görüntünün kenarlarına 0 ekleyerek genişletiyor.Classification,İşlem sırasında boyut kaybını engellemek için.
#same 0 eklenerek girdi ve çıktının boyutunun aynı olmasını sağlıyor.
model = Sequential()
model.add(Conv2D(128, (7,7), strides=1, padding="same", activation="relu", input_shape=(150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding="same"))

model.add(Conv2D(64, (5,5), strides = 1, padding = "same", activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = "same"))
#32 filtre olacak her bir filtrenin boyutu 3*3

model.add(Conv2D(32, (3,3), strides = 1, padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding="same"))

#Classification Blok:
#işlemler sonucunda image matrisi ortaya çıkıyor bunu düzleştirip iki boyut haline getirmek için flatten kullanıyoruz.
#Headen layerler :units,relu training aşamasında kolay türevlenebilir.
#Dense layer output layerı.units = 1 demek 1 output.
model.add(Flatten())
model.add(Dense(units = 128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()
#Model training
#Öğrenme gerçekleştiğinde rate reduction azalacak öğrenme oranımız artacak.
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience = 2, verbose = 1, factor = 0.3, min_lr = 0.000001)
epoch_number = 3 # 15 öğrenme azalıyor 15 i dene.Veya test veri seti sayısını arttır.
history = model.fit(datagen.flow(x_train, y_train, batch_size = 32), epochs = epoch_number, validation_data = datagen.flow(x_test, y_test), callbacks = [learning_rate_reduction])

print("Loss of Model: ", model.evaluate(x_test, y_test)[0])
print("Accuracy of Model: ", model.evaluate(x_test, y_test)[1]*100)
# %% evaluation
epochs = [i for i in range(epoch_number)]

fig, ax = plt.subplots(1,2)

train_acc = history.history["accuracy"]
train_loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]
#0 1.satır 1. sütun
ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
ax[0].plot(epochs, val_acc, "ro-", label = "Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "go-", label="Training Loss")
ax[1].plot(epochs, val_loss, "ro-", label = "Validation Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")



epochs = [i for i in range(epoch_number)]

fig, ax = plt.subplots(1,2)

train_acc = history.history["accuracy"]
train_loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
ax[0].plot(epochs, val_acc, "ro-", label = "Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "go-", label="Training Loss")
ax[1].plot(epochs, val_loss, "ro-", label = "Validation Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")





