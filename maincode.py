from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model =Sequential()
model.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(init="uniform",activation="sigmoid",units=1))
model.compile(loss= "binary_crossentropy",optimizer='adam',metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen =ImageDataGenerator(rescale=1./255)
x_train = train_datagen.flow_from_directory(r"C:\Users\konatham suresh babu\Downloads\dataset\lung_cancer\training_set",target_size=(64,64),batch_size=32,class_mode='binary')
x_test = train_datagen.flow_from_directory(r"C:\Users\konatham suresh babu\Downloads\dataset\lung_cancer\test_set",target_size=(64,64),batch_size=32,class_mode='binary')
model.fit_generator(x_train,steps_per_epoch=11,epochs=1,validation_data=x_test,validation_steps=3)
model.save("rnn.h5") 
#secound type of prediction
from keras.models import load_model
from keras.preprocessing import image
import numpy as np 
model = load_model(r"C:\Users\konatham suresh babu\rnn.h5")
img = image.load_img(r"C:\Users\konatham suresh babu\Pictures\nocancer.jpg",target_size= (64,64))
x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
x.shape
pred = model.predict_classes(x)
print(pred)
import cv2
import numpy as np
import matplotlib.pyplot as plt
img =cv2.imread(r"C:\Users\konatham suresh babu\Pictures\nocancer.jpg")
edges =cv2.Canny(img,100,200)
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(edges,cmap="gray")
plt.show()   