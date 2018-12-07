from keras.models import load_model
import numpy as np
import cv2
import cv2
from keras.datasets import mnist
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_train, y_train), (X_test, y_test) = mnist.load_data()


model1 = load_model('modeldnn.h5')
model3 = load_model('modelcnn.h5')

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



classes = model1.predict_classes(np.reshape(X_train[0],[1, 1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[0],[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()



classes = model1.predict_classes(np.reshape(X_train[1],[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[1],[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.show()



classes = model1.predict_classes(np.reshape(X_train[2],[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[2],[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.show()


classes = model1.predict_classes(np.reshape(X_train[3],[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[3],[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()



 
img = cv2.imread('1.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()




img = cv2.imread('2.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('3.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('4.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('5.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = cv2.imread('6.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = cv2.imread('7.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('8.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('9.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('10.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = cv2.imread('11.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()



img = cv2.imread('12.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('13.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('14.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('15.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('16.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()



img = cv2.imread('17.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = cv2.imread('18.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()



img = cv2.imread('19.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('20.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()


img = cv2.imread('21.png') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim = (28, int(28))
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
classes = model1.predict_classes(np.reshape(img,[1,1,28,28]))
print ("DNN: ", classes)
classes = model3.predict_classes(np.reshape(img,[1,1,28,28]))
print ("CNN: ", classes, "\n")
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()