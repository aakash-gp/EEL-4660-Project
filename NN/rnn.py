import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Remove Warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load data


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], -1, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], -1, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# RNN Starts
model = Sequential()
model.add(SimpleRNN(100, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))



# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)

#print model Details
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("RNN Error: %.2f%%" % (100-scores[1]*100))
model.summary()

#save model
model.save('modelrnn.h5')


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
