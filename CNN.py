import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model

R = 6   # 6  # number of rows (board height)
C = 7  # 7  # number of columns (board width)
win_num = 4  # 4  # number of symbol in line to win
if win_num > C or win_num > R:
    raise ValueError('win_num is larger than board dimension!')
save_title = "R_" + str(R) + "_C_" + str(C) +"_win_num_" + str(win_num) + "_"
X = np.load('CNN_data/' + save_title + 'data.npy')
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
col = np.load('CNN_data/' + save_title + 'col.npy')
y = to_categorical(col)
N = col.shape[0]
N_split = int(N*3/5)  # 60 games for training, 40 games for testing  # todo
X_train = X[:N_split, :, :, :]
y_train = y[:N_split, :]
X_test = X[N_split:, :, :, :]
y_test = y[N_split:, :]


# create model  # todo
model = Sequential()
model.add(Conv2D(R*C, kernel_size=win_num, activation='relu', input_shape=(R, C, 1)))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Conv2D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(R*C, activation='relu'))
# model.add(Dense(R*C, activation='relu'))
model.add(Dense(C, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('CNN_model/' + save_title + 'curve.png')


# creates a HDF5 file 'my_model.h5'
model.save('CNN_model/' + save_title + 'model.h5')
del model
model = load_model('CNN_model/' + save_title + 'model.h5')

print(model.predict(X_test[:4]))
print(y_test[:4])


plt.show()
