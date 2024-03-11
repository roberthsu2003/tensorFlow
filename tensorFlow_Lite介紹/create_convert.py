import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense


model = Sequential([Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs,ys, epochs=500)

export_dir = 'saved_model'
tf.saved_model.save(model, export_dir)