import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout
import keras


def load_model(Tx, n_x, n, location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (Tx, n_x,))
	X_gender = Input(shape = (1,))

	Y = LSTM(n, name = 'lstm_cell')(X)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(32, activation = 'relu')(Y)
	Y = Dropout(rate = 0.2)(Y)

	Y = Dense(8, activation = 'relu')(Y)
	Y = Dropout(rate = 0.125)(Y)
	
	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model