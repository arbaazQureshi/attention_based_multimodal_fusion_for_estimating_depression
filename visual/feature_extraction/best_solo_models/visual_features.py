import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout
import keras

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 2482,))

	Y = CuDNNLSTM(256, name = 'lstm_cell')(X)
	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(20, activation = 'relu')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model