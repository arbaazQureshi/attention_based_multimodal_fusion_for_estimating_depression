import numpy as np
import keras

from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Dropout
from keras import regularizers


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (3000,))
	X_gender = Input(shape = (1,))

	Y = Concatenate(axis = -1)([X, X_gender])
	
	Y = Dense(1930, activation = 'relu')(Y)
	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(420, activation = 'relu')(Y)
	Y = Dropout(rate = 0.25)(Y)

	Y = Dense(75, activation = 'relu')(Y)
	Y = Dropout(rate = 0.2)(Y)

	#Y = Dense(40, activation = 'relu')(Y)
	#Y = Dropout(rate = 0.15)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model