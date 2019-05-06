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

	X = Input(shape = (52,))
	#X_gender = Input(shape = (1,))

	#Y = Concatenate(axis = -1)([X, X_gender])
	
	#Y = Dense(29, activation = 'relu')(Y)
	Y = Dense(31, activation = 'relu')(X)
	Y = Dropout(rate = 7/27)(Y)

	Y = Dense(12, activation = 'relu')(Y)
	Y = Dropout(rate = 3/11)(Y)

	Y = Dense(1, activation = None)(Y)

	#model = Model(inputs = [X, X_gender], outputs = Y)
	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model