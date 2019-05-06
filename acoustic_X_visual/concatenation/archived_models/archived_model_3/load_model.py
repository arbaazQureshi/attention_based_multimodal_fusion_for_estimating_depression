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

	X = Input(shape = (387,))
	X_gender = Input(shape = (1,))

	Y = Concatenate(axis = -1)([X, X_gender])

	Y = Dense(200, activation = 'relu')(Y)		#, kernel_regularizer = regularizers.l2(0.01)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(79, activation = 'relu')(Y)
	Y = Dropout(rate = 7/27)(Y)

	Y = Dense(28, activation = 'relu')(Y)
	Y = Dropout(rate = 3/11)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model