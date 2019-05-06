import numpy as np
import keras

from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Dropout, Add, Lambda
from keras import regularizers
from keras import backend as K

from keras.engine.topology import Layer


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	COVAREP_X_FORMANT = Input(shape = (418,))
	facial_X_pose = Input(shape = (1542,))
	gaze_X_action = Input(shape = (1040,))

	X_gender = Input(shape = (1,))


	COVAREP_X_FORMANT_shortened = Dense(460, activation = 'relu')(COVAREP_X_FORMANT)

	facial_X_pose_shortened = Dense(900, activation = 'relu')(facial_X_pose)
	facial_X_pose_shortened = Dense(460, activation = 'relu')(facial_X_pose_shortened)

	gaze_X_action_shortened = Dense(600, activation = 'relu')(gaze_X_action)
	gaze_X_action_shortened = Dense(460, activation = 'relu')(gaze_X_action_shortened)


	B = Concatenate(axis = 1)([COVAREP_X_FORMANT_shortened, facial_X_pose_shortened, gaze_X_action_shortened])
	#B = K.stack([COVAREP_X_FORMANT_shortened, facial_X_pose_shortened, gaze_X_action_shortened], axis = 2)

	P = Dense(200, activation = 'tanh')(B)

	alpha = Dense(3, activation = 'softmax')(P)

	F = Lambda(lambda x : alpha[:,0:1]*COVAREP_X_FORMANT_shortened + alpha[:,1:2]*facial_X_pose_shortened + alpha[:,2:3]*gaze_X_action_shortened)(alpha)

	Y = Concatenate(axis = -1)([F, X_gender])

	Y = Dense(210, activation = 'relu')(Y)		#, kernel_regularizer = regularizers.l2(0.01)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(63, activation = 'relu')(Y)
	Y = Dropout(rate = 0.2)(Y)

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [COVAREP_X_FORMANT, facial_X_pose, gaze_X_action, X_gender], outputs = Y)

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()