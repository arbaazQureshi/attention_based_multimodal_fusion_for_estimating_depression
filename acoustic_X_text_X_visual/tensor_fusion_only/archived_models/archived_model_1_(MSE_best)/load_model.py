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
	transcript = Input(shape = (200,))

	X_gender = Input(shape = (1,))


	COVAREP_X_FORMANT_shortened = Dense(450, activation = 'relu')(COVAREP_X_FORMANT)

	facial_X_pose_shortened = Dense(600, activation = 'relu')(facial_X_pose)
	facial_X_pose_shortened = Dense(450, activation = 'relu')(facial_X_pose_shortened)

	gaze_X_action_shortened = Dense(570, activation = 'relu')(gaze_X_action)
	gaze_X_action_shortened = Dense(450, activation = 'relu')(gaze_X_action_shortened)

	transcript_elongated = Dense(315, activation = 'relu')(transcript)
	transcript_elongated = Dense(450, activation = 'relu')(transcript_elongated)

	B = Concatenate(axis = 1)([COVAREP_X_FORMANT_shortened, facial_X_pose_shortened, gaze_X_action_shortened, transcript_elongated])

	Y = Dense(300, activation = 'relu')(B)
	Y = Dropout(rate = 0.25)(Y)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [COVAREP_X_FORMANT, facial_X_pose, gaze_X_action, transcript, X_gender], outputs = Y)

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()