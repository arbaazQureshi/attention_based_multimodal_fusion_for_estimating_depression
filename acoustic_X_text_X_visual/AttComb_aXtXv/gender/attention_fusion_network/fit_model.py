from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="1"



if(path.exists('training_progress.csv')):
	progress = np.loadtxt('training_progress.csv', delimiter=',').tolist()

else:
	progress = []

if(path.exists('optimal_weights.h5')):
	model = load_model()
	model.load_weights('optimal_weights.h5')
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

else:
	model = load_model()
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

training_COVAREP_X_FORMANT, training_facial_X_pose, training_gaze_X_action, training_transcript, training_Y, training_X_gender = load_training_data()
dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_transcript, dev_Y, dev_X_gender = load_development_data()

	
min_rmse_dev = 10000
min_mae_dev = 10000

current_epoch_number = 1
total_epoch_count = 1


no_of_epochs = 10000

m = training_COVAREP_X_FORMANT.shape[0]
batch_size_list = list(range(1, m))

print("\n\n")

min_epoch = None


while(current_epoch_number < no_of_epochs):
	
	print(str(total_epoch_count)*30)
	print(no_of_epochs - current_epoch_number, "epochs to go.")

	#batch_size = random.choice(batch_size_list)
	batch_size = m
	#batch_size = 21
	print("Batch size is", batch_size)
	
	hist = model.fit([training_COVAREP_X_FORMANT, training_facial_X_pose, training_gaze_X_action, training_X_gender, training_transcript], training_Y, batch_size = batch_size, epochs = 1)
	mse_train = hist.history['loss'][-1]
	mse_dev, mae_dev = model.evaluate([dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_X_gender, dev_transcript], dev_Y, batch_size = dev_COVAREP_X_FORMANT.shape[0])

	print(mse_dev, mae_dev)

	if(mse_dev < min_rmse_dev):

		min_rmse_dev = mse_dev
		min_mae_dev = mae_dev
		min_epoch = current_epoch_number

		model.save_weights('optimal_weights.h5')
		print("SAVING THE WEIGHTS!"+"*"*5000+"\n\n")

		np.savetxt('learner_params.txt', np.array([min_rmse_dev, min_mae_dev, min_epoch, mse_train]), fmt='%.4f')

	current_epoch_number = current_epoch_number + 1


	progress.append([total_epoch_count, mse_train, mse_dev, mae_dev])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')

	total_epoch_count = total_epoch_count + 1
	print("\n\n")