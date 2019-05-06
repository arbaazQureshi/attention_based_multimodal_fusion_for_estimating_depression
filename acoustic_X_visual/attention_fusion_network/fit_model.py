from keras.models import Model, load_model

from load_model import load_model
from load_data import load_training_data, load_development_data
import keras

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="2"



if(path.exists('training_progress.csv')):
	progress = np.loadtxt('training_progress.csv', delimiter=',').tolist()

else:
	progress = []

if(path.exists('best_weights.h5')):
	model = load_model()
	model.load_weights('best_weights.h5')
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

else:
	model = load_model()
	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

training_COVAREP_X_FORMANT, training_facial_X_pose, training_gaze_X_action, training_Y, training_X_gender = load_training_data()
dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_Y, dev_X_gender = load_development_data()




if(path.exists('learner_params.txt')):
	learner_params = np.loadtxt('learner_params.txt')
	min_loss_dev = learner_params[0]
	min_mae = learner_params[1]
	prev_loss_dev = learner_params[4]
	loss_dev = [learner_params[2], learner_params[3]]

	current_epoch_number = int(learner_params[6])
	total_epoch_count = int(learner_params[7]) + 1

else:	
	min_loss_dev = 10000
	min_mae = 10000
	prev_loss_dev = 10000
	loss_dev = [10000, 10000]

	current_epoch_number = 1
	total_epoch_count = 1


increase_count = 0
increase_count_threshold = 100000
no_of_downward_epochs = 1000

m = training_COVAREP_X_FORMANT.shape[0]
batch_size_list = list(range(1, m))

print("\n\n")



while(current_epoch_number < no_of_downward_epochs):
	
	print(str(total_epoch_count)*30)
	print(no_of_downward_epochs - current_epoch_number, "epochs to go.")

	prev_loss_dev = loss_dev[0]

	#batch_size = random.choice(batch_size_list)
	batch_size = m
	#batch_size = 21
	print("Batch size is", batch_size)
	
	hist = model.fit([training_COVAREP_X_FORMANT, training_facial_X_pose, training_gaze_X_action, training_X_gender], training_Y, batch_size = batch_size, epochs = 1)

	loss_train = hist.history['loss'][-1]
	loss_dev = model.evaluate([dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_X_gender], dev_Y, batch_size = dev_COVAREP_X_FORMANT.shape[0])

	print(loss_train, loss_dev[0], loss_dev[1])

	if(loss_dev[0] < min_loss_dev):
		min_loss_dev = loss_dev[0]
		min_mae = loss_dev[1]
		model.save_weights('optimal_weights.h5')
		print("SAVING THE WEIGHTS!"+"*"*2000+"\n\n")
		increase_count = 0
		current_epoch_number = current_epoch_number + 1

	else:

		if(loss_dev[0] >= prev_loss_dev):
			increase_count = increase_count + 1

			if(increase_count >= increase_count_threshold):
				model.load_weights('best_weights.h5')
				increase_count = 0
				print("\nGOING BACK TO THE min_model!")

		else:
			increase_count = increase_count - 1

			if(increase_count < 0):
				increase_count = 0

	progress.append([total_epoch_count, loss_train, loss_dev[0], loss_dev[1]])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')
	np.savetxt('learner_params.txt', np.array([min_loss_dev, min_mae, loss_dev[0], loss_dev[1], prev_loss_dev, increase_count, current_epoch_number, total_epoch_count]), fmt='%.4f')

	total_epoch_count = total_epoch_count + 1
	print("\n\n")