import numpy as np
import sklearn.metrics

from load_data import load_development_data
from load_model import load_model


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

	model = load_model()
	model.load_weights('optimal_weights.h5')

	dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_transcript, dev_Y, dev_X_gender = load_development_data()

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mean_absolute_error'])

	dev_Y_hat = model.predict([dev_COVAREP_X_FORMANT, dev_facial_X_pose, dev_gaze_X_action, dev_transcript, dev_X_gender])

	dev_Y = np.array(dev_Y)
	dev_Y_hat = dev_Y_hat.reshape((dev_Y.shape[0],))

	RMSE = np.sqrt(sklearn.metrics.mean_squared_error(dev_Y, dev_Y_hat))
	MAE = sklearn.metrics.mean_absolute_error(dev_Y, dev_Y_hat)
	EVS = sklearn.metrics.explained_variance_score(dev_Y, dev_Y_hat)

	print('RMSE :', RMSE)
	print('MAE :', MAE)
	#print(np.std(dev_Y - dev_Y_hat))
	print('EVS :', EVS)

	with open('regression_metrics.txt', 'w') as f:
		f.write('RMSE\t:\t' + str(RMSE) + '\nMAE\t\t:\t' + str(MAE) + '\nEVS\t\t:\t' + str(EVS))