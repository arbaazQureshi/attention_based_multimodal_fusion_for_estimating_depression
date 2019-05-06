import numpy as np
import sklearn.metrics

from load_data import load_development_data
from keras.models import load_model


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

	model = load_model('min_model.h5')

	X_dev, Y_dev, X_dev_gender = load_development_data()

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mean_absolute_error'])

	Y_hat_dev = model.predict([X_dev, X_dev_gender])

	Y_dev = np.array(Y_dev)
	Y_hat_dev = Y_hat_dev.reshape((Y_dev.shape[0],))

	RMSE = np.sqrt(sklearn.metrics.mean_squared_error(Y_dev, Y_hat_dev))
	MAE = sklearn.metrics.mean_absolute_error(Y_dev, Y_hat_dev)
	EVS = sklearn.metrics.explained_variance_score(Y_dev, Y_hat_dev)

	print('RMSE :', RMSE)
	print('MAE :', MAE)
	#print(np.std(Y_dev - Y_hat_dev))
	print('EVS :', EVS)

	with open('regression_metrics.txt', 'w') as f:
		f.write('RMSE\t:\t' + str(RMSE) + '\nMAE\t\t:\t' + str(MAE) + '\nEVS\t\t:\t' + str(EVS))