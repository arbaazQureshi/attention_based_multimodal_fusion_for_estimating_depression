import numpy as np

training_COVAREP_output_encoding = np.load('training_set_encoding/COVAREP_output_training_encoding.npy')
training_FORMANT_features_encoding = np.load('training_set_encoding/FORMANT_features_training_encoding.npy')

m_train = training_COVAREP_output_encoding.shape[0]

training_COVAREP_output_encoding = np.hstack((training_COVAREP_output_encoding, np.ones((m_train,1))))
training_FORMANT_features_encoding = np.hstack((training_FORMANT_features_encoding, np.ones((m_train, 1))))

training_outer_products = []

for i in range(m_train):
	training_outer_products.append(np.outer(training_COVAREP_output_encoding[i], training_FORMANT_features_encoding[i]).ravel().tolist())

np.save('speech_training_set_combined_features.npy', np.array(training_outer_products))




dev_COVAREP_output_encoding = np.load('development_set_encoding/COVAREP_output_development_encoding.npy')
dev_FORMANT_features_encoding = np.load('development_set_encoding/FORMANT_features_development_encoding.npy')

m_dev = dev_COVAREP_output_encoding.shape[0]

dev_COVAREP_output_encoding = np.hstack((dev_COVAREP_output_encoding, np.ones((m_dev,1))))
dev_FORMANT_features_encoding = np.hstack((dev_FORMANT_features_encoding, np.ones((m_dev, 1))))

#np.save('speech_training_set_combined_features.npy', np.hstack((training_COVAREP_output_encoding, training_FORMANT_features_encoding)))

dev_outer_products = []

for i in range(m_dev):
	dev_outer_products.append(np.outer(dev_COVAREP_output_encoding[i], dev_FORMANT_features_encoding[i]).ravel().tolist())

np.save('speech_development_set_combined_features.npy', np.array(dev_outer_products))