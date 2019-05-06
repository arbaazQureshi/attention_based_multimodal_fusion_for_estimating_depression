import numpy as np



training_gaze_features_embedding = np.load('training_set_embeddings/gaze_features_training_embedding.npy')
training_action_units_embedding = np.load('training_set_embeddings/action_units_training_embedding.npy')

m_train = training_gaze_features_embedding.shape[0]

training_gaze_features_embedding = np.hstack((training_gaze_features_embedding, np.ones((m_train,1))))
training_action_units_embedding = np.hstack((training_action_units_embedding, np.ones((m_train, 1))))

training_gaze_X_action = []

for i in range(m_train):
	training_gaze_X_action.append(np.outer(training_gaze_features_embedding[i], training_action_units_embedding[i]).ravel().tolist())

np.save('training_tensor_fusion_output/gaze_X_action_training.npy', np.array(training_gaze_X_action))




dev_gaze_features_embedding = np.load('development_set_embeddings/gaze_features_development_embedding.npy')
dev_action_units_embedding = np.load('development_set_embeddings/action_units_development_embedding.npy')

m_dev = dev_gaze_features_embedding.shape[0]

dev_gaze_features_embedding = np.hstack((dev_gaze_features_embedding, np.ones((m_dev,1))))
dev_action_units_embedding = np.hstack((dev_action_units_embedding, np.ones((m_dev, 1))))

dev_gaze_X_action = []

for i in range(m_dev):
	dev_gaze_X_action.append(np.outer(dev_gaze_features_embedding[i], dev_action_units_embedding[i]).ravel().tolist())

np.save('development_tensor_fusion_output/gaze_X_action_development.npy', np.array(dev_gaze_X_action))




training_facial_landmarks_embedding = np.load('training_set_embeddings/facial_landmarks_training_embedding.npy')
training_pose_features_embedding = np.load('training_set_embeddings/pose_features_training_embedding.npy')

m_train = training_facial_landmarks_embedding.shape[0]

training_facial_landmarks_embedding = np.hstack((training_facial_landmarks_embedding, np.ones((m_train,1))))
training_pose_features_embedding = np.hstack((training_pose_features_embedding, np.ones((m_train, 1))))

training_facial_X_pose = []

for i in range(m_train):
	training_facial_X_pose.append(np.outer(training_facial_landmarks_embedding[i], training_pose_features_embedding[i]).ravel().tolist())

np.save('training_tensor_fusion_output/facial_X_pose_training.npy', np.array(training_facial_X_pose))




dev_facial_landmarks_embedding = np.load('development_set_embeddings/facial_landmarks_development_embedding.npy')
dev_pose_features_embedding = np.load('development_set_embeddings/pose_features_development_embedding.npy')

m_dev = dev_facial_landmarks_embedding.shape[0]

dev_facial_landmarks_embedding = np.hstack((dev_facial_landmarks_embedding, np.ones((m_dev,1))))
dev_pose_features_embedding = np.hstack((dev_pose_features_embedding, np.ones((m_dev, 1))))

dev_facial_X_pose = []

for i in range(m_dev):
	dev_facial_X_pose.append(np.outer(dev_facial_landmarks_embedding[i], dev_pose_features_embedding[i]).ravel().tolist())

np.save('development_tensor_fusion_output/facial_X_pose_development.npy', np.array(dev_facial_X_pose))





training_COVAREP_output_embedding = np.load('training_set_embeddings/COVAREP_output_training_embedding.npy')
training_FORMANT_features_embedding = np.load('training_set_embeddings/FORMANT_features_training_embedding.npy')

m_train = training_COVAREP_output_embedding.shape[0]

training_COVAREP_output_embedding = np.hstack((training_COVAREP_output_embedding, np.ones((m_train,1))))
training_FORMANT_features_embedding = np.hstack((training_FORMANT_features_embedding, np.ones((m_train, 1))))

training_COVAREP_X_FORMANT = []

for i in range(m_train):
	training_COVAREP_X_FORMANT.append(np.outer(training_COVAREP_output_embedding[i], training_FORMANT_features_embedding[i]).ravel().tolist())

np.save('training_tensor_fusion_output/COVAREP_X_FORMANT_training.npy', np.array(training_COVAREP_X_FORMANT))




dev_COVAREP_output_embedding = np.load('development_set_embeddings/COVAREP_output_development_embedding.npy')
dev_FORMANT_features_embedding = np.load('development_set_embeddings/FORMANT_features_development_embedding.npy')

m_dev = dev_COVAREP_output_embedding.shape[0]

dev_COVAREP_output_embedding = np.hstack((dev_COVAREP_output_embedding, np.ones((m_dev,1))))
dev_FORMANT_features_embedding = np.hstack((dev_FORMANT_features_embedding, np.ones((m_dev, 1))))

dev_COVAREP_X_FORMANT = []

for i in range(m_dev):
	dev_COVAREP_X_FORMANT.append(np.outer(dev_COVAREP_output_embedding[i], dev_FORMANT_features_embedding[i]).ravel().tolist())

np.save('development_tensor_fusion_output/COVAREP_X_FORMANT_development.npy', np.array(dev_COVAREP_X_FORMANT))





training_transcript_embedding = np.load('training_set_embeddings/transcript_features_training_embedding.npy')
np.save('training_tensor_fusion_output/transcript_training.npy', np.array(training_transcript_embedding))

dev_transcript_embedding = np.load('development_set_embeddings/transcript_features_development_embedding.npy')
np.save('development_tensor_fusion_output/transcript_development.npy', np.array(dev_transcript_embedding))