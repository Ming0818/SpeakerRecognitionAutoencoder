from src.model import Autoencoder
from data.datasets import Dataset
from sklearn import svm
import pickle


import numpy as np
import progressbar

from configuration import train_set_encoded_save_folder, SVM_save_folder
from utilities import read_random_line

from sklearn.metrics import accuracy_score

###############################################
# Instantiate and load the model
#
#
ae = Autoencoder()
ae.load("/home/amir/Projects/VoxCelebAutoencoder/10Layer_model_4filter")
###############################################

###############################################
# Load Train Set and Test Set
#
#
train_ds = Dataset(data_set_name="VoxCeleb1")
test_ds = Dataset(data_set_name="VoxCeleb1_Test_Set")
###############################################

###############################################
# Training Phase
#
#
# loss = np.Inf
# while loss > 1:
#     training_input, training_output = train_ds.get_training_data_samples(count=1000)
#     loss = ae.train_to_epoch(training_input, training_output)
#     ae.save()
###############################################

###############################################
# Extract Encoded Data
#
#
# for user in progressbar.progressbar(train_ds.users_list):
#     raw_data = train_ds.read_whole_data_set_for_user(user)
#     encoded_data = ae.encode_input(input_data=raw_data)
#     records_count = len(encoded_data[0])
#     encoded_data = np.reshape(encoded_data, (records_count, -1))
#     np.savetxt(train_set_encoded_save_folder.format(user), encoded_data)
###############################################

###############################################
# Extract Train Data Points and Train SVM
#
#
for user in progressbar.progressbar(test_ds.users_list):
    raw_data = test_ds.read_whole_data_set_for_user(user)
    encoded_records = ae.encode_input(input_data=raw_data)
    reference_records = encoded_records[0][:5]
    SVM_train_user_set = np.random.choice(train_ds.users_list, 30)

    train_data, train_label = [], []
    for Non_User in SVM_train_user_set:
        reference_points = np.reshape(reference_records, (len(reference_records), -1))
        train_data.extend(reference_points)
        train_label.extend([1]*len(reference_records))


        other_records = list(map(float, read_random_line(train_set_encoded_save_folder.format(Non_User)).split()))

        negative_records = [other_records]

        train_data.extend(negative_records)
        train_label.extend([0]*len(negative_records))
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(train_data, train_label)
    pickle.dump(clf, open(SVM_save_folder.format(data_set_folder='VOXCELEB1_test_SVMs', user=user), 'wb'))
###############################################


###############################################
# Test Classifiers
#
#
for user in progressbar.progressbar(test_ds.users_list):
    test_data, test_label = [], []
    raw_data = test_ds.read_whole_data_set_for_user(user)
    encoded_records = ae.encode_input(input_data=raw_data)
    questioned_training_records_cubed = encoded_records[0][5:]
    questioned_training_records = np.reshape(
                                            questioned_training_records_cubed,
                                            (len(questioned_training_records_cubed), -1)
                                            )
    test_data.extend(questioned_training_records)
    test_label.extend([1] * len(questioned_training_records))

    possible_users = list(test_ds.users_list)
    possible_users.remove(user)
    test_user = np.random.choice(possible_users, 1)[0]
    raw_data = test_ds.read_whole_data_set_for_user(test_user)
    encoded_records = ae.encode_input(input_data=raw_data)
    questioned_training_records_cubed = encoded_records[0]
    questioned_training_records = np.reshape(questioned_training_records_cubed,
                                             (len(questioned_training_records_cubed), -1))
    test_data.extend(questioned_training_records)
    test_label.extend([0] * len(questioned_training_records))

    clf = pickle.load(
                        open(SVM_save_folder.format(data_set_folder='VOXCELEB1_test_SVMs', user=user), 'rb')
                     )  # type: svm.SVC
    print(accuracy_score(clf.predict(test_data), test_label))
###############################################
