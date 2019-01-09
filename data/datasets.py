from configuration import datasets, data_folder


import numpy as np
import os
import json
import librosa

# listing functions
def user_record(data_set):
    print('Listing for Dataset {}'.format(data_set['name']))
    dataset_path = data_folder.format(data_set_folder=data_set['data_set_folder'])
    user_files = dict()
    try:
        with open(os.path.join(dataset_path, 'users_file.records'), 'r') as records_file:
            content = records_file.read()
            user_files = json.loads(content)
        with open(os.path.join(dataset_path, 'users_list.records'), 'r') as users_file:
            content = users_file.read()
            users_list = json.loads(content)
    except IOError:
        users_list = os.listdir(dataset_path)
        for user in users_list:
            user_files[user] = [os.path.join(dataset_path, user, file_name) for file_name in
                                   os.listdir(os.path.join(dataset_path, user))]
        with open(os.path.join(dataset_path,'users_file.records'), 'w+') as outputfile:
            outputfile.write(json.dumps(user_files))
        with open(os.path.join(dataset_path,'users_list.records'), 'w+') as users_file:
            users_file.write(json.dumps(users_list))
    return users_list, user_files


def u_s_r(data_set):
    print('Listing for Dataset {}'.format(data_set['name']))
    dataset_path = data_folder.format(data_set_folder=data_set['data_set_folder'])
    user_files = dict()
    try:
        with open(os.path.join(dataset_path, 'users_file.records'), 'r') as records_file:
            content = records_file.read()
            user_files = json.loads(content)
        with open(os.path.join(dataset_path, 'users_list.records'), 'r') as users_file:
            content = users_file.read()
            users_list = json.loads(content)
    except IOError:
        users_list = os.listdir(dataset_path)
        for user in users_list:
            for session in os.listdir(os.path.join(dataset_path, user)):
                user_files[user] = user_files.get(user, []) + [os.path.join(dataset_path, user, session, file_name)
                                            for file_name in os.listdir(os.path.join(dataset_path, user, session))]
        with open(os.path.join(dataset_path, 'users_file.records'), 'w+') as outputfile:
            outputfile.write(json.dumps(user_files))
        with open(os.path.join(dataset_path, 'users_list.records'), 'w+') as users_file:
            users_file.write(json.dumps(users_list))
    return users_list, user_files


class Dataset:
    name = ''
    users = []
    users_dict = dict()

    listing_options = {
        'user_record': user_record,
        'user_session_record': u_s_r
    }

    def __init__(self, data_set_name=None):
        self.name = data_set_name
        self.data_set = datasets[self.name]
        self.users_list, self.users_dict = self.listing_options[self.data_set['listing_method']](self.data_set)

    def read_user_sample(self, user_id):
        user_files = self.users_dict.get(user_id, [])
        input_file = np.random.choice(user_files)
        output_file = np.random.choice(user_files)
        y = librosa.load(input_file , sr=16000, dtype=np.float64)[0]
        random_point = np.random.randint(0, len(y) - 48000)
        input = np.array([[x*10] for x in y[random_point:random_point + 48000]])

        y = librosa.load(output_file, sr=16000, dtype=np.float64)[0]
        random_point = np.random.randint(0, len(y) - 48000)
        output= np.array([[x*10] for x in y[random_point:random_point + 48000]])
        """ y, s=  librosa.load(file, sr=16000, duration=3.0, dtype=np.float64) is the common use of this function 
        but since the sample rate is fixed and also we don't need it we only get the first token also since we have 
        extremely low values of sound sample which is hard to work with I have multiplied it by 10 to make it work
        better"""
        return input, output

    def get_training_data_samples(self, count=500):
        train_user_list = np.random.choice(self.users_list, count, replace=False)
        train_input, train_output = [], []
        for user in train_user_list:
            input, output = self.read_user_sample(user)
            train_input.append(np.array(input))
            train_output.append(np.array(output))
        return train_input, train_output

    def read_whole_data_set_for_user(self, user_id):
        user_files = self.users_dict.get(user_id, [])
        read_files = []
        for input_file in user_files:
            y = librosa.load(input_file, sr=16000, dtype=np.float64)[0]
            random_point = np.random.randint(0, len(y) - 48000)
            input = np.array([[x * 10] for x in y[random_point:random_point + 48000]])
            read_files.append(input)
        return read_files


if __name__ == '__main__':
    pass