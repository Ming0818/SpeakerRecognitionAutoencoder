import pathlib
import os

abs_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

data_folder = os.path.join(abs_dir, 'data', '{data_set_folder}')
SVM_save_folder = os.path.join(abs_dir, 'data', '{data_set_folder}','{user}')
"""
                    possible_ dataset_names:
                        Speaker_in_the_wild,
                        VoxCeleb1, 
                        VoxCeleb2,
                        VoxCeleb1_Test_Set,
                    datasets and configuration saved in the datasdets dict
"""

model_save_format = os.path.join(abs_dir, 'models', 'CheckPoints', '{}')

train_set_encoded_save_folder = os.path.join(abs_dir, 'encodes2', '{}')


datasets = {
    'Speaker_in_the_wild': {
        "name": "Speaker_in_the_wild",
        "listing_method": "",
        "types": ["dev", "eval"],
        "components": ['audio', 'keys', 'lists'],
        "data_set_folder": "SITW"
    },
    'VoxCeleb1': {
        "name": "VoxCeleb1",
        # "listing_method": "user_session_record",   # fresh download
        "listing_method": "user_record",   # modified for kaldi
        "user_count": 1251,
        "data_set_folder": os.path.join("VoxCeleb1", "voxceleb1_wav")
    },
    'VoxCeleb2': {
        "name": "VoxCeleb2",
        "listing_method": "user_session_record",   # modified for kaldi
        "types": ["dev"],
        "aac": ["aac"],
        "user_count": 5994,
        "data_set_folder": "VoxCeleb2/dev/aac"

    },
    'VoxCeleb1_Test_Set': {
        "name": "VoxCeleb1_Test_Set",
        # "listing_method": "user_session_record",   # fresh download
        "listing_method": "user_record",   # modified for kaldi
        "user_count": 40,
        "data_set_folder": os.path.join("VoxCeleb1", "voxceleb1_wav_test")
    },
}


def make_folders():
    try:
        pathlib.Path(model_save_format.format('')).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass


if __name__ == '__main__':
    make_folders()
    print(data_folder)
