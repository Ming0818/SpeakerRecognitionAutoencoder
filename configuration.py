import pathlib, os

abs_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])

data_folder = os.path.join(abs_dir, 'data', 'data_set_name')
"""
                    possible_ dataset_names:
                        Speaker_in_the_wild_3_Seconds,
                        Speaker_in_the_wild_25_Milli,
"""

model_save_format = os.path.join(abs_dir, 'models', 'CheckPoints', '{}')


def make_folders():
    try:
        pathlib.Path(model_save_format.format('')).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass


if __name__ == '__main__':
    make_folders()
