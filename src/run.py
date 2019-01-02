from src.model import Autoencoder
from data.datasets import Dataset

ae = Autoencoder()
ds = Dataset(data_set_name="VoxCeleb2")
training_input, training_output = ds.get_training_data_samples(count=500)
ae.train_to_loss(training_input, training_output, loss_limit=100.0)
ae.save()