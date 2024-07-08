import lightning as pl
from torch.utils.data import DataLoader, random_split
import os
from kinematic_bicycle_datasetclass import KinematicBicycleDataset


class KinematicBicycleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, past_sequence_length, future_sequence_length):

        super().__init__()
        # self.data_path = data_path
        self.batch_size = batch_size
        # self.transform = None

        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.sequence_length = self.past_sequence_length + self.future_sequence_length
        # self.features = features
        self.save_hyperparameters()

    def setup(self, stage: str):
        """Setup the data.
        Parameters
        ----------
        stage : str
            Stage of the data. Can be "fit", "test" or "predict".
        """
        if stage == "test":
            self.test = KinematicBicycleDataset(self.sequence_length)
        if stage == "fit":
            full = KinematicBicycleDataset(self.sequence_length)
            data_size = len(full)
            # TODO: change the ration between train and val if you like!
            train_size = int(0.9 * data_size)
            val_size = int(data_size - train_size)
            self.train, self.val = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          persistent_workers=True,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=os.cpu_count() - 2,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True
                          )
