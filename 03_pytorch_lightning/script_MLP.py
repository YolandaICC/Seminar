import lightning as pl
from experiment_setup import ExperimentSetup
from kinematic_bicycle_datamodule import KinematicBicycleDataModule
from models import MultiLayerPerceptron


def main():
    # HEYYY!! Put the debug point in line 7 and step through the code to see the values of the variables
    batch_size = 32
    past_sequence_length = 4
    future_sequence_length = 3
    max_epochs = 10  # How many times should it go through the whole data set before finishing the training

    # initiliazes KinematicBicycleDataModule class with the necessary parameters to load the data and create the dataloaders.
    dm = KinematicBicycleDataModule(batch_size=batch_size,
                                    past_sequence_length=past_sequence_length,
                                    future_sequence_length=future_sequence_length)

    dm.setup('fit')
    model = MultiLayerPerceptron(input_dim=3 * past_sequence_length, hidden_dim=32, output_dim=3)
    experiment_setup = ExperimentSetup(model,
                                       past_sequence_length=past_sequence_length,
                                       future_sequence_length=future_sequence_length)

    trainer = pl.Trainer(max_epochs=max_epochs,
                         fast_dev_run=False,
                         devices="auto",
                         accelerator="auto",
                         # logger = logger,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         precision='64',
                         )
    # assert issubclass(experiment_setup, pl.LightningModule), "LightningModule is not a subclass of pl.LightningModule"
    trainer.fit(experiment_setup, dm)

    dm.setup('test')
    experiment_setup.eval()
    trainer.test(experiment_setup, dm)


if __name__ == '__main__':
    main()
