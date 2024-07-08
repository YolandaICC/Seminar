import lightning as pl
from kinematic_bicycle_datamodule import KinematicBicycleDataModule
from experiment_setup import ExperimentSetup
from models import ConstantVelocityModel

# HEYYY!! Put the debug point in line 7 and step through the code to see the values of the variables
batch_size = 32
past_sequence_length = 1 # Must be 1 since the ConstantVelocityModel only takes the last state as input
future_sequence_length = 3 # can be any value since you can apply the model multiple times
max_epochs = 10

# initiliazes KinematicBicycleDataModule class with the necessary parameters to load the data and create the dataloaders.
dm = KinematicBicycleDataModule(batch_size=batch_size,
                                past_sequence_length=past_sequence_length,
                                future_sequence_length=future_sequence_length)

# REMARK: Since you do not need to train the model, you can use the test dataset for training and testing
# You also don't call trainer.fit() since the model is not "trained"
# You can directly call trainer.test() to evaluate the model on the test dataset
dm.setup('test')
model = ConstantVelocityModel(dt=0.1)
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
trainer.test(experiment_setup, dm)
test = 0
