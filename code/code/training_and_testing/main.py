# This file is the main file for the lightning training of the neural network models.
# The main file is responsible for the following:
# 1. Creating the datamodule
# 2. Creating the model
# 3. Creating the callbacks
# 4. Creating the logger
# 5. Creating the trainer
# 6. Fitting the model

# READ THE COMMENTS AND TODOs CAREFULLY!
# THIS IS NOT A FULL TUTORIAL OR DOCUMENTATION!
# Please check the pytorch lightning documentation for more information:
# https://lightning.ai/docs/pytorch/stable/

import logging
import lightning as pl
import torch
import wandb
from callbacks import create_callbacks
from lit_datamodule import inD_RecordingModule
from lit_module import LitModule
from utils import create_wandb_logger, get_data_path, build_module
from nn_modules import ConstantVelocityModel, MultiLayerPerceptron, LSTMModel, ConstantAccelerationModel
from select_features import select_features

##################################################################
# torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(True)
# TODO: check get_data_path() in utils.py. You can change the data path there.
data_path, log_path = get_data_path()
# TODO: Check out weights and biases. It is a great tool for logging and visualizing your results.
#  For students, accounts are free!
# wandb.login()
# TODO: Remove this line if you don't want to use wandb
##################################################################

project_name = "SS2024_motion_prediction"

# TODO: The stages are defined in the lit_datamodule.py file. Right now, we have a train, val, and test stage.
#  For some of the models you dont actually train anything, like the constant velocity model, you can simply use the test stage.
#  The test stage should also be used for the final evaluation of any model.
stage = "fit"
# stage = "test"
###S################# Training Parameters #####################################
# TODO: Change the recording_ID to the recordings you want to train on
#recording_ID = ["01", "02"]#, "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"]
recording_ID = ["00"]

# TODO: Change the features to the features you want to use. The features are defined in the select_features.py file
# This is referring to an unmodified dataset. So depending on your goal, modify the dataset and set the features accordingly.
#  If you change your dataset, you have to change recreate a feature list that suits your dataset
features_tracks, features_tracksmeta, number_of_features = select_features()
past_sequence_length = 2
future_sequence_length = 3
sequence_length = past_sequence_length + future_sequence_length

#################### Model Parameters #####################################

batch_size = 50

# Use first input_size for MLP and second input_size fo LSTM
# input_size = number_of_features * past_sequence_length
input_size = number_of_features
output_size = number_of_features
hidden_size = 32

if __name__ == '__main__':
    #################### Create Models #####################################
    # TODO: In the lit_module.py file, the model is defined. The model is defined using the LitModule class. Right now,
    #  we simply load the MLP model. Depending on your research question, you have to change the model.
    # TODO: Create you models in the nn_modules.py file. You can create as many models as you want. The models should be
    #  defined as a class. The class should inherit from torch.nn.Module. Check out the MLPModel class in the nn_modules.py!
    # mdl = ConstantVelocityModel()
    # mdl = ConstantAccelerationModel()
    # mdl = MultiLayerPerceptron(input_size, hidden_size, output_size)
    mdl = LSTMModel(input_size, hidden_size, output_size, future_sequence_length=future_sequence_length)
    # hidden_tensor = mdl.init_zero_hidden(batch_size)



    # TODO: In the datamodule, the dataset is created. The dataset is created using the inD_RecordingDataset class. Right now,
    #  we simply load the csv files and concatenate them. Depending on your research question, you have to rearrange the data.
    #  The datamodule is defined in the kinematic_bicycle_datasetclass.py file
    #  The data set is defined in the lit_dataset.py file
    #  Check them out now!
    dm = inD_RecordingModule(data_path, recording_ID, sequence_length, past_sequence_length, future_sequence_length, features_tracks, features_tracksmeta, batch_size=batch_size)



    #################### Setup Training #####################################
    # TODO: Change the epochs to the number of epochs you want to train
    epochs = 10
    # model = LitModule(mdl, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size, hidden_tensor)
    model = LitModule(mdl, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size)

    dm.setup(stage=stage)

    # TODO: Change the callbacks to the callbacks you want to use. The callbacks are defined in the callbacks.py file
    #  Simply uncomment some and see what happens! Mostly they save the model, log the model, or do something else.
    callbacks = create_callbacks()
    #
    # wandb_logger = create_wandb_logger(log_path, project_name, recording_ID)
    # wandb_logger.experiment.config.update({"batch_size": batch_size,
    #                                        "sequence_length": sequence_length})
    logging.getLogger(log_path + "/lightning").setLevel(logging.ERROR)

    #################### Start Training #####################################
    trainer = pl.Trainer(max_epochs=epochs,
                         fast_dev_run=False,
                         devices="auto",
                         accelerator="auto",
                         log_every_n_steps=5,
                         # logger=wandb_logger,
                         callbacks=callbacks,
                         check_val_every_n_epoch=1,
                         precision="64-true"
                         )

    if stage == "fit":
        trainer.fit(model, dm)
    elif stage == "test":
        trainer.test(model, dm)
    # wandb.finish()