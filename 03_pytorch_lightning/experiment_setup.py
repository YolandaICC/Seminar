import torch
import lightning as pl
import torch.nn.functional as F
from models import ConstantVelocityModel
import matplotlib.pyplot as plt


class ExperimentSetup(pl.LightningModule):
    def __init__(self, nn_module, past_sequence_length=50, future_sequence_length=50):
        super().__init__()
        self.model = nn_module
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.sequence_length = self.past_sequence_length + self.future_sequence_length
        self.test_step_outputs = []

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, batch, batch_idx):
        # The batch of data is assigned to the variable 'data'
        data = batch
        # Initialize empty list to store the predicted output values
        y_predicted_list = []
        # The input sequence 'x_t' is extracted from the first 'past_sequence_length' elements of 'data'
        x_t = data[:, 0:self.past_sequence_length]
        # The output sequence 'y' is extracted from the remaining elements of 'data'
        y = data[:, self.past_sequence_length::]
        # Loop over the range of 'future_sequence_length'
        for t in range(self.future_sequence_length):
            # For each iteration, make a prediction 'y_predicted_t' based on the current input sequence 'x_t'
            y_predicted_t = self.forward(x_t)
            # Append the predicted output to the 'y_predicted_list'
            y_predicted_list.append(y_predicted_t)
            # Now we have to differentiate the implementation whether we use the constant velocity or the MLP model
            # since the CV only uses one past value for the prediction. All the squeezes and unsqueezes
            # are necessary because of that
            # If the length of 'x_t' is less than 2, replace 'x_t' with the current prediction 'y_predicted_t'
            if x_t.shape[1] < 2:
                x_t = y_predicted_t
            else:
                # Otherwise, remove the first element of 'x_t' and append 'y_predicted_t' to the end
                x_t_without_first = x_t[:, 1:, :]
                x_t = torch.cat((x_t_without_first, y_predicted_t.unsqueeze(1)), dim=1)
        # Stack the predicted outputs in 'y_predicted_list' into a tensor 'y_predicted'
        y_predicted = torch.stack(y_predicted_list, dim=1).squeeze()
        # Compute the mean squared error loss between the predicted and actual output sequences
        loss = F.mse_loss(y_predicted, y)
        # Log the loss with the key "train_loss"
        self.log("train_loss", loss)
        # Return a dictionary containing the loss and the predicted and actual output values
        return_dict = {"loss": loss, "y_predicted": y_predicted, "y": y}
        return return_dict

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return_dict = self.step(batch, batch_idx)
        y = return_dict["y"]
        y_predicted = return_dict["y_predicted"]
        self.test_step_outputs.append(y)
        self.test_step_outputs.append(y_predicted)
        loss = return_dict["loss"]
        return loss

    def on_test_epoch_end(self):
        # This method just visualizes the trajectories of the bicycle model
        # You don't really have to get this part, except you are interested in visualizing the results
        y_list = []
        y_predicted_list = []
        for i in range(0, len(self.test_step_outputs), 2):
            y_list.append(self.test_step_outputs[i])
            y_predicted_list.append(self.test_step_outputs[i + 1])

        y = torch.cat(y_list, dim=0)
        y_predicted = torch.cat(y_predicted_list, dim=0)
        y = y.detach().cpu().numpy()
        y_predicted = y_predicted.detach().cpu().numpy()
        trajectories = [y, y_predicted]
        names = ['Ground Truth', 'Predicted']
        fig = plt.figure(figsize=(8, 6))
        for k in range(0, len(trajectories)):
            name = names[k]
            X_coord = trajectories[k][:, 0, 0]
            Y_coord = trajectories[k][:, 0, 1]
            plt.plot(X_coord, Y_coord, label=f"Bicycle Path_{name}")
            plt.scatter(X_coord[0], Y_coord[0], label=f"Start_{name}")
            plt.scatter(X_coord[-1], Y_coord[-1], label=f"End_{name}")
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            plt.title('Bicycle Trajectory')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
