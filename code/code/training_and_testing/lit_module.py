import lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#will influence all the major process in the training process

class LitModule(pl.LightningModule):
    """
    This is a standard PyTorch Lightning module,
    with a few PyTorch-Lightning-specific things added.

    The main things to notice are:
    - Instead of defining different steps for training, validation, and testing,
        we define a single `step` function, and then define `training_step`,
        `validation_step`, and `test_step` as thin wrappers that call `step`.
    """
    # def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size, hidden_tensor):
    def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size, hidden_tensor = None):
        super().__init__()
        self.model = model
        self.nx = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size
        self.test_step_outputs = []
        self.hidden = hidden_tensor


    def forward(self, x):
        y_hat, hidden = self.model(x, self.hidden_tensor)
        return y_hat, hidden
        #
        # y_hat = self.model(x)
        # return y_hat

    def training_step(self, batch, batch_idx):
        string = "training"
        loss = self.step(batch, batch_idx, string)
        return loss

    def validation_step(self, batch, batch_idx):
        string = "validation"
        loss = self.step(batch, batch_idx, string)
        return loss

    def test_step(self, batch, batch_idx):
        string = "test"
        loss = self.step(batch, batch_idx, string)
        return loss

    def step(self, batch, batch_idx, string):
        """
        This is the main step function that is used by training_step, validation_step, and test_step.
        """
        # TODO: You have to modify this based on your task, model and data. This is where most of the engineering happens!
        x, y = self.prep_data_for_step(batch)

        self.hidden = self.model.init_hidden(x.size(0)) # Initializing the hidden layer every step may not save the hidden values, we need to look into that.

        # Detach hidden state from previous sequence
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        # print(x.shape, self.hidden[0].shape)
        output, self.hidden = self.model(x, self.hidden)

        loss = F.mse_loss(output, y)

        self.log("train_loss", loss)
        return loss
    
    def backward(self, loss):
        loss.backward(retain_graph=True)

    def on_test_epoch_end(self):
        # This method just visualizes the trajectories of the bicycle model
        # You don't really have to get this part, except you are interested in visualizing the results
        y_list = []
        y_hat_list = []
        for i in range(0, len(self.test_step_outputs), 2):
            y_list.append(self.test_step_outputs[i])
            y_hat_list.append(self.test_step_outputs[i + 1])

        y = torch.cat(y_list, dim=0)
        y_predicted = torch.cat(y_hat_list, dim=0)
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

    def prep_data_for_step(self, batch):
        # TODO: This is a hacky way to load one rectangular block from the data, and divide it into x and y of different
        #  sizes afterwards.
        #  If you don't do it like this, you run into trouble. Just stay aware of this.
        x = batch[:, :self.past_sequence_length, :]
        y = batch[:, self.past_sequence_length:, :]
        return x, y

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        # if parameters:
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-3,
                                     eps=1e-5,
                                     fused=False,
                                     amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=3,
            threshold=1e-4,
            cooldown=2,
            eps=1e-6,
            verbose=True,
        )
        optimizer_and_scheduler = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "verbose": True,
                "frequency": 1,
                "strict": True}
        }
        return optimizer_and_scheduler
        # else:
        #     return []


