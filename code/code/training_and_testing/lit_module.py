import lightning as pl
import torch
from step_factory import StepFactory

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
    def __init__(self, model, number_of_features, sequence_length, past_sequence_length, future_sequence_length, batch_size, current_model):
        super().__init__()
        self.model = model
        self.nx = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size
        self.current_model = current_model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        string = "training"
        loss = StepFactory.get_step(self, batch, batch_idx, string, self.current_model)
        return loss

    def validation_step(self, batch, batch_idx):
        string = "validation"
        loss = StepFactory.get_step(self, batch, batch_idx, string, self.current_model)
        return loss

    def test_step(self, batch, batch_idx):
        string = "test"
        loss = StepFactory.get_step(self, batch, batch_idx, string, self.current_model)
        return loss

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
                "monitor": "training_loss",
                "frequency": 1,
                "strict": True}
        }
        return optimizer_and_scheduler
        # else:
        #     return []

