from enums.model import Model
from Step.Steps import mlp_step, lstm_step, constant_velocity_step, constant_accelaration_step, single_track_step, hybrid_step


class StepFactory:

    def get_step(self, batch, batch_idx, string, model: str):

        if model == Model.CONSTANT_VELOCITY.value:
            return constant_velocity_step(self, batch, batch_idx, string)

        elif model == Model.CONSTANT_ACCELERATION.value:
            return constant_accelaration_step(self, batch, batch_idx, string)

        elif model == Model.SINGLE_TRACK.value:
            return single_track_step(self, batch, batch_idx, string)

        elif model == Model.MLP.value:
            return mlp_step(self, batch, batch_idx, string)

        elif model == Model.LSTM.value:
            return lstm_step(self, batch, batch_idx, string)

        elif model == Model.HYBRID.value:
            return hybrid_step(self, batch, batch_idx, string)