from enum import Enum


class Model(Enum):
    MLP = "MLP"
    LSTM = "LSTM"
    CONSTANT_VELOCITY = "CONSTANT_VELOCITY"
    CONSTANT_ACCELERATION = "CONSTANT_ACCELERATION"
    SINGLE_TRACK = "SINGLE_TRACK"
    HYBRID_PARALLEL = "HYBRID_PARALLEL"
    HYBRID_SERIAL = "HYBRID_SERIAL"