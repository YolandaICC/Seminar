from lightning.pytorch.callbacks import LearningRateMonitor


def create_callbacks():
    # TODO: Here are some examples for callback. They save your model, when a new best is achieved. Or help to adjust
    #  the learning rate.
    cb = \
        [
            # ModelCheckpoint(monitor="training_loss",
            #                 filename="TRAIN_CKPT_{training_loss:.2f}-{validation_loss:.2f}-{epoch}",
            #                 mode="min",
            #                 every_n_epochs=1,
            #                 save_top_k=2,
            #                 verbose="True",
            #                 auto_insert_metric_name="True",
            #                 save_on_train_epoch_end=True,),
            # ModelCheckpoint(monitor="validation_loss",
            #                 filename="VAL_CKPT_{validation_loss:.2f}-{training_loss:.2f}-{epoch}",
            #                 mode="min",
            #                 every_n_epochs=1,
            #                 save_top_k=2,
            #                 verbose="True",
            #                 auto_insert_metric_name="True",),
            # ModelCheckpoint(filename="REC_CKPT_{epoch}-{validation_loss:.2f}-{training_loss:.2f}",
            #                 every_n_epochs=10,
            #                 verbose="True",
            #                 auto_insert_metric_name="True",),
            LearningRateMonitor(logging_interval='step')
        ]
    return cb
