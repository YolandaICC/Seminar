

def select_features():
    # TODO: Here you have to select the features you want to use for your model.
    #  If you change your data set, you have to change this function accordingly.
    features = [
        "recordingId",
        "trackId",
        #"frame",
        #"trackLifetime",
        "xCenter",
        "yCenter",
        "heading",
        # "width",
        #"length",
         "xVelocity",
         "yVelocity",
        "xAcceleration",
        "yAcceleration",
        # "lonVelocity",
        # "latVelocity",
        # "lonAcceleration",
        # "latAcceleration",
        ]
    number_of_features = len(features) + 1
    return features, number_of_features


