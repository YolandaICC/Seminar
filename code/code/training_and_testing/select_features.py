

def select_features():
    # TODO: Here you have to select the features you want to use for your model.
    #  If you change your data set, you have to change this function accordingly.
    features_tracks = [
        #"recordingId",
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
    features_tracksmeta = [
        "class",
        "trackId"
        ]
    number_of_features = len(features_tracks) + len(features_tracksmeta)-1
    return features_tracks, features_tracksmeta, number_of_features
