import opensmile
import numpy as np

def opensmile_features(data : np.ndarray):

    """
        Inputs:
            -data: Array de dos dimensiones con los features

        Output:
            Lista con los features
    """

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    x = []
    for i in data:
        x.append(list(smile.process_signal(i, sampling_rate = 16000).values[0]))

    return x