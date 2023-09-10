from typing import Tuple
import pandas as pd
import numpy as np
import time
import torch


def get_prices_predictions(model, timestamp_tuple: Tuple[int, int, int], device):
    """
    Predict prices using a given LSTM/Transformer model.

    Args:
    - model: Trained LSTM/Transformer model.
    - timestamp_tuple (tuple): (from_timestamp, to_timestamp, step_seconds).
                               Timestamps are in the format '2017-05-08 21:45:35+00:00'.
    - device (str): Torch device ('cuda' or 'cpu').

    Returns:
    - predictions (torch.Tensor): Tensor with predicted prices for each provider and each timestamp.

    Example Usage:
    --------------
    # Load your trained model
    # model = ...

    # Set your device ('cuda' or 'cpu')
    # device = ...

    # Predict prices every hour for a day
    timestamps = ('2017-05-08 21:45:35+00:00', '2017-05-09 21:45:35+00:00', 3600)
    prices = predict_prices(model, timestamps, device)
    """

    # Extract timestamps from tuple
    from_timestamp, to_timestamp, step_seconds = timestamp_tuple

    # Convert timestamps to datetime format and then to UNIX
    start_unix = pd.Timestamp(from_timestamp).timestamp()
    end_unix = pd.Timestamp(to_timestamp).timestamp()

    # Create a range of UNIX timestamps
    timestamps_range = np.arange(start_unix, end_unix + step_seconds, step_seconds)

    all_predictions = []

    # Model prediction
    model.eval()
    with torch.no_grad():
        for ts in timestamps_range:
            # Convert UNIX timestamp to tensor format
            input_tensor = torch.FloatTensor([[ts]]).to(device)

            # Make a prediction for the current timestamp
            prediction = model(input_tensor)
            all_predictions.append(prediction)

    # Stack all predictions into a single tensor
    predictions_tensor = torch.stack(all_predictions, dim=0)

    return predictions_tensor
