"""Feature engineering for the California housing dataset."""

import pandas as pd


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the raw feature matrix.

    New columns added:
        - RoomsPerHousehold: AveRooms / HouseAge (rooms per year of age)
        - BedroomRatio: AveBedrms / AveRooms
        - PopulationPerHousehold: Population / Households

    Args:
        X: Raw feature DataFrame from load_dataset().

    Returns:
        New DataFrame with original columns plus derived features.
    """
    raise NotImplementedError("implement in later commit")
