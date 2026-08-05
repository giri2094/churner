import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)
