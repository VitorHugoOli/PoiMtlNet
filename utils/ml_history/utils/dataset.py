from typing import Optional


class DatasetHistory:
    """
    Class to manage the history of datasets.
    """

    def __init__(self,
                 raw_data: str,
                 description: str,
                 folds_signature: Optional[str] = None):
        """
        Initialize the DatasetHistory object.
        :param raw_data:
        :param description: 
        """

        self.raw_data = raw_data
        self.folds_signature = None
        self.description = description
