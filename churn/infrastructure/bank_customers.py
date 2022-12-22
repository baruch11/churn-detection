"""This module loads bank customers data from different files."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class BankCustomersData:
    """Represents bank customers datas."""

    indicators_file_name: str
    customers_file_name: str

    def load_data(self):
        """Collect customer data from files and merge into a DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        indicators = pd.read_csv(self.indicators_file_name,
                                 sep=";").set_index("ID_CLIENT")
        customers = pd.read_csv(self.customers_file_name,
                                sep=";").set_index("ID_CLIENT")
        merged_data = indicators.join(customers, how='outer')

        return merged_data
