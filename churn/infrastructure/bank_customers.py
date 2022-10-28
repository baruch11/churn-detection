"""This module loads bank customers data from different files."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class BankCustomersData:
    """Represents bank customers datas."""
    indicators_file_name: str
    customers_file_name: str

    def load_data(self):
        """Collect customer data from files and merge & clean them.
        """
        indicators = pd.read_csv(self.indicators_file_name,
                                 sep=";").set_index("ID_CLIENT")
        customers = pd.read_csv(self.customers_file_name,
                                sep=";").set_index("ID_CLIENT")
        raw_data = indicators.join(customers, how='outer')

        # data cleaning
        clean_data = raw_data.assign(
            CARTE_CREDIT=lambda x: x.CARTE_CREDIT == "Yes",
            MEMBRE_ACTIF=lambda x: x.MEMBRE_ACTIF == "Yes",
            SEXE=lambda x: x.SEXE == "F",
            DATE_ENTREE=pd.to_datetime(raw_data.DATE_ENTREE),
        )
        if "CHURN" in raw_data.columns:
            clean_data["CHURN"] = clean_data["CHURN"] == "Yes"
        return clean_data
