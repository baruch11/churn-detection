"""Unit tests for model."""
import pandas as pd
from churn.domain.churn_model import ChurnModelFinal
from churn.domain.domain_utils import get_test_set
import numpy as np
from sklearn.metrics import f1_score
from unittest import TestCase


def test_churnmodelfinal():
    model = ChurnModelFinal.load()

    input_data = [{'BALANCE': 0.0, 'NB_PRODUITS': 2, 'CARTE_CREDIT': 'YES',
                   'SALAIRE': 88947.56, 'SCORE_CREDIT': 677.0,
                   'DATE_ENTREE': '2015-06-01 00:00:00',
                   'NOM': 'Tai', 'PAYS': 'Espagne', 'SEXE': 'M', 'AGE': 40,
                   'MEMBRE_ACTIF': 'NO'},
                  {'BALANCE': 0.0, 'NB_PRODUITS': 14, 'CARTE_CREDIT': 'YES',
                   'SALAIRE': np.nan, 'SCORE_CREDIT': np.nan,
                   'DATE_ENTREE': '2014-04-01 00:00:00',
                   'NOM': 'Ross', 'PAYS': 'Espagne', 'SEXE': 'F', 'AGE': 29,
                   'MEMBRE_ACTIF': 'NO'}]
    input_df = pd.DataFrame(input_data)

    model.predict(input_df)


def test_model_performance():
    """Test performance of the model."""
    X_test, y_test = get_test_set()
    model = ChurnModelFinal.load()
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f"f1 {f1_score(y_test, y_pred)}")

    TestCase().assertGreaterEqual(f1, 0.619)
