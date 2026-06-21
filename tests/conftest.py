import copy

import pytest


# A complete, conservative config that comfortably succeeds so tests are stable.
BASE_CFG = {
    "current_age": 40,
    "max_age": 95,
    "starting_amount": 1_500_000,
    "annual_contribution": 55_000,
    "annual_spending": 80_000,
    "spending_reduction_after_75": 0.0,
    "inflation_rate": 0.025,
    "years_already_worked": 15,
    "trials": 2000,
    "include_social_security": True,
    "social_security_claim_age": 67,
    "ss_max_benefit": 24_000,
}


@pytest.fixture
def cfg():
    """Fresh deep copy of the base config for each test."""
    return copy.deepcopy(BASE_CFG)
