"""Phase 1 tests — run with:  pytest -q

Uses the SYNTHETIC fixture, so they pass without the real Kaggle file.
"""
from pathlib import Path

import pytest

from src.data_check import check
from src.load_data import EXPECTED_COLUMNS, RAW_PATH, load_raw

FIXTURE = Path(__file__).parent / "fixtures" / "sample_telco_SYNTHETIC.csv"


@pytest.fixture(scope="module")
def sample():
    return load_raw(FIXTURE)


def test_schema_matches(sample):
    assert list(sample.columns) == EXPECTED_COLUMNS
    assert len(EXPECTED_COLUMNS) == 21


def test_check_reports_blank_totalcharges(sample):
    r = check(sample)
    assert r["missing_columns"] == [] and r["extra_columns"] == []
    zero_tenure = int((sample["tenure"] == 0).sum())
    assert r["totalcharges_unparseable"] == zero_tenure
    assert 0 < r["churn_rate"] < 1
    assert r["duplicate_ids"] == 0


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_raw(tmp_path / "nope.csv")


@pytest.mark.skipif(not RAW_PATH.exists(), reason="real Kaggle CSV not in data/raw/ yet")
def test_real_dataset_shape():
    df = load_raw()
    assert df.shape[1] == 21
    assert df.shape[0] > 7000
