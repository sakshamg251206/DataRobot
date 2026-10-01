import pytest

from autods.core.samples import SAMPLES, load_sample


@pytest.mark.parametrize("key", list(SAMPLES))
def test_samples_load_with_suggested_target(key):
    df = load_sample(key)
    assert len(df) > 100
    assert SAMPLES[key].suggested_target in df.columns
