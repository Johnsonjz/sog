import pytest

from sog import Sog


def test_sigma_is_derived_from_r_cut() -> None:
    model = Sog({"use_atomwise": False}, r_cut=3.9785073678160534)
    assert model.sigma == pytest.approx(2.0)


def test_r_cut_overrides_sigma_argument() -> None:
    model = Sog({"use_atomwise": False, "sigma": 100.0, "r_cut": 3.9785073678160534})
    assert model.sigma == pytest.approx(2.0)


def test_non_positive_r_cut_raises() -> None:
    with pytest.raises(ValueError, match="r_cut"):
        Sog({"use_atomwise": False, "r_cut": 0.0})
