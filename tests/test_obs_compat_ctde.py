from core.obs_compat import obs_modes_compatible


def test_flat_to_dict_is_incompatible():
    assert obs_modes_compatible("flat", "flat") is True
    assert obs_modes_compatible("dict", "dict") is True
    assert obs_modes_compatible("flat", "dict") is False
    assert obs_modes_compatible("dict", "flat") is False
