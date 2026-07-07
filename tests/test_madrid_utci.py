from thermal_access_pilot.madrid_utci import _tile_codes_for_lonlat, utci_category


def test_tile_codes_for_madrid():
    codes = _tile_codes_for_lonlat(latitude=40.4168, longitude=-3.7038)
    assert codes["srtm_ns"] == "N40"
    assert codes["srtm_tile"] == "N40W004"
    assert codes["cover_tile"] == "N39W006"


def test_utci_category_thresholds():
    assert utci_category(8.9) == "cold_stress"
    assert utci_category(9.0) == "no_thermal_stress"
    assert utci_category(26.0) == "moderate_heat_stress"
    assert utci_category(32.0) == "strong_heat_stress"
    assert utci_category(38.0) == "very_strong_heat_stress"
    assert utci_category(46.1) == "extreme_heat_stress"
