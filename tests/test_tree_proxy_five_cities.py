from thermal_access_pilot.tree_proxy_five_cities import (
    _class_midpoint,
    _radius_from_crown_width_class,
)


def test_class_midpoint_parses_common_height_labels():
    assert _class_midpoint("0-5 m") == 2.5
    assert _class_midpoint("2-4 m") == 3.0
    assert _class_midpoint("<6") == 3.0
    assert _class_midpoint(">20") == 20.0


def test_radius_from_crown_width_class_uses_half_of_width_midpoint():
    assert _radius_from_crown_width_class("2-4 m") == 1.5
