from thermal_access_pilot.coolwalks_nyc import (
    UWS_1500M,
    _classify_shadow_length_fisher_jenks,
    _tree_attr_columns,
    square_bbox,
)


def test_square_bbox_is_large_enough():
    min_lon, min_lat, max_lon, max_lat = square_bbox(UWS_1500M)
    assert min_lon < UWS_1500M.center_lon < max_lon
    assert min_lat < UWS_1500M.center_lat < max_lat
    assert (max_lon - min_lon) > 0.01
    assert (max_lat - min_lat) > 0.01


def test_tree_attr_column_detection():
    columns = ["FID", "reg", "Hmax,m", "Area,m2"]
    assert _tree_attr_columns(columns) == ("FID", "reg", "Hmax,m", "Area,m2")


def test_shadow_length_classes_use_strict_fisher_jenks_and_keep_zero_separate():
    labels, class_labels = _classify_shadow_length_fisher_jenks(
        [0, 0, 1, 2, 3, 4, 5, 20, 21, 22, 100, 110],
        k=4,
    )
    assert labels[:2] == ["0 m", "0 m"]
    assert labels[2:] == ["0–5 m"] * 5 + ["5–22 m"] * 3 + ["22–100 m", "100–110 m"]
    assert class_labels == ["0 m", "0–5 m", "5–22 m", "22–100 m", "100–110 m"]
