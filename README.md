# Thermal access pilot

Kaliningrad building-level walking access to PT stops with heat exposure along paths.

Run:

```bash
cd /Users/gk/Code/super-duper-disser/thermal_access_pilot
uv run thermal-access-pilot --config configs/kaliningrad.toml --force
```

Key outputs:

- `outputs/kaliningrad/thermal/headline_utci.tif`
- `outputs/kaliningrad/routes/routes.parquet`
- `outputs/kaliningrad/tables/building_results.parquet`
- `outputs/kaliningrad/maps/*.png`
- `outputs/kaliningrad/summary.json`

The default Kaliningrad pilot threshold is UTCI > 29 °C because the selected real 2025
summer archive hour reaches UTCI max ≈30.4 °C; UTCI > 32 °C is also reported in
thermal metadata and is zero for this run.

Caveat: the pilot is heat-only. Cold-wind/URock and PALM-4U are deferred in `aggregated_spatial_pipeline/BACKLOG.md`.
