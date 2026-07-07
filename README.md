# thermal_access_pilot

Heat-aware walking access and service routing pilot.

## Scheme

```mermaid
flowchart LR
    A[Inputs] --> B[Run: src/thermal_access_pilot/__main__.py]
    B --> C[Checked outputs]
    C --> D[Paper / thesis use]
```

## Main Result

![Main result](outputs/batch_service_access_hottest_summer2025/four_city_heat_composite_polyclinic.png)

## Run

Entrypoint: `src/thermal_access_pilot/__main__.py`

Human:

```bash
uv run thermal-access-pilot --config configs/kaliningrad.toml --force
```

Agent:

Inspect maps, parquet row counts, and summary JSON after each run.

## Publication

No standalone publication yet; thesis integration in parent repo.

## Next Steps / Heuristics

Heuristic: heat-only UTCI path is current scope; wind/URock/PALM are deferred until validated.
