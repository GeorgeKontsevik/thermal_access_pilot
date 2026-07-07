# thermal_access_pilot

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![affine](https://img.shields.io/badge/AFFiNE-1E96EB.svg?style={0}&logo=AFFiNE&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/Pytest-0A9EDC.svg?style={0}&logo=Pytest&logoColor=white)

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

thermal_access_pilot is a Python project for heat-aware walking accessibility analysis, combining geospatial surface generation, thermal exposure modeling, and route evaluation into a single pipeline. It is intended for developers and researchers working on urban accessibility or thermal comfort studies who need reproducible map, table, and summary outputs rather than an end-user application. The repository exposes a command-line workflow that runs the full analysis in stages, from loading local inputs through producing thermal, routing, and map artifacts. For runnable steps and configuration details, see Getting Started.

---

## Core Features

- Runs a full heat-aware walking-access pipeline from a single command, producing a complete study run with inputs, thermal fields, routing results, and maps.
- Loads study settings from a TOML configuration, letting developers change the city, model extent, routing thresholds, and penalty scenarios without editing code.
- Builds geospatial outputs for buildings, routes, and exposed walk edges in parquet and JSON formats, making downstream analysis and reuse straightforward.
- Combines thermal raster processing with network-based routing so developers can evaluate how heat exposure changes route choice and travel time.
- Renders and validates map outputs for the study area, giving a quick visual check of inputs, thermal conditions, route examples, and accessibility effects.

---

## Installation

**Prerequisites:** requires Python >=3.12,<3.14

Install thermal_access_pilot using one of the following methods:

**Build from source:**

1. Clone the thermal_access_pilot repository:
```sh
git clone https://github.com/GeorgeKontsevik/thermal_access_pilot
```

2. Navigate to the project directory:
```sh
cd thermal_access_pilot
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Getting Started

### Prerequisites

- Python project with dependencies managed in `pyproject.toml`, `requirements.txt`, and `uv.lock`.
- A configuration file at `configs/kaliningrad.toml` is used by the default CLI entrypoint.

### Run a first execution

1. Install the project dependencies using the repository’s standard Python environment setup.
2. Run the pipeline from the repository root:

```bash
python -m thermal_access_pilot
```

3. To use a different configuration file, pass it with `--config`:

```bash
python -m thermal_access_pilot --config configs/kaliningrad.toml
```

4. If you need to rerun and overwrite existing outputs, add `--force`:

```bash
python -m thermal_access_pilot --force
```

5. Check the output directory defined by the selected config for the generated tables, routes, maps, and summary files.

---

## Architecture

The project is organized as a staged geospatial pipeline with a single CLI entry point. Running the command-line interface loads a TOML configuration, then executes the end-to-end workflow in order: local inputs, external tile retrieval, surface generation, thermal modeling, routing, and map rendering.

At a high level, the pipeline combines three kinds of data:

- **Local study data**: buildings, origins, stops, and a walk graph for the configured city area.
- **External raster inputs**: tiles fetched for the study region and turned into model surfaces.
- **Derived analysis products**: thermal rasters, route results, exposure tables, and map outputs.

The control flow is linear: local inputs are prepared first, external tiles are fetched next, then surfaces are built and passed into the thermal step. The thermal output feeds routing, where edge exposure is attached to the walk graph and origin-to-stop routes are computed for a baseline and configured penalty scenarios. The final stage renders maps and validates the generated map set.

Outputs are written under the configured output directory and include tabular route results, exposed walk edges, route geometries, summary and manifest JSON files, and a set of map images. The repository also contains city-specific modules and tests, suggesting the same core workflow can be adapted to different study areas, but the main architecture is pipeline-oriented rather than service-based.

---

## Documentation

A detailed thermal_access_pilot description is available [here](https://github.com/GeorgeKontsevik/thermal_access_pilot/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/GeorgeKontsevik/thermal_access_pilot/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/GeorgeKontsevik/thermal_access_pilot/tree/main/CONTRIBUTING.md)**: To learn more about making a contribution to thermal_access_pilot.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    GeorgeKontsevik (2026). thermal_access_pilot repository [Computer software]. https://github.com/GeorgeKontsevik/thermal_access_pilot

### BibTeX format:

    @misc{thermal_access_pilot,

        author = {GeorgeKontsevik},

        title = {thermal_access_pilot repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/GeorgeKontsevik/thermal_access_pilot}},

        url = {https://github.com/GeorgeKontsevik/thermal_access_pilot}

    }

---