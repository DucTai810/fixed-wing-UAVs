# Fixed-wing UAV telemetry benchmark

This repository contains the implementation, notebooks, and supporting scripts for a fixed-wing UAV telemetry analysis project. The work focuses on learning from time-series flight data, comparing different modeling approaches, and generating figures and evaluation artifacts for research reporting.

## Project overview

The project uses flight telemetry data collected from fixed-wing UAV operations. Its main purpose is to study how well machine learning models can learn temporal patterns from flight behavior and support downstream analysis for monitoring, diagnostics, and research reporting.

The main objectives are to:

- preprocess and clean raw telemetry sequences,
- construct a structured time-series dataset suitable for learning,
- build predictive models for flight-state behavior,
- compare state-only and actuator-aware modeling setups,
- evaluate model quality with regression metrics and diagnostic summaries,
- generate plots, performance reports, and paper-ready artifacts.

The repository includes both exploratory notebooks and reproducible scripts for producing the final results.

## What is included

- `data/` — cleaned telemetry dataset used throughout the experiments.
- `notebooks/` — Jupyter notebooks for the main analysis workflows.
- `scripts/` — Python scripts for reproducing figures and final artifact generation.
- `paper/` — LaTeX source and related paper artifacts.
- `lncs_figures/` — generated figures, summary tables, and visualization outputs.
- `output/` — additional generated results and intermediate outputs.

## Main experiments

The repository currently supports two main analysis directions:

1. State-only modeling
   - Notebook: `notebooks/IDF_DS_state-time_only_no_PWM.ipynb`
   - Focuses on using flight-state variables without actuator-specific inputs.

2. Actuator-aware modeling
   - Notebook: `notebooks/IDF_DS_actuator_aware_PWM.ipynb`
   - Includes actuator-related features and evaluates whether they improve the learned representation.

In addition, the repository contains scripts to create the final dashboard rendering and to synchronize the GRU-related experiment outputs.

## Data description

The dataset is a cleaned time-series telemetry table designed for sequential modeling. Each row represents a time step from one flight, and the data is organized so that models can learn from both current state and recent history.

### What the data contains

The telemetry table includes variables such as:

- position and velocity states: `pos_vx`, `pos_vy`, `pos_vz`, `speed_horizontal`,
- acceleration and attitude variables: `pos_ax`, `pos_ay`, `pos_az`, `roll`, `pitch`, `tilt_angle`,
- aerodynamic and environmental variables: `true_airspeed`, `headwind_component`, `air_rho`, `air_ambient_temperature`,
- temporal features derived from elapsed time: sinusoidal/cosine encodings and log-scaled elapsed time,
- metadata identifying each flight and campaign: `flight_id`, `flight_num`, `campaign_id`.

### Why this dataset is useful

This dataset is useful because it captures both the dynamic behavior of the aircraft and the temporal evolution of its states over time. That makes it suitable for:

- time-series forecasting,
- behavior analysis,
- anomaly or deviation detection,
- model comparison studies across different feature sets.

The data is expected to be stored in `data/dataset_timeseries_cleaned_v2.csv`.

## Environment setup

1. Clone the repository.
2. Create a Python environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. If you want to generate the Playwright-based dashboard screenshot, install the browser runtime:

```bash
playwright install chromium
```

## How to run the project

### Run the scripts

```bash
python scripts/render_smartcity_dashboard.py
python scripts/sync_final_gru_artifacts.py
```

### Open the notebooks

You can open the notebooks in Jupyter Notebook or Jupyter Lab:

```bash
jupyter lab
```

## Notes for reproducibility

- The project assumes the cleaned dataset is present in `data/`.
- Generated outputs such as figures and summaries are stored under `lncs_figures/` and `output/`.
- For large datasets or generated artifacts, consider storing them separately from the repository or using Git LFS.

## Repository purpose

This repository is intended as a reproducible research workspace for exploring fixed-wing UAV telemetry modeling and producing publication-ready outputs. It is especially useful for researchers or students who want to understand how telemetry data can be transformed into a structured machine learning problem, how different feature sets affect model performance, and how experiments can be packaged into a clean, shareable GitHub project.
