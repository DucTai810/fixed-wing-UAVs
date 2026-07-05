# Fixed-Wing UAS Telemetry Benchmark

This repository is a reproducible research workspace for fixed-wing UAV telemetry modeling. It includes the cleaned time-series dataset, experiment notebooks, utility scripts, and paper-ready outputs used to compare state-only and actuator-aware learning setups.

## Repository Structure

| Path | Description |
| --- | --- |
| `data/` | Cleaned telemetry dataset used in the experiments. |
| `notebooks/` | Main Jupyter workflows for modeling and analysis. |
| `scripts/` | Reproducible scripts for dashboard rendering and artifact sync. |
| `lncs_figures/` | Generated metrics, summaries, figures, and dashboard output. |
| `paper/` | LaTeX source and paper artifacts. |
| `output/` | Additional generated results and intermediate files. |

## Dataset

The main dataset is:

```text
data/dataset_timeseries_cleaned_v2.csv
```

Each row represents one telemetry time step from a flight. The table includes flight and campaign identifiers, state variables, attitude variables, environmental features, temporal encodings, and derived quantities for sequential modeling.

Example feature groups:

- Position, velocity, and acceleration states
- Roll, pitch, tilt, and airspeed-related variables
- Wind, air density, and ambient temperature
- Elapsed-time encodings
- Flight and campaign metadata

## Experiments

The project contains two primary notebook workflows:

| Experiment | Notebook | Focus |
| --- | --- | --- |
| State/time only | `notebooks/IDF_DS_state-time_only_no_PWM.ipynb` | Models flight behavior without actuator-specific inputs. |
| Actuator aware | `notebooks/IDF_DS_actuator_aware_PWM.ipynb` | Adds actuator-related features and compares their impact. |

Supporting scripts:

```bash
python scripts/render_smartcity_dashboard.py
python scripts/sync_final_gru_artifacts.py
```

## Method Explanation

The benchmark treats UAV telemetry as a sequential learning problem. Instead of analyzing each row independently, the notebooks use time-ordered flight records so the models can learn how aircraft states evolve during a flight.

The state/time-only workflow uses flight dynamics and temporal features as inputs. This setup provides a baseline for understanding how much predictive signal is available from motion, attitude, environment, and elapsed-time information alone.

The actuator-aware workflow adds actuator-related signals to the feature set. Comparing it with the state/time-only workflow helps measure whether control inputs provide additional information for modeling flight behavior.

Evaluation artifacts in `lncs_figures/` summarize model performance, feature importance, uncertainty, energy-related diagnostics, and paper-ready tables or figures. These outputs make it easier to compare experiments and trace final results back to the notebooks or scripts that produced them.

## Setup

Create an environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For dashboard screenshot generation, install the Playwright browser runtime:

```powershell
playwright install chromium



