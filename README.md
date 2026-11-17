# Value of Flexibility in Dike Design under Uncertain Sea Level Rise

Code of the master's thesis _Value of Flexibility in Climate Adaptation: Flood Protection Design under Sea-Level Rise Uncertainty_ (TU Delft x TUM, MSc Risk and Safety, 2026).

## What the project does

Designing flood protection under sea-level rise uncertainty is challenging due to the large uncertainty underlying sea-level rise projections, but this uncertainty shrinks as sea level is observed over time.
A dike can be designed to anticipate this learning. A **flexible** dike costs less to raise later,
so it can start lower and be raised when needed. An **inflexible** dike costs more to raise later, so
it builds conservatively and raises less often. This project computes how much this flexibility is worth.

The model combines three parts:

- **Sea level rise model**: KNMI23 projections, combined over three emission scenarios, in a lattice
  that represents learning about sea level rise (direct fit method of Völz et al., 2023).
- **Cost model**: the cost of every dike upgrade, for two flexibility concepts:
  - _retention flexibility_: the part of the existing dike that is kept in an upgrade;
  - _land availability_: land reserved for future upgrades before it is built up.
- **Risk model**: the expected annual flood loss of a dike.

Backward induction finds the optimal sequence of dike upgrades. The value of flexibility is the
difference in expected cost between the optimal flexible and the optimal inflexible dike.

## Why the project is useful

- It quantifies the value of flexibility in dike design, including how it depends on costs,
  discounting, flood damage, land use and the way uncertainty is resolved over time.
- Its models are separate parts behind small interfaces, so a new cost, risk or sea level rise model
  can be added without changing the optimization (see `source/decision_analysis.py`).
- The notebooks reproduce every result of the thesis.

## Getting started

1. Install [Python](https://www.python.org/) 3.10 or newer (developed with 3.13) and
   [R](https://www.r-project.org/) 4.5 or newer.
2. Install the R package `PearsonDS`. In an R console, run:

   ```r
   install.packages("PearsonDS")
   ```

3. Clone the repository and create a virtual environment in its root:

   ```bash
   git clone https://github.com/kmheimann/value-of-flexibility-in-climate-adaptation.git
   cd value-of-flexibility-in-climate-adaptation
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Open the notebooks with the `.venv` kernel, with the repository root as working directory (the
   notebooks import `source`), e.g. with `jupyter lab` or VS Code.
5. Run the notebooks in this order. Each builds on the previous ones:

   | Notebook                     | Content                                                     | Code                             |
   | ---------------------------- | ----------------------------------------------------------- | -------------------------------- |
   | `Data.ipynb`                 | KNMI23 data, extrapolation, distribution fits               | `source/data.py`                 |
   | `Sea_Level_Rise_Model.ipynb` | Lattice of sea level rise and learning                      | `source/sea_level_rise_model.py` |
   | `Cost_Model.ipynb`           | Upgrade costs of both flexibility concepts                  | `source/cost_model.py`           |
   | `Risk_Model.ipynb`           | Expected annual flood loss                                  | `source/risk_model.py`           |
   | `Decision_Analysis.ipynb`    | Optimal designs, value of flexibility, sensitivity analysis | `source/decision_analysis.py`    |

   `Decision_Analysis.ipynb` stores the results of slow cells in `cache/` (`%%cache`, see
   `source/cellcache_magic.py`). Its first full run takes several hours, mostly for the sensitivity
   analysis; later runs load the stored results. After changing code in `source/`, delete `cache/`
   to recompute.

A minimal example:

```python
import numpy as np
from source.cost_model import RetentionCostModel
from source.decision_analysis import backward_induction, extract_optimal_designs
from source.risk_model import StandardRiskModel
from source.sea_level_rise_model import DirectFitSeaLevelRiseModel

sea_level_rise_model = DirectFitSeaLevelRiseModel(
    start_year=2025, end_year=2150, step=10, method="mixture", distribution=None,
    weights=[0.25, 0.5, 0.25], extrapolate=True,
)
risk_model = StandardRiskModel(r=2.25, a=0.0, b=1 / np.log(10), D=10e9)
cost_model = RetentionCostModel(c=10e6, v=60e6, phi_minus_1=0.8)
backward_kwargs = {"discount_rate": 0.0225, "h_minus_1": 3.0, "max_possible_h": 10.0, "h_step": 0.1}

_, rewards = backward_induction(sea_level_rise_model, risk_model, cost_model, backward_kwargs)
vof, inflexible, flexible = extract_optimal_designs(rewards)
print(f"Value of flexibility: {vof / 1e6:.1f} M€")
```

## Repository structure

```
data/       KNMI23 sea level rise projections
source/     Model code, one module per notebook
*.ipynb     Notebooks (see above)
```

All code uses SI units (m, years) and euros; plots show heights in cm and costs in M€. Variable
names follow the thesis notation where possible, e.g. `h_minus_1` for $h_{-1}$, `phi` for the
retention flexibility $\tilde{\varphi}$, `lambda_enc` for $\lambda_\mathrm{enc}$.

## Where to get help

- The thesis describes the methods and results in full.
  The docstrings and notebooks refer to thesis sections.
- Every module in `source/` starts with an overview of its contents; every function is documented.
- For questions or problems, open an issue in this repository.

## Who maintains and contributes to the project

The project is maintained by Kaan Heimann, who wrote it as part of the master's thesis at TU Delft and TUM.
Contributions are welcome as issues or pull requests.

## License

The code is released under the [MIT License](LICENSE).

The KNMI23 sea level rise projections in `data/` are provided by the Royal Netherlands Meteorological Institute
(KNMI) and are not covered by this license; their use is subject to KNMI's terms.
