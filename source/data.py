"""
data.py — KNMI23 sea level rise projections and fitted quantile functions (thesis Appendix A4).

Pipeline used in the thesis:
    1. `load_data`: quantiles of sea level rise per year (1995-2100, plus 2150 and 2300) for the
       scenarios SSP1-2.6, SSP2-4.5 and SSP5-8.5, from `data/Data_SLR_KNMI23_ref_period_1995_2014.xlsx`.
    2. `extrapolate_quantiles`: quadratic extrapolation of every scenario and percentile to 2300.
    3. `get_aggregated_quantile_fn(..., method="mixture")`: a Pearson type IV distribution fitted per
       scenario and year, combined into a weighted mixture of the scenarios (weights w). Returns a
       quantile function q(t, prob) [m].

`get_aggregated_quantile_fn_cached` stores fits in memory, so that sea level rise models with the same
calibration share one fit. The `plot_*` functions show the data and the fits.

Requires R with the package PearsonDS (called through rpy2) for the Pearson type IV distribution.
Units: sea level rise in m (plots in cm), time in years.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeWarning, brentq, curve_fit
from scipy.stats import chi2, expon, johnsonsu, lognorm, norm, pareto

pearsonds = importr("PearsonDS")  # Import R's Pearson Distribution System package

_DATA_FILE = Path("data") / "Data_SLR_KNMI23_ref_period_1995_2014.xlsx"
_PERCENTILE_LEVELS = [1, 5, 10, 17, 20, 50, 80, 83, 90, 95, 99]


def load_data() -> dict:
    """Loads the KNMI23 quantiles of sea level rise (no extrapolation).

    Every year 1995-2100 has all three scenarios and percentile levels. The years 2150 and 2300 only
    have SSP1-2.6 and SSP5-8.5 at the 17th, 50th and 83rd percentile (no median in 2300).

    Returns:
        Dict year -> scenario ("ssp126", "ssp245", "ssp585") -> (quantiles [m], probabilities [-]),
        e.g. `q, p = data[2100]["ssp585"]`.
    """
    df = pd.read_excel(_DATA_FILE)
    time_steps = np.floor(df["time"].values).astype(int).tolist()

    def get_percentiles(ssp, idx, levels):
        return (
            np.array([df[f"{ssp}_{level}"][idx] for level in levels])
            / 100.0  # Convert cm to m
        )

    empirical_quantiles: dict = {}
    for idx, t in enumerate(time_steps):
        for ssp in ["ssp126", "ssp245", "ssp585"]:
            q = get_percentiles(ssp, idx, _PERCENTILE_LEVELS)
            p = np.array(_PERCENTILE_LEVELS) / 100
            empirical_quantiles.setdefault(t, {})[ssp] = (q, p)

    # Incorporate the actual (sparse, non-extrapolated) benchmark quantiles
    # tabulated for years beyond 2100.
    for t, per_ssp in _load_future_benchmark_data().items():
        empirical_quantiles.setdefault(t, {}).update(per_ssp)

    return empirical_quantiles


def extrapolate_quantiles(data: dict) -> dict:
    """Extends `data` (from `load_data`) to every year until 2300.

    For every scenario and percentile level, a quadratic trend is fitted to 1995-2100 and evaluated
    for 2101-2300. This replaces the sparse KNMI values for 2150 and 2300 (compare them with
    `plot_extrapolation_validation`). Percentile levels are extrapolated independently, so their
    order is not enforced.

    Returns:
        A new dict with the structure of `data`, for every year 1995-2300.
    """
    end_year = 2300
    years_hist = sorted(t for t in data if t <= 2100)
    t_hist = np.array(years_hist, dtype=float) + 0.5

    extrapolated: dict = {t: dict(data[t]) for t in years_hist}

    for ssp in ["ssp126", "ssp245", "ssp585"]:
        q_hist = np.array([data[t][ssp][0] for t in years_hist])  # shape (n_years, n_levels) # fmt: skip
        p = data[years_hist[0]][ssp][1]

        coeffs_per_level = [
            np.polyfit(t_hist, q_hist[:, i], 2) for i in range(q_hist.shape[1])
        ]

        for year in range(2101, end_year + 1):
            t_query = year + 0.5
            q = np.array([np.polyval(c, t_query) for c in coeffs_per_level])

            extrapolated.setdefault(year, {})[ssp] = (q, p)

    return extrapolated


def _load_future_benchmark_data() -> dict:
    """Loads the KNMI quantiles after 2100 (sheet "Data to 2300"): SSP1-2.6 and SSP5-8.5 at the 17th,
    50th and 83rd percentile, in the structure of `load_data`; missing values are dropped."""
    df = pd.read_excel(_DATA_FILE, sheet_name="Data to 2300")
    df = df.iloc[:5]  # First table only (values in meters); rows below repeat the same data in cm behind a text separator. # fmt: skip

    levels = [17, 50, 83]
    benchmarks: dict = {}
    for _, row in df.iterrows():
        year = int(np.floor(row["time"]))
        if year <= 2100:
            continue
        for ssp in ["ssp126", "ssp585"]:
            q_vals, p_vals = [], []
            for level in levels:
                val = row[f"{ssp}_{level}"]
                if pd.notna(val):
                    q_vals.append(float(val))
                    p_vals.append(level / 100)
            if q_vals:
                benchmarks.setdefault(year, {})[ssp] = (np.array(q_vals), np.array(p_vals))  # fmt: skip

    return benchmarks


_SSP_ORDER = ["ssp126", "ssp245", "ssp585"]


def get_aggregated_quantile_fn(
    start_year: int,
    end_year: int,
    method: str = "uncertainty_band",
    distribution: str | None = None,
    weights: list[float] | None = None,
    extrapolate: bool = False,
) -> tuple[Callable[[int, float], float], dict]:
    """Quantile function of sea level rise that combines the three SSP scenarios.

    Args:
        start_year: First year of the quantile function [yr].
        end_year: Last year of the quantile function [yr].
        method: How the scenarios are combined:
            - "mixture" (thesis): a distribution is fitted to every scenario; the quantile function
              inverts the weighted mixture of their CDFs.
            - "uncertainty_band": a normal distribution whose 1st and 99th percentile are the 5th
              percentile of SSP1-2.6 and the 95th percentile of SSP5-8.5.
        distribution: Fitted distribution; None for the default ("pearsonIV" for "mixture",
            "normal" for "uncertainty_band").
        weights: Scenario weights (SSP1-2.6, SSP2-4.5, SSP5-8.5) for "mixture"; they must sum to 1.
            Defaults to equal weights (the thesis uses 0.25, 0.5, 0.25).
        extrapolate: Use `extrapolate_quantiles` (needed for years after 2100).

    Returns:
        (quantile_fn, infos): quantile_fn(t, prob) [m] with t in years after start_year, and fit
        diagnostics per t.
    """
    if method not in ("uncertainty_band", "mixture"):
        raise ValueError(f"Invalid method '{method}'. Choose from 'uncertainty_band', 'mixture'.")  # fmt:skip

    if distribution is not None and distribution not in (
        "best",
        "normal",
        "chi2",
        "exponential",
        "lognormal",
        "pareto",
        "johnsonsu",
        "pearsonIV",
    ):
        raise ValueError(
            f"Invalid distribution '{distribution}'. Choose from 'pearsonIV', 'normal', 'lognormal', 'johnsonsu', 'chi2', 'exponential', 'pareto', or None (defaults to 'normal' for 'uncertainty_band' and 'pearsonIV' for 'mixture')."
        )

    if not extrapolate and end_year > 2100 and method == "uncertainty_band":
        raise ValueError(
            f"Cannot use uncertainty band aggregation rule for end_year={end_year} without extrapolation, since the 5th and 95th quantiles are not provided. "
            "Use extrapolate=True to extend the data beyond 2100."
        )

    if method == "mixture":
        weights_list = [1 / 3, 1 / 3, 1 / 3]
        if weights is not None:
            weights_list = weights
        if len(weights_list) != 3:
            raise ValueError(f"'weights' must have 3 elements (one per SSP scenario: {_SSP_ORDER}), got {len(weights_list)}.")  # fmt:skip
        if any(w < 0 for w in weights_list):
            raise ValueError(f"'weights' must be non-negative, got {weights_list}.")
        if not np.isclose(sum(weights_list), 1.0):
            raise ValueError(f"'weights' must sum to 1, got {weights_list} (sum={sum(weights_list)}).")  # fmt:skip

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    cache = {}  # Cache fitted functions indexed by t_i
    infos = {}  # Store uncertainty band and fitted parameters for each t_i

    # Build the cache
    for t in [t for t in data if start_year <= t <= end_year]:
        q_ssp126, p_levels = data[t]["ssp126"]
        if not extrapolate and t > 2100:
            q_ssp245 = None
        else:
            q_ssp245, _ = data[t]["ssp245"]
        q_ssp585, _ = data[t]["ssp585"]

        if method == "uncertainty_band":
            # Define the uncertainty band as interval between the 5th percentile of SSP126 and the 95th percentile of SSP585
            uncertainty_band = [q_ssp126[p_levels == 0.05][0], q_ssp585[p_levels == 0.95][0]]  # fmt: skip
            if distribution is None or distribution == "normal":
                # Fit a normal distribution to the uncertainty band assuming that the lower bound corresponds to the 1st percentile and the upper bound corresponds to the 99th percentile
                mu = (uncertainty_band[0] + uncertainty_band[1]) / 2  # The Normal distribution is symmetric, so its 1st and 99th percentiles are equidistant from the mean. # fmt: skip

                # The quantile function of a normal distribution is given as: q(p) = mu + sigma * PHI^-1(p), where PHI^-1 is the inverse CDF of the standard normal distribution
                # => q(0.99) = mu + sigma * PHI^-1(0.99)
                # <=> sigma = (q(0.99) - mu) / PHI^-1(0.99)
                sigma = (uncertainty_band[1] - mu) / norm.ppf(0.99)

                # Bind mu/sigma at definition time to avoid late-binding closure over loop vars.
                def q_func_uncertainty_band(p: float, loc: float = float(mu), scale: float = float(sigma)) -> float:  # fmt: skip
                    return float(norm.ppf(p, loc=loc, scale=scale))

                q_func = q_func_uncertainty_band

                info = {
                    "uncertainty_band": (
                        float(uncertainty_band[0]),
                        float(uncertainty_band[1]),
                    ),
                    "mu": float(mu),
                    "sigma": float(sigma),
                }
            else:
                _, info = _parametric(
                    np.array([0.01, 0.99]),
                    np.array(uncertainty_band),
                    distribution=distribution,
                )

                fit = info.get("selected", None)

                if fit is None:
                    raise RuntimeError(
                        f"Failed to fit {distribution} to uncertainty band at t={t}."
                    )

        elif method == "mixture":
            if not extrapolate and t > 2100:
                q_by_ssp = {"ssp126": q_ssp126, "ssp585": q_ssp585}
                normalized_weights = [
                    weights_list[0] / (weights_list[0] + weights_list[2]),
                    weights_list[2] / (weights_list[0] + weights_list[2]),
                ]  # Renormalize weights for the two remaining SSPs
            else:
                q_by_ssp = {"ssp126": q_ssp126, "ssp245": q_ssp245, "ssp585": q_ssp585}
                normalized_weights = weights_list

            # Fit a distribution to each SSP scenario individually.
            fits = {}
            for ssp in q_by_ssp:
                _, info = _parametric(
                    p_levels,
                    q_by_ssp[ssp],
                    distribution=distribution
                    if distribution is not None
                    else "pearsonIV",
                )

                fit = info.get("selected", None)

                if fit is None:
                    raise RuntimeError(
                        f"{distribution} fit failed for '{ssp}' at t={t}."
                    )
                fits[ssp] = fit

            cdf_fns = [fits[ssp]["cdf_fn"] for ssp in q_by_ssp]
            ppf_fns = [fits[ssp]["ppf_fn"] for ssp in q_by_ssp]

            # Bind cdf_fns/ppf_fns/weights at definition time to avoid late-binding closure over loop vars.
            def q_func_mixture(
                p: float,
                cdf_fns: list[Callable] = cdf_fns,
                ppf_fns: list[Callable] = ppf_fns,
                weights: list[float] = normalized_weights,
            ) -> float:
                p = float(np.clip(p, 1e-9, 1 - 1e-9))

                def mixture_cdf(x: float) -> float:
                    return sum(w * cdf_fn(x) for w, cdf_fn in zip(weights, cdf_fns))

                # Bracket the root using the individual scenarios' extreme quantiles, widened
                # slightly to guard against floating-point edge effects.
                lo = min(fn(1e-9) for fn in ppf_fns)
                hi = max(fn(1 - 1e-9) for fn in ppf_fns)
                span = hi - lo
                lo -= 0.01 * span
                hi += 0.01 * span

                root = brentq(lambda x: mixture_cdf(x) - p, lo, hi, xtol=1e-10)  # fmt: skip
                return float(root)  # type: ignore[arg-type]

            q_func = q_func_mixture

            info = {
                "weights": normalized_weights,
                "fits": {
                    ssp: {
                        "params": fits[ssp]["params"],
                        "mse": fits[ssp]["mse"],
                        "mae": fits[ssp]["mae"],
                    }
                    for ssp in q_by_ssp
                },
            }

        cache[t - start_year] = q_func

        infos[t - start_year] = info

    def quantile_fn(t: int, p: float) -> float:
        return cache[t](p)

    return quantile_fn, infos


_quantile_fn_cache: dict[tuple, tuple[Callable[[int, float], float], dict]] = {}


def get_aggregated_quantile_fn_cached(
    start_year: int,
    end_year: int,
    method: str = "uncertainty_band",
    distribution: str | None = None,
    weights: list[float] | None = None,
    extrapolate: bool = False,
) -> tuple[Callable[[int, float], float], dict]:
    """`get_aggregated_quantile_fn` with results cached in memory (to avoid a repeated refitting which is slow)."""
    key = (start_year, end_year, method, distribution, tuple(weights) if weights is not None else None, extrapolate)  # fmt: skip
    if key not in _quantile_fn_cache:
        _quantile_fn_cache[key] = get_aggregated_quantile_fn(start_year, end_year, method=method, distribution=distribution, weights=weights, extrapolate=extrapolate)  # fmt: skip
    return _quantile_fn_cache[key]


def clear_quantile_fn_cache() -> None:
    """Frees the cache of `get_aggregated_quantile_fn_cached`."""
    _quantile_fn_cache.clear()


def get_scenario_quantile_fn(
    start_year: int,
    end_year: int,
    ssp: str,
    method: str = "parametric",
    distribution: str = "best",
    criterion: str = "mse",
    extrapolate: bool = False,
) -> tuple[Callable[[int, float], float], dict]:
    """Quantile function of sea level rise for one SSP scenario.

    Args:
        start_year, end_year: Years of the quantile function [yr].
        ssp: Scenario, "ssp126", "ssp245" or "ssp585".
        method, distribution, criterion: See `get_continuous_quantile_fn_from_data`.
        extrapolate: Use `extrapolate_quantiles` (needed for years after 2100).

    Returns:
        (quantile_fn, infos): quantile_fn(t, prob) [m] with t in years after start_year, and fit
        diagnostics per t.
    """

    if ssp not in ["ssp126", "ssp245", "ssp585"]:
        raise ValueError(
            f"Invalid SSP '{ssp}'. Choose from 'ssp126', 'ssp245', 'ssp585'."
        )

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    cache = {}  # Cache fitted functions by t_i
    infos = {}  # Store fit diagnostics for each t_i

    # Build the cache
    for t in [t for t in data if start_year <= t <= end_year]:
        if not extrapolate and t > 2100 and ssp == "ssp245":
            # The KNMI23 data does not provide any ssp245 quantiles beyond 2100, so we cannot fit a quantile function for those years.
            continue
        quantiles, levels = data[t][ssp]
        q_func, info = get_continuous_quantile_fn_from_data(
            levels,
            quantiles,
            method=method,
            criterion=criterion,
            distribution=distribution,
        )
        cache[t - start_year] = q_func
        infos[t - start_year] = info

    def quantile_fn(t: int, p: float) -> float:
        return cache[t](p)

    return quantile_fn, infos


def get_continuous_quantile_fn_from_data(
    p_emp: np.ndarray,
    q_emp: np.ndarray,
    method: str = "linear",
    distribution: str = "best",
    criterion: str = "mse",
) -> tuple[Callable[[float], float], dict | None]:
    """Continuous quantile function q(prob) from a discrete set of quantiles of a continuous distribution.

    Args:
        p_emp: Cumulative probability levels, ascending in [0, 1].
        q_emp: Quantiles at these levels.
        method: "nearest" (nearest level), "linear" or "quadratic" (interpolation), or
            "parametric" (fitted distribution). Probabilities outside the levels are clipped.
        distribution: For "parametric": the name of a distribution ("normal", "lognormal",
            "pearsonIV", ...), or "best" to select the best fit by `criterion`.
        criterion: "mse" or "mae", the error measure used to select the best fit.

    Returns:
        (fn, info): the quantile function fn(prob), and for "parametric" a dict with the fits
        ("best", "best_by_mse", "best_by_mae", "all_fits", "criteria_agree"), otherwise None.

    Remarks:
        - Only for continuous distributions. The quantile function of a discrete distribution (e.g.
          of a lattice) is `sea_level_rise_model.get_generalized_inverse_fn`.
    """
    p_emp = np.asarray(p_emp, dtype=float)
    q_emp = np.asarray(q_emp, dtype=float)

    if len(p_emp) < 2 and method == "linear":
        raise ValueError(
            "Linear interpolation requires at least 2 empirical points; "
            f"got {len(p_emp)} points."
        )

    if len(p_emp) < 3 and method == "quadratic":
        raise ValueError(
            "Quadratic interpolation requires at least 3 empirical points; "
            f"got {len(p_emp)} points."
        )

    if method == "nearest":
        return _nearest(p_emp, q_emp)
    elif method == "linear":
        return _interpolated(p_emp, q_emp, kind="linear")
    elif method == "quadratic":
        return _interpolated(p_emp, q_emp, kind="quadratic")
    elif method == "parametric":
        return _parametric(p_emp, q_emp, distribution=distribution, criterion=criterion)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'nearest', 'linear', 'quadratic', 'parametric'."
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _PearsonIV:
    """Wrapper for the Pearson type IV distribution from R's PearsonDS package.

    Exposes ppf/cdf/pdf methods compatible with the scipy.stats-like interface
    used elsewhere in this module. Parameters, in PearsonDS order:
        m        : first shape parameter (m > 0.5)
        nu       : second shape parameter / skewness (nu != 0)
        location : location parameter
        scale    : scale parameter (scale > 0)
    """

    @staticmethod
    def _call(
        r_func,
        x: np.ndarray | float,
        m: float,
        nu: float,
        location: float,
        scale: float,
    ) -> np.ndarray | float:
        """Call an R PearsonDS function.

        Returns an array if ``x`` is array-like, or a plain float if ``x`` is scalar
        (matching scipy.stats distribution conventions, e.g. norm.ppf/cdf/pdf).
        """
        x_in = np.asarray(x, dtype=float)
        x_arr = np.atleast_1d(x_in)
        result = r_func(
            FloatVector(x_arr),
            m=float(m),
            nu=float(nu),
            location=float(location),
            scale=float(scale),
        )
        out = np.asarray(result, dtype=float)
        return float(out[0]) if x_in.ndim == 0 else out

    def ppf(
        self, p: np.ndarray | float, m: float, nu: float, location: float, scale: float
    ) -> np.ndarray | float:
        """Quantile function (percent point function)."""
        return self._call(pearsonds.qpearsonIV, p, m, nu, location, scale)

    def cdf(
        self, x: np.ndarray | float, m: float, nu: float, location: float, scale: float
    ) -> np.ndarray | float:
        """Cumulative distribution function."""
        return self._call(pearsonds.ppearsonIV, x, m, nu, location, scale)

    def pdf(
        self, x: np.ndarray | float, m: float, nu: float, location: float, scale: float
    ) -> np.ndarray | float:
        """Probability density function."""
        return self._call(pearsonds.dpearsonIV, x, m, nu, location, scale)


def _nearest(p_emp: np.ndarray, q_emp: np.ndarray) -> tuple[Callable, None]:
    """Snap to the nearest available quantile level.

    Implicitly clips at the boundary: any p outside [p_emp[0], p_emp[-1]]
    maps to the first or last available quantile.
    """

    def fn(p: float) -> float:
        idx = int(np.argmin(np.abs(p_emp - p)))
        return float(q_emp[idx])

    return fn, None


def _interpolated(
    p_emp: np.ndarray,
    q_emp: np.ndarray,
    kind: str,
) -> tuple[Callable, None]:
    """Piecewise interpolation; clips at boundary via fill_value."""

    if len(p_emp) < 2 and kind == "linear":
        raise ValueError(
            "Linear interpolation requires at least 2 empirical points; "
            f"got {len(p_emp)} points."
        )

    if len(p_emp) < 3 and kind == "quadratic":
        raise ValueError(
            "Quadratic interpolation requires at least 3 empirical points; "
            f"got {len(p_emp)} points."
        )

    f = interp1d(
        p_emp,
        q_emp,
        kind=kind,
        bounds_error=False,
        fill_value=(float(q_emp[0]), float(q_emp[-1])),  # type: ignore
    )

    def fn(p: float) -> float:
        return float(f(p))

    return fn, None


def _parametric(
    p_emp: np.ndarray,
    q_emp: np.ndarray,
    distribution: str = "best",
    criterion: str = "mse",
    verbose: bool = False,
) -> tuple[Callable, dict]:
    """Fit parametric distribution specified in ``distribution``."""

    if distribution not in (
        "best",
        "normal",
        "chi2",
        "exponential",
        "lognormal",
        "pareto",
        "johnsonsu",
        "pearsonIV",
    ):
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'best', 'normal', 'chi2', 'exponential', "
            "'lognormal', 'pareto', 'johnsonsu', 'pearsonIV'."
        )
    if distribution == "best" and criterion not in ("mse", "mae"):
        raise ValueError(f"Unknown criterion '{criterion}'. Choose 'mse' or 'mae'.")

    if distribution != "best":
        all_fits = _fit_distributions(p_emp, q_emp, distribution=distribution)
    else:
        all_fits = _fit_distributions(p_emp, q_emp, distribution="all")

    if not all_fits:
        raise RuntimeError(
            "Distribution fits failed; no valid callable can be returned."
        )

    if distribution != "best":
        selected = next((f for f in all_fits if f["name"] == distribution), None)

        if selected is None:
            raise ValueError(f"Failed to fit '{distribution}'.")

        info = {
            "selected": selected,
            "mse": selected["mse"],  # type: ignore
            "mae": selected["mae"],  # type: ignore
        }
    else:
        best_by_mse = min(all_fits, key=lambda r: r["mse"])
        best_by_mae = min(all_fits, key=lambda r: r["mae"])
        criteria_agree = best_by_mse["name"] == best_by_mae["name"]
        selected = best_by_mse if criterion == "mse" else best_by_mae
        info = {
            "selected": selected,
            "mse": selected["mse"],
            "mae": selected["mae"],
            "best_by_mse": best_by_mse,
            "best_by_mae": best_by_mae,
            "criterion": criterion,
            "criteria_agree": criteria_agree,
            "all_fits": all_fits,
        }

        if not criteria_agree and verbose:
            print(
                f"[data.py] Note: MSE and MAE criteria select different distributions.\n"
                f"  MSE → {best_by_mse['name']:<12}  "
                f"MSE={best_by_mse['mse']:.4g},  MAE={best_by_mse['mae']:.4g}\n"
                f"  MAE → {best_by_mae['name']:<12}  "
                f"MSE={best_by_mae['mse']:.4g},  MAE={best_by_mae['mae']:.4g}\n"
                f"  Using criterion='{criterion}' → selecting '{selected['name']}'"
            )

    return selected["ppf_fn"], info  # type: ignore


def _fit_distributions(
    p_emp: np.ndarray, q_emp: np.ndarray, distribution: str = "all"
) -> list[dict]:
    """Fit all candidate distributions to (p_emp, q_emp) and return result dicts.

    Args:
        p_emp (np.ndarray): Empirical cumulative probabilities.
        q_emp (np.ndarray): Empirical quantiles corresponding to p_emp.
        distribution (str): Specific distribution to fit. If "all", fits all distributions, else fits the specified distribution.

    Returns:
        list[dict]: List of dictionaries containing fit results for each candidate distribution.
    """

    if distribution not in (
        "all",
        "normal",
        "chi2",
        "exponential",
        "lognormal",
        "pareto",
        "johnsonsu",
        "pearsonIV",
    ):
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'all', 'normal', 'chi2', 'exponential', "
            "'lognormal', 'pareto', 'johnsonsu', 'pearsonIV'."
        )

    med = float(np.interp(0.50, p_emp, q_emp))
    spread = float(np.interp(0.84, p_emp, q_emp) - np.interp(0.16, p_emp, q_emp)) / 2
    spread = max(spread, 1e-3)  # guard against degenerate data

    def _make_ppf(dist, params: np.ndarray) -> Callable[[float], float]:
        """Closure factory — binds dist and params at fit time."""
        _params = tuple(params)
        return lambda p: float(dist.ppf(np.clip(p, 1e-9, 1 - 1e-9), *_params))

    def _make_cdf(dist, params: np.ndarray) -> Callable:
        _params = tuple(params)
        return lambda x: dist.cdf(x, *_params)

    def _make_pdf(dist, params: np.ndarray) -> Callable:
        _params = tuple(params)
        return lambda x: dist.pdf(x, *_params)

    all_candidates = [
        {
            "name": "normal",
            "curve_fn": lambda x, loc, scale: norm.cdf(x, loc, scale),
            "p0": [med, spread],
            "dist": norm,
        },
        {
            "name": "chi2",
            "curve_fn": lambda x, df, loc, scale: chi2.cdf(x, df, loc, scale),
            "p0": [2.0, 0.0, 1.0],
            "dist": chi2,
        },
        {
            "name": "exponential",
            "curve_fn": lambda x, loc, scale: expon.cdf(x, loc, scale),
            "p0": [0.0, spread],
            "dist": expon,
        },
        {
            "name": "lognormal",
            "curve_fn": lambda x, s, loc, scale: lognorm.cdf(x, s, loc, scale),
            "p0": [0.5, 0.0, spread],
            "dist": lognorm,
        },
        {
            "name": "pareto",
            "curve_fn": lambda x, b, loc, scale: pareto.cdf(x, b, loc, scale),
            "p0": [1.0, 0.0, 1.0],
            "dist": pareto,
        },
        {
            "name": "johnsonsu",
            "curve_fn": lambda x, a, b, loc, scale: johnsonsu.cdf(x, a, b, loc, scale),
            "p0": [0.0, 1.0, med, spread],
            "dist": johnsonsu,
        },
        {
            "name": "pearsonIV",
            "curve_fn": lambda x, m, nu, location, scale: _PearsonIV().cdf(
                x, m, nu, location, scale
            ),
            "p0": [5.0, 0.1, med, spread],
            "bounds": (
                [0.5 + 1e-6, -np.inf, -np.inf, 1e-6],
                [np.inf, np.inf, np.inf, np.inf],
            ),
            "dist": _PearsonIV(),
        },
    ]

    if distribution != "all":
        candidates = [c for c in all_candidates if c["name"] == distribution]
    else:
        candidates = all_candidates

    results = []
    for c in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", OptimizeWarning
                )  # The covariance is not used
                popt, _ = curve_fit(
                    c["curve_fn"],
                    q_emp,
                    p_emp,
                    p0=c["p0"],
                    bounds=c.get("bounds", (-np.inf, np.inf)),
                    maxfev=10_000,
                )
            p_pred = c["curve_fn"](q_emp, *popt)
            mse = float(np.mean((p_pred - p_emp) ** 2))
            mae = float(np.mean(np.abs(p_pred - p_emp)))
            results.append(
                {
                    "name": c["name"],
                    "params": popt,
                    "mse": mse,
                    "mae": mae,
                    "ppf_fn": _make_ppf(c["dist"], popt),
                    "cdf_fn": _make_cdf(c["dist"], popt),
                    "pdf_fn": _make_pdf(c["dist"], popt),
                }
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Fitting '{c['name']}' failed: {exc}",
                RuntimeWarning,
                stacklevel=4,
            )

    return results


# ---------------------------------------------------------------------------
# Static plotting helpers for visualizing the data and fitted distributions
# ---------------------------------------------------------------------------


def plot_data(
    t: int, extrapolate: bool = False, axes: plt.axes.Axes = None
) -> plt.figure.Figure:
    """
    Plot empirical quantiles for a given time step and SSP scenario.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        if not extrapolate and t > 2100 and ssp == "ssp245":
            continue  # No ssp245 data beyond 2100 in the raw dataset.

        q, p = data[t][ssp]
        q = q * 100  # Convert to cm

        # CDF
        axes[0].plot(q, p, marker="o", markersize=4, linestyle="dashed", label=label, color=color)  # fmt:skip

        # Approximate PDF via finite differences: dp/dq
        pdf_approx = np.diff(p) / np.diff(q)
        q_mid = (q[:-1] + q[1:]) / 2
        axes[1].plot(
            q_mid,
            pdf_approx,
            marker="o",
            markersize=4,
            linestyle="dashed",
            label=label,
            color=color,
        )

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF (approximate)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    plt.suptitle(f"Sea Level Rise Distributions at t={t} per SSP scenario")
    plt.tight_layout()
    plt.show()

    return fig


def plot_extrapolation_validation(
    start_year: int = 2025,
    end_year: int = 2300,
    axes: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Visually validate the quadratic extrapolation of quantiles beyond 2100.

    For each requested SSP scenario, plots the extrapolated quantile
    trajectory produced by ``extrapolate_quantiles`` (quadratic trend fitted
    on 1995-2100 data) against the actual sparse benchmark quantiles that
    ``load_data`` incorporates from the KNMI23 workbook's "Data to 2300"
    sheet at 2150 and 2300. Only ssp126/ssp585 and the 17th/50th/83rd
    percentile levels have benchmark data available beyond 2100; other
    scenarios/levels will simply have no benchmark markers plotted.

    Args:
        start_year: First year to plot the extrapolated trajectory from.
        axes: Optional matplotlib axes (one per SSP scenario) to draw on; if None, a new figure is created.

    Returns:
        matplotlib.figure.Figure
    """
    raw = load_data()
    extrapolated = extrapolate_quantiles(raw)

    years = sorted(t for t in extrapolated if start_year <= t <= end_year)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5), squeeze=False)  # fmt: skip
        axes = axes[0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    levels = [17, 50, 83]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 3))  # type: ignore[attr-defined]
    for ax, ssp in zip(axes, _SSP_ORDER):
        for level, color in zip(levels, colors):
            p_target = level / 100

            q_series = np.array(
                [
                    extrapolated[year][ssp][0][extrapolated[year][ssp][1] == p_target][0]
                    for year in years
                ]
            ) * 100  # Convert to cm # fmt: skip

            ax.plot(
                years,
                q_series,
                color=color,
                linewidth=2,
                label=f"{level}th pct (quadratic extrapolation)",
            )

            # Overlay the actual (raw, non-extrapolated) sparse benchmark points, where available.
            bench_years, bench_values = [], []
            for year in (2150, 2300):
                if ssp == "ssp245":
                    continue  # No ssp245 benchmark data beyond 2100 in the raw dataset.

                if year == 2300 and p_target == 0.5:
                    continue  # No 50th percentile benchmark for ssp126/ssp585 at 2300 in the raw dataset.

                if year > end_year:
                    continue  # Skip benchmark years beyond the requested end_year.

                q_bench, p_bench = raw[year][ssp]
                bench_years.append(year)
                bench_values.append(q_bench[p_bench == p_target][0] * 100)  # Convert to cm # fmt: skip

            if bench_years:
                ax.scatter(
                    bench_years,
                    bench_values,
                    color=color,
                    marker="x",
                    s=90,
                    linewidth=2.5,
                    zorder=5,
                    label=f"{level}th pct (data)",
                )

        ax.axvline(2100, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Sea level rise [cm]")
        ax.set_title(f"Extrapolation validation - {ssp.upper()}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    plt.suptitle("Quadratic Extrapolation of Quantiles Beyond 2100 vs. Actual Data")  # fmt: skip
    plt.tight_layout()
    plt.show()

    return fig


def plot_fan_charts(
    start_year: int,
    end_year: int,
    step: int
    | None = None,  # Only used for tick formatting; the actual data is always annual.
    extrapolate: bool = False,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the KNMI quantiles of every scenario as a fan chart from start_year to end_year
    (`extrapolate`: use `extrapolate_quantiles`; `step`: spacing of the year ticks)."""
    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if start_year not in data or end_year not in data:
        raise ValueError(f"Data for years {start_year} or {end_year} not found.")

    years = np.array([t for t in data if start_year <= t <= end_year])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        levels = [0.01, 0.05, 0.17, 0.50, 0.83, 0.95, 0.99]  # Percentiles to plot

        # Extract quantiles for each percentile across all years
        def extract_quantile(year, ssp, level):
            q, p = data.get(year, {}).get(ssp, (np.array([]), np.array([])))
            mask = p == level
            if mask.any():
                return float(q[mask][0])
            else:
                return np.nan

        quantile_data = {
            level: [extract_quantile(year, ssp, level) * 100.0 for year in years]
            for level in levels
        }

        # Unpack into named variables for plotting
        _q01, q05, q17, q50, q83, q95, _q99 = [
            np.array(quantile_data[level]) for level in levels
        ]

        ax.plot(
            years,
            q50,
            color=color,
            linewidth=3,
            label=f"{label} Median",
        )

        ax.fill_between(
            years,
            q05,
            q95,
            color=color,
            alpha=0.3,
            label=f"{label} 5% - 95% percentile",
        )

        if not extrapolate and end_year > 2100:
            idx = np.where(years > 2100)[0][0]
            ax.fill_between(
                years[idx:],
                q17[idx:],
                q83[idx:],
                color=color,
                alpha=0.45,
                label=f"{label} 17% - 83% percentile",
            )

    # Improved tick formatting
    if step is not None:
        ax.set_xticks(range(start_year, end_year + 1, step))

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")
    ax.set_title("Scenario Fan Charts")

    _, labels = ax.get_legend_handles_labels()
    active_labels = [label for label in labels if label and not label.startswith("_")]
    if active_labels:
        ax.legend(loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_continuous_quantile_fns(t: int, extrapolate: bool = False, axes: plt.axes.Axes = None) -> plt.figure.Figure:  # fmt: skip
    """Plots the CDF and PDF of the continuous quantile functions of every interpolation method and
    scenario in year t."""
    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    methods = ["nearest", "linear", "quadratic", "parametric"]
    linestyles = {
        "nearest": (0, (1, 1)),  # densely dotted
        "linear": "--",
        "quadratic": "-.",
        "parametric": "-",  # densely dashed
    }

    p_plot = np.linspace(1e-2, 1 - 1e-2, 500)

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        if not extrapolate and t > 2100 and ssp == "ssp245":
            continue  # No ssp245 data beyond 2100 in the raw dataset.

        q, p = data[t][ssp]
        q = q * 100  # Convert to cm

        # Empirical CDF and PDF
        axes[0].plot(
            q, p, marker="o", markersize=5, linestyle="none", linewidth=1.5, color=color
        )
        pdf_emp = np.diff(p) / np.diff(q)
        q_mid_emp = (q[:-1] + q[1:]) / 2
        axes[1].plot(
            q_mid_emp,
            pdf_emp,
            marker="o",
            markersize=5,
            linestyle="none",
            linewidth=1.5,
            color=color,
        )

        realistic_peak = 0  # keep track of realistic peak for the PDF plot to avoid extreme spikes in the tails
        for method in methods:
            if not extrapolate and t == 2300 and method == "quadratic":
                continue  # Quadratic extrapolation is not defined at t=2300 (only 2 points available for fitting).

            fn, _ = get_continuous_quantile_fn_from_data(p, q, method=method)
            q_plot = np.array([fn(p) for p in p_plot])

            # CDF: invert quantile function → (q, p)
            axes[0].plot(
                q_plot, p_plot, linestyle=linestyles[method], linewidth=1.5, color=color
            )

            # PDF: dp/dq — nearest omitted (step function has no continuous PDF)
            if method != "nearest":
                dq = np.diff(q_plot)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pdf_method = np.where(dq > 0, np.diff(p_plot) / dq, np.nan)
                    if method == "parametric":
                        realistic_peak = np.nanmax(
                            pdf_method
                        )  # update realistic peak for parametric method
                axes[1].plot(
                    (q_plot[:-1] + q_plot[1:]) / 2,
                    pdf_method,
                    linestyle=linestyles[method],
                    linewidth=1.5,
                    color=color,
                )

    # Two-part legend: SSP colour + method linestyle
    ssp_handles = [
        Line2D([0], [0], color="steelblue", lw=2, label="SSP1-2.6"),
        Line2D([0], [0], color="darkorange", lw=2, label="SSP2-4.5"),
        Line2D([0], [0], color="firebrick", lw=2, label="SSP5-8.5"),
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="k",
            lw=1.5,
            linestyle="none",
            marker="o",
            markersize=5,
            label="empirical",
        ),
        Line2D([0], [0], color="k", lw=1.5, linestyle=(0, (1, 1)), label="nearest"),
        Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="linear"),
        Line2D([0], [0], color="k", lw=1.5, linestyle="-.", label="quadratic"),
        Line2D([0], [0], color="k", lw=1.5, linestyle="-", label="parametric"),
    ]
    for ax in axes:
        ax.add_artist(ax.legend(handles=ssp_handles, fontsize=9, loc="upper left"))
        ax.legend(handles=method_handles, fontsize=9, loc="lower right")

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF (approximate)")
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    # Limit pdf plot to a reasonable range to avoid extreme spikes in the tails
    axes[1].set_ylim(bottom=0, top=realistic_peak * 1.2)  # add slight headroom

    plt.suptitle(f"Continuous Quantile Function Methods at t={t} per SSP scenario", fontsize=12)  # fmt:skip
    plt.tight_layout()
    plt.show()

    return fig


def plot_data_against_distribution(
    t: int,
    distribution: str,
    extrapolate: bool = False,
    axes: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the KNMI quantiles of year t against the fitted `distribution`, per scenario."""
    if distribution not in (
        "normal",
        "chi2",
        "exponential",
        "lognormal",
        "pareto",
        "johnsonsu",
        "pearsonIV",
    ):
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'normal', 'chi2', 'exponential', "
            "'lognormal', 'pareto', 'johnsonsu', 'pearsonIV'."
        )

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        if not extrapolate and t > 2100 and ssp == "ssp245":
            continue  # No ssp245 data beyond 2100 in the raw dataset.

        q, p = data[t][ssp]
        q = q * 100  # Convert to cm

        all_fits = _fit_distributions(p, q, distribution=distribution)

        if len(all_fits) == 1:
            fit = all_fits[0]
        else:
            raise ValueError(
                f"Expected a single fit for distribution '{distribution}', but got {len(all_fits)} fits."
            )

        # print(normal_fit["params"])

        x = np.linspace(q.min(), q.max(), 100)

        # CDF
        axes[0].plot(
            q,
            p,
            marker="o",
            markersize=5,
            linestyle="--",
            color=color,
            label=f"{label} (empirical)",
        )
        axes[0].plot(
            x,
            fit["cdf_fn"](x),
            linestyle="-",
            linewidth=1.5,
            color=color,
            label=f"{label} (fitted {distribution})",
        )

        # PDF
        pdf_approx = np.diff(p) / np.diff(q)
        q_mid = (q[:-1] + q[1:]) / 2
        axes[1].plot(
            q_mid,
            pdf_approx,
            marker="o",
            markersize=5,
            linestyle="--",
            color=color,
            label=f"{label} (empirical)",
        )
        axes[1].plot(
            x,
            fit["pdf_fn"](x),
            linestyle="-",
            linewidth=1.5,
            color=color,
            label=f"{label} (fitted {distribution})",
        )

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF (approximate)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    plt.suptitle(
        f"Data vs {distribution.capitalize()} Distribution at t={t} per SSP scenario"
    )
    plt.tight_layout()
    plt.show()

    return fig


def plot_data_against_all_distributions(
    t: int, extrapolate: bool = False, axes: plt.axes.Axes = None
) -> plt.figure.Figure:
    """Plots the KNMI quantiles of year t against every fitted distribution, per scenario."""
    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
    else:
        fig = axes.figure

    ssp_labels = {
        "ssp126": "SSP1-2.6",
        "ssp245": "SSP2-4.5",
        "ssp585": "SSP5-8.5",
    }

    for col, (ssp, ssp_label) in enumerate(ssp_labels.items()):
        if not extrapolate and t > 2100 and ssp == "ssp245":
            continue  # No ssp245 data beyond 2100 in the raw dataset.

        q, p = data[t][ssp]
        q = q * 100  # Convert to cm

        all_fits = _fit_distributions(p, q, distribution="all")

        x = np.linspace(q.min(), q.max(), 100)

        for fit in all_fits:  # pyright: ignore[reportOptionalSubscript]
            axes[0, col].plot(
                x,
                fit["cdf_fn"](x),
                linestyle="-",
                linewidth=1.5,
                label=f"{fit['name']} fit",
            )
            axes[1, col].plot(
                x,
                fit["pdf_fn"](x),
                linestyle="-",
                linewidth=1.5,
                label=f"{fit['name']} fit",
            )

        # Empirical CDF
        axes[0, col].plot(
            q,
            p,
            marker="o",
            markersize=5,
            linestyle="none",
            color="black",
            label="Empirical",
        )

        # Empirical PDF via finite differences
        pdf_approx = np.diff(p) / np.diff(q)
        q_mid = (q[:-1] + q[1:]) / 2
        axes[1, col].plot(
            q_mid,
            pdf_approx,
            marker="o",
            markersize=5,
            linestyle="none",
            color="black",
            label="Empirical",
        )

        for row in range(2):
            axes[row, col].set_xlabel("Sea level rise [cm]")
            axes[row, col].set_title(ssp_label)
            axes[row, col].legend(fontsize=9)
            axes[row, col].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[0, 0].set_ylabel("CDF [-]")
    axes[1, 0].set_ylabel("PDF [1/cm]")

    plt.suptitle(f"Data vs. Fitted Parametric Distributions at t={t} per SSP scenario", fontsize=16)  # fmt:skip
    plt.tight_layout()
    plt.show()

    return fig


def plot_data_against_best_fitting_distribution(
    t: int, extrapolate: bool = False, axes: plt.axes.Axes = None
) -> plt.figure.Figure:
    """Plots the KNMI quantiles of year t against the best fitting distribution, per scenario."""
    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        if not extrapolate and t > 2100 and ssp == "ssp245":
            continue  # No ssp245 data beyond 2100 in the raw dataset.

        q, p = data[t][ssp]
        q = q * 100  # Convert to cm
        x = np.linspace(q.min(), q.max(), 100)

        _, info = get_continuous_quantile_fn_from_data(p, q, method="parametric")
        best = info["selected"]  # pyright: ignore[reportOptionalSubscript]

        # CDF
        axes[0].plot(
            x,
            best["cdf_fn"](x),
            linestyle="-",
            linewidth=1.5,
            color=color,
            label=f"{label} (fitted {best['name']})",
        )
        axes[0].plot(
            q,
            p,
            marker="o",
            markersize=5,
            linestyle="--",
            color=color,
            label=f"{label} (empirical)",
        )

        # PDF
        pdf_approx = np.diff(p) / np.diff(q)
        q_mid = (q[:-1] + q[1:]) / 2
        axes[1].plot(
            x,
            best["pdf_fn"](x),
            linestyle="-",
            linewidth=1.5,
            color=color,
            label=f"{label} (fitted {best['name']})",
        )
        axes[1].plot(
            q_mid,
            pdf_approx,
            marker="o",
            markersize=5,
            linestyle="--",
            color=color,
            label=f"{label} (empirical)",
        )

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF (approximate)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    plt.suptitle(f"Data vs. Best-Fitting Parametric Distribution at t={t} per SSP scenario", fontsize=12)  # fmt:skip
    plt.tight_layout()
    plt.show()

    return fig


def plot_uncertainty_band_aggregation_rule(
    t: int, extrapolate: bool = False, axes: plt.axes.Axes = None
) -> plt.figure.Figure:
    """Plot the data, the uncertainty band and the fitted normal distribution for time step t"""

    if not extrapolate and t > 2100:
        raise ValueError(
            f"Cannot use uncertainty band aggregation rule for t={t} without extrapolation. "
            "Use extrapolate=True to extend the data beyond 2100."
        )

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    q_ssp126, p = data[t]["ssp126"]
    q_ssp585, _ = data[t]["ssp585"]
    uncertainty_band_lower = float(q_ssp126[p == 0.05][0])
    uncertainty_band_upper = float(q_ssp585[p == 0.95][0])

    _, infos = get_aggregated_quantile_fn(t, t, method="uncertainty_band", extrapolate=extrapolate)  # fmt:skip
    mu = infos[0]["mu"]
    sigma = infos[0]["sigma"]

    for ssp, color, label in [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]:
        q, p = data[t][ssp]
        q = q * 100  # Convert to cm

        # Empirical CDF
        axes[0].plot(q, p, "--", marker="o", color=color, label=f"{label} (empirical)")

        # Empirical PDF (finite differences)
        pdf_empirical = np.diff(p) / np.diff(q)
        q_mid = 0.5 * (q[:-1] + q[1:])
        axes[1].plot(q_mid, pdf_empirical, "--", marker="o", color=color, label=f"{label} (empirical)")  # fmt:skip

    # Normal fit
    x = np.linspace(uncertainty_band_lower*100-5, uncertainty_band_upper*100+5, 100)  # fmt:skip
    axes[0].plot(x, norm.cdf(x, loc=mu*100, scale=sigma*100), "-", color="black", linewidth=1.8, label="Normal fit to uncertainty band")  # fmt:skip
    axes[1].plot(x, norm.pdf(x, loc=mu*100, scale=sigma*100), "-", color="black", linewidth=1.8, label="Normal fit to uncertainty band")  # fmt:skip

    # Uncertainty band
    for ax in axes:
        ax.axvspan(uncertainty_band_lower*100, uncertainty_band_upper*100, color="gray", alpha=0.18, label="Uncertainty band")  # fmt:skip

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF")
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    axes[1].legend(fontsize=8)

    plt.suptitle(f"Uncertainty Band Aggregation Rule at t={t}")
    plt.tight_layout()
    plt.show()

    return fig


def plot_aggregated_quantile_fn(
    t: int,
    method: str = "uncertainty_band",
    distribution: str | None = None,
    weights: list[float] | None = None,
    extrapolate: bool = False,
    title: str | None = None,
    axes: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plot the aggregated distribution from ``get_aggregated_quantile_fn`` together with its generating (source) distributions.

    For ``method="uncertainty_band"``, this simply delegates to
    ``plot_uncertainty_band_aggregation_rule``, which plots the empirical data per SSP, the
    uncertainty band, and the normal distribution fitted to it. For ``method="mixture"``, this
    plots the per-SSP empirical data and fitted distributions alongside the resulting mixture
    (aggregated) CDF/PDF, obtained as the weighted sum of the per-SSP fits.

    Args:
        t: Time step (year) to plot.
        method: Aggregation method, see ``get_aggregated_quantile_fn``.
        distribution: Distribution to fit (default: "normal" for "uncertainty_band", "pearsonIV"
            for "mixture"). Only "normal" is supported for "uncertainty_band", since
            ``plot_uncertainty_band_aggregation_rule`` does not support other distributions.
        weights: Only used for ``method="mixture"``. Sequence of 3 non-negative weights for
            (ssp126, ssp245, ssp585), must sum to 1. Defaults to uniform weights (1/3, 1/3, 1/3).
        extrapolate: If True, extend the raw data (1995-2100, plus sparse 2150/2300
            benchmarks) to a continuous annual series through 2300 via
            ``extrapolate_quantiles`` before building the quantile function. If False
            (default), only the raw data returned by ``load_data()`` is used.
        title: Optional title for the figure. If None, a default title is generated based on
            the time step and aggregation method.
        axes: Optional pair of axes (CDF, PDF) to plot into. If None, a new figure is created.

    Returns:
        The matplotlib figure containing the CDF and PDF plots.
    """

    if method not in ("uncertainty_band", "mixture"):
        raise ValueError(f"Invalid method '{method}'. Choose from 'uncertainty_band', 'mixture'.")  # fmt:skip

    if method == "uncertainty_band":
        if distribution not in (None, "normal"):
            raise ValueError(
                "plot_aggregated_quantile_fn only supports distribution='normal' (or None) "
                "for method='uncertainty_band', since plot_uncertainty_band_aggregation_rule "
                "does not support fitting other distributions to the uncertainty band."
            )
        return plot_uncertainty_band_aggregation_rule(
            t, extrapolate=extrapolate, axes=axes
        )

    # method == "mixture"
    weights_list = [1 / 3, 1 / 3, 1 / 3]
    if weights is not None:
        weights_list = weights
    if len(weights_list) != 3:
        raise ValueError(f"'weights' must have 3 elements (one per SSP scenario: {_SSP_ORDER}), got {len(weights_list)}.")  # fmt:skip
    if any(w < 0 for w in weights_list):
        raise ValueError(f"'weights' must be non-negative, got {weights_list}.")
    if not np.isclose(sum(weights_list), 1.0):
        raise ValueError(f"'weights' must sum to 1, got {weights_list} (sum={sum(weights_list)}).")  # fmt:skip

    distribution_name = distribution if distribution is not None else "pearsonIV"

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if t not in data:
        raise ValueError(f"Time step {t} not found in data.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = axes.figure

    include_ssp245 = extrapolate or t <= 2100
    ssp_info = [
        ("ssp126", "steelblue", "SSP1-2.6"),
        ("ssp245", "darkorange", "SSP2-4.5"),
        ("ssp585", "firebrick", "SSP5-8.5"),
    ]
    if not include_ssp245:
        ssp_info = [s for s in ssp_info if s[0] != "ssp245"]
        normalized_weights = [
            weights_list[0] / (weights_list[0] + weights_list[2]),
            weights_list[2] / (weights_list[0] + weights_list[2]),
        ]
    else:
        normalized_weights = weights_list

    fits = {}
    q_all = []
    for ssp, color, label in ssp_info:
        q, p = data[t][ssp]
        q = q * 100  # Convert to cm
        q_all.append(q)

        all_fits = _fit_distributions(p, q, distribution=distribution_name)
        if len(all_fits) != 1:
            raise RuntimeError(
                f"Expected a single fit for distribution '{distribution_name}' for '{ssp}' at t={t}, got {len(all_fits)}."
            )
        fits[ssp] = all_fits[0]

        # Empirical CDF/PDF
        axes[0].plot(q, p, marker="o", markersize=5, linestyle="--", color=color, label=f"{label} (empirical)")  # fmt:skip
        pdf_empirical = np.diff(p) / np.diff(q)
        q_mid = (q[:-1] + q[1:]) / 2
        axes[1].plot(q_mid, pdf_empirical, marker="o", markersize=5, linestyle="--", color=color, label=f"{label} (empirical)")  # fmt:skip

    x = np.linspace(min(q.min() for q in q_all), max(q.max() for q in q_all), 300)

    # Per-SSP fitted (source) distributions
    for ssp, color, label in ssp_info:
        fit = fits[ssp]
        axes[0].plot(x, fit["cdf_fn"](x), linestyle="-", linewidth=1.5, color=color, label=f"{label} (fitted {distribution_name})")  # fmt:skip
        axes[1].plot(x, fit["pdf_fn"](x), linestyle="-", linewidth=1.5, color=color, label=f"{label} (fitted {distribution_name})")  # fmt:skip

    # Aggregated mixture distribution: weighted sum of the per-SSP fitted CDFs/PDFs
    mixture_cdf = sum(w * fits[ssp]["cdf_fn"](x) for w, (ssp, _, _) in zip(normalized_weights, ssp_info))  # fmt:skip
    mixture_pdf = sum(w * fits[ssp]["pdf_fn"](x) for w, (ssp, _, _) in zip(normalized_weights, ssp_info))  # fmt:skip
    axes[0].plot(
        x, mixture_cdf, "-", color="black", linewidth=2.2, label="Mixture (aggregated)"
    )
    axes[1].plot(
        x, mixture_pdf, "-", color="black", linewidth=2.2, label="Mixture (aggregated)"
    )

    axes[0].set_xlabel("Sea level rise [cm]")
    axes[0].set_ylabel("CDF [-]")
    axes[0].set_title("CDF")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Sea level rise [cm]")
    axes[1].set_ylabel("PDF [1/cm]")
    axes[1].set_title("PDF")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    plt.suptitle(title if title is not None else f"Aggregated Distribution at t={t}")
    plt.tight_layout()
    plt.show()

    return fig
