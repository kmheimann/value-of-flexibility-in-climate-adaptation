"""
decision_analysis.py — Optimal dike design under uncertain sea level rise (thesis Chapter 6).

Combines three submodels in a sequential decision problem, solved by backward induction:

    sea_level_rise_model  A `SeaLevelRiseModel` (source.sea_level_rise_model); its `.lattice` gives
                          the possible sea level rise states and their transition probabilities.
    risk_model            A `RiskModel` (source.risk_model): expected annual flood loss of a dike.
    cost_model            A `CostModel` (source.cost_model): cost of every upgrade, including the
                          choice of flexibility at the first decision.
    backward_kwargs       A dict with the parameters of the optimization itself:
                            discount_rate   discount rate rho [-]
                            h_minus_1       capacity of the existing dike [m]
                            max_possible_h  largest upgrade above h_minus_1 that is considered [m]
                            h_step          spacing of the height grid [m]

Rewards are negative expected discounted costs (the thesis' W_i); results are shown as expected cost.

Solving:
    backward_induction          Optimal decisions in every state, returned as a `RewardsTable`.
    extract_optimal_designs     Optimal inflexible and flexible dike and the value of flexibility.
    extract_optimal_design_at_flexibility
                                Optimal first decision at a fixed flexibility value.
    decision_path, compute_realized_reward
                                Decisions and realized cost along one sampled trajectory.
    cost_by_capacity, cost_by_flexibility, around_optimum
                                Expected cost of the first-decision designs.
    design_summary              Table of the optimal designs of several cases.

Sensitivity analysis:
    sweep_designs               Solves the problem for every value of one or several parameters.
    sweep_results, sweep_summary, sweep_table
                                Expected costs and (relative) value of flexibility of sweeps.
    compare_learning_models     Solves the problem for other sea level rise (learning) models.
    cost_ratio_pairs, cost_level_pairs, cost_level
                                Sweep values of the fixed and variable cost (c, v).

Plots:
    plot_optimal_decisions_along_trajectory, plot_optimal_decisions_along_trajectories,
    plot_optimal_decisions_by_model, plot_cost_by_capacity, plot_cost_by_flexibility, plot_sweep, match_ylim_spans.

Tables are printed with thesis precision: costs in M€ with one decimal, relative values in % with
one decimal, heights in cm.

This module only uses the abstract submodel classes, so a new cost, risk or sea level rise model can
be used without changing it.
"""

import dataclasses
import math
import numbers
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from tabulate import tabulate

from source.cost_model import (
    CostModel,
    FlexibilitySpec,
    LandCostModel,
    available_crest_height,
)
from source.risk_model import RiskModel
from source.sea_level_rise_model import SeaLevelRiseModel, plot_trajectory

EUR_PER_MEUR = 1e6  # Display unit of costs [€ per M€]
INFLEXIBLE_DIKE_COLOR = "#B46E28"
FLEXIBLE_DIKE_COLOR = "#DAB794"
WAITING_REGION_COLOR = "0.93"  # Shading of sweep values at which waiting (deferral) is optimal # fmt: skip
FLEXIBILITY_CMAP = "Blues"  # Colors of designs at fixed flexibility values (dark = least flexible) # fmt: skip
FLEXIBILITY_CMAP_RANGE = (1.0, 0.45)  # Positions on the color map of the least/most flexible value # fmt: skip

_BACKWARD_KWARGS_KEYS = {"discount_rate", "h_minus_1", "max_possible_h", "h_step"}


# ─── Backward induction ─────────────────────────────────────────────────────────


def discount_factor(t_i: int, discount_rate: float) -> float:
    """Discount factor 1 / (1 + discount_rate)^t_i for t_i years after t_0."""
    return 1 / (1 + discount_rate) ** t_i


@dataclasses.dataclass(frozen=True)
class RewardsTable:
    """Result of `backward_induction`.

    Attributes:
        t0: Every first-decision design with its expected reward [€], indexed by
            ("h_0", "<flexibility>_0"), e.g. ("h_0", "phi_0"). Column "reward".
        rest: The optimal decision in every later state, indexed by ("t_i", "theta_i",
            "h_i_minus_1", "<flexibility>_i_minus_1"), with columns "h_i", "<flexibility>_i" and
            "reward". t_i are years after the lattice start year, theta_i are sea level rise states [m].
        h_minus_1: Capacity of the existing dike [m].
        spec: The cost model's `FlexibilitySpec`, which names the flexibility columns.
    """

    t0: pd.DataFrame
    rest: pd.DataFrame
    h_minus_1: float
    spec: FlexibilitySpec


def _lookup_optimal(
    rest: pd.DataFrame,
    spec: FlexibilitySpec,
    t_i: int,
    theta_i: float,
    h_i_minus_1: float,
    f_i_minus_1: tuple[float, ...],
) -> tuple[float, tuple[float, ...], float]:
    """Optimal (h_i, f_i, reward) in the state (t_i, theta_i, h_i_minus_1, f_i_minus_1) of `rest`."""
    best = rest.loc[(t_i, theta_i, h_i_minus_1, *f_i_minus_1)]  # type: ignore
    return best["h_i"], tuple(best[f"{c}_i"] for c in spec.components), best["reward"]


def _cost_time_kwargs(
    cost_model: CostModel, t_i: int, t_i_plus_1: int, t_0: int
) -> dict:
    """Years since the first decision t_0, passed to `adapt_cost_fn` of a time-aware cost model."""
    if not cost_model.time_aware:
        return {}
    return {"t_i": t_i - t_0, "t_i_plus_1": t_i_plus_1 - t_0}


def _candidate_reward(
    initial: bool,
    t_i: int,
    t_i_plus_1: int,
    t_0: int,
    lattice,
    theta_i: float,
    h_i_minus_1: float,
    f_i_minus_1: tuple[float, ...],
    h_i: float,
    f_i: tuple[float, ...],
    cost_model: CostModel,
    risk_model: RiskModel,
    spec: FlexibilitySpec,
    h_minus_1: float,
    discount_rate: float,
    continuation,
) -> float:
    """Expected reward [€] of the decision (h_i, f_i) in the state (t_i, theta_i, h_i_minus_1,
    f_i_minus_1): minus the discounted upgrade cost, minus the discounted expected flood loss in every
    year until t_i_plus_1, plus the expected optimal reward at t_i_plus_1 (`continuation(theta, h, f)`).
    """
    f_standing = spec.priced_state(h_i_minus_1, f_i_minus_1, h_minus_1)
    if initial:
        cost = cost_model.initial_cost_fn(h_i_minus_1, *f_standing, h_i, *f_i)  # type: ignore
    else:
        time_kwargs = _cost_time_kwargs(cost_model, t_i, t_i_plus_1, t_0)
        cost = cost_model.adapt_cost_fn(h_i_minus_1, *f_standing, h_i, *f_i, **time_kwargs)  # type: ignore # fmt: skip

    # Expected annual flood loss, given the sea level rise state at the next time step
    transitions = lattice.get_transition_probs(t_i, theta_i, None)
    expected_loss = 0
    for prob, theta_i_plus_1 in transitions:
        expected_loss += prob * risk_model.risk_fn(h_i, theta_i_plus_1)

    immediate_reward = -discount_factor(t_i, discount_rate) * cost + sum(-discount_factor(t, discount_rate) * expected_loss for t in range(t_i + 1, t_i_plus_1 + 1))  # fmt: skip

    future_rewards = 0
    for prob, theta_i_plus_1 in transitions:
        future_rewards += prob * continuation(theta_i_plus_1, h_i, f_i)

    return immediate_reward + future_rewards


def _freeze(value):
    """Hashable representation of a submodel or `backward_kwargs`, used as a cache key.

    A dataclass is represented by its class (including its identity, so that a class redefined in a
    notebook gets a new key) and its fields.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        class_id = (cls.__module__, cls.__qualname__, id(cls))
        return (class_id,) + tuple((f.name, _freeze(getattr(value, f.name))) for f in dataclasses.fields(value) if f.compare)  # fmt: skip
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


_BACKWARD_INDUCTION_CACHE_MAXSIZE = 100  # Number of solutions kept in memory
_backward_induction_cache: OrderedDict[tuple, tuple] = OrderedDict()


def clear_backward_induction_cache() -> None:
    """Frees the memory of the `backward_induction` cache."""
    _backward_induction_cache.clear()


def _height_grid(h_minus_1: float, max_possible_h: float, h_step: float) -> list[float]:
    """Absolute height grid h_minus_1 + {0, h_step, ..., max_possible_h} [m].

    Computed in whole centimeters to avoid floating point errors. If max_possible_h is not a multiple
    of h_step, the grid ends at the multiple below it (with a warning).
    """
    h_minus_1_cm = round(h_minus_1 * 100)
    max_possible_h_cm = round(max_possible_h * 100)
    h_step_cm = round(h_step * 100)
    num_h = max_possible_h_cm // h_step_cm + 1
    if (num_h - 1) * h_step_cm != max_possible_h_cm:
        warnings.warn(f"max_possible_h={max_possible_h} is not a multiple of h_step={h_step}; the height grid ends at {(num_h - 1) * h_step_cm / 100} m above h_minus_1.", stacklevel=3)  # fmt: skip
    return ((h_minus_1_cm + np.arange(num_h) * h_step_cm) / 100).tolist()


def backward_induction(
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
) -> tuple[tuple[float, ...], RewardsTable]:
    """Solves the sequential decision problem by backward induction (thesis Sec. 6.4).

    At every decision time t_i (the lattice time steps except the last) and in every state
    (theta_i, h_i_minus_1, f_i_minus_1), every feasible decision (h_i, f_i) is evaluated: dikes can
    only be raised, and the flexibility f is chosen at the first decision only (see the cost model's
    `flexibility_spec`). The reward of a decision is minus its discounted upgrade cost and expected
    flood loss, plus the expected optimal reward of the next decision.

    Args:
        sea_level_rise_model: Sea level rise model; its lattice gives the states and transitions.
        risk_model: Risk model, `risk_fn(h_i, theta_i_plus_1)` [€/yr].
        cost_model: Cost model, `initial_cost_fn`/`adapt_cost_fn` [€] and `flexibility_spec`.
        backward_kwargs: Dict with `discount_rate` [-], `h_minus_1` [m], `max_possible_h` [m] (largest
            upgrade above h_minus_1) and `h_step` [m].

    Returns:
        (optimal_initial_design, rewards): the optimal first decision (h_0, f_0, reward) and a
        `RewardsTable` with every first-decision design and the optimal later decisions.

    Remarks:
        - Results are cached in memory, keyed on the values of all arguments. The returned objects are
          shared between calls, so they must not be modified. `clear_backward_induction_cache()`
          frees the cache.
        - The design classes (inflexible/flexible) are determined afterwards by
          `extract_optimal_designs`.
    """
    unknown = set(backward_kwargs) - _BACKWARD_KWARGS_KEYS
    if unknown:
        raise ValueError(f"Unknown backward_kwargs {sorted(unknown)}; expected {sorted(_BACKWARD_KWARGS_KEYS)}.")  # fmt: skip

    key = (_freeze(sea_level_rise_model), _freeze(risk_model), _freeze(cost_model), _freeze(backward_kwargs))  # fmt: skip
    if key in _backward_induction_cache:
        _backward_induction_cache.move_to_end(key)
        return _backward_induction_cache[key]

    spec = cost_model.flexibility_spec
    discount_rate = backward_kwargs["discount_rate"]
    h_minus_1 = backward_kwargs["h_minus_1"]
    lattice = sea_level_rise_model.lattice
    time_steps, _ = lattice.time_steps()
    h_alternatives = _height_grid(h_minus_1, backward_kwargs["max_possible_h"], backward_kwargs["h_step"])  # fmt: skip
    t_0 = time_steps[0]

    flex_cols = list(spec.components)
    records_t0 = []  # (h_0, *f_0, reward) of every first-decision design
    records_rest = []  # (t_i, theta_i, h_i_minus_1, *f_i_minus_1, h_i, *f_i, reward) of every later optimum
    optimal_decisions = {}  # t_i -> {(theta_i, h_i_minus_1, f_i_minus_1): (h_i, f_i, reward)}

    for idx in reversed(
        range(len(time_steps) - 1)
    ):  # The last time step has no decision
        initial = idx == 0
        t_i, t_i_plus_1 = time_steps[idx], time_steps[idx + 1]
        if initial:
            states = [(h_minus_1, spec.initial_state(h_minus_1))]  # The existing dike
        else:
            states = [(h, f) for h in h_alternatives for f in spec.state_candidates(h, h_alternatives)]  # fmt: skip

        if idx == len(time_steps) - 2:
            continuation = lambda theta, h, f: 0.0
        else:
            next_optimum = optimal_decisions[t_i_plus_1]
            continuation = lambda theta, h, f: next_optimum[(theta, h, f)][2]  # noqa: B023

        optimal_decisions[t_i] = {}
        for theta_i in lattice.states(t_i):
            for h_i_minus_1, f_i_minus_1 in states:
                candidates = []  # (h_i, f_i, reward)
                for h_i in (
                    h for h in h_alternatives if h >= h_i_minus_1
                ):  # Dikes are never lowered
                    for f_i in spec.action_candidates(h_i_minus_1, f_i_minus_1, h_i, h_alternatives, initial):  # fmt: skip
                        reward = _candidate_reward(initial, t_i, t_i_plus_1, t_0, lattice, theta_i, h_i_minus_1, f_i_minus_1, h_i, f_i, cost_model, risk_model, spec, h_minus_1, discount_rate, continuation)  # fmt: skip
                        candidates.append((h_i, f_i, reward))

                h_i_opt, f_i_opt, reward_opt = max(
                    candidates, key=lambda c: c[-1]
                )  # First maximum
                optimal_decisions[t_i][(theta_i, h_i_minus_1, f_i_minus_1)] = (h_i_opt, f_i_opt, reward_opt)  # fmt: skip
                if initial:
                    records_t0 += [
                        (h_i, *f_i, reward) for h_i, f_i, reward in candidates
                    ]
                else:
                    records_rest.append((t_i, theta_i, h_i_minus_1, *f_i_minus_1, h_i_opt, *f_i_opt, reward_opt))  # fmt: skip

    h_0_opt, f_0_opt, reward_opt = max(
        optimal_decisions[t_0].values(), key=lambda c: c[2]
    )
    optimal_initial_design = (h_0_opt, *f_0_opt, reward_opt)

    t0_cols = ["h_0", *[f"{c}_0" for c in flex_cols]]
    t0 = pd.DataFrame(records_t0, columns=[*t0_cols, "reward"]).set_index(t0_cols).sort_index()  # fmt: skip
    rest_index = [
        "t_i",
        "theta_i",
        "h_i_minus_1",
        *[f"{c}_i_minus_1" for c in flex_cols],
    ]
    rest = pd.DataFrame(records_rest, columns=[*rest_index, "h_i", *[f"{c}_i" for c in flex_cols], "reward"])  # fmt: skip
    rest = rest.set_index(rest_index).sort_index()

    result = (optimal_initial_design, RewardsTable(t0=t0, rest=rest, h_minus_1=h_minus_1, spec=spec))  # fmt: skip
    _backward_induction_cache[key] = result
    if len(_backward_induction_cache) > _BACKWARD_INDUCTION_CACHE_MAXSIZE:
        _backward_induction_cache.popitem(
            last=False
        )  # Remove the least recently used solution
    return result


# ─── Designs ────────────────────────────────────────────────────────────────────


def extract_optimal_designs(
    rewards: RewardsTable,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Optimal inflexible and flexible dike and the value of flexibility (thesis Sec. 3.2).

    The inflexible dike is the best design without flexibility (phi_0 = 0, or h_max_0 = h_0); the
    flexible dike is the best design overall (flexibility allowed, not required). Both may wait at the
    first decision (h_0 = h_minus_1, see `is_waiting`).

    Args:
        rewards: Result of `backward_induction`.

    Returns:
        (vof, inflex, flex): the value of flexibility vof = E[cost(inflex)] - E[cost(flex)] >= 0 [€],
        and the two designs as (h_0 [m], f_0, reward [€]) tuples (possibly the same design).
    """
    t0 = rewards.t0.reset_index()
    masks = rewards.spec.design_masks(t0)
    columns = ["h_0", *[f"{c}_0" for c in rewards.spec.components], "reward"]

    def best(mask: pd.Series) -> tuple:
        row = t0[mask].loc[t0[mask]["reward"].idxmax()]  # type: ignore
        return tuple(row[c] for c in columns)

    inflex, flex = best(masks["inflexible"]), best(masks["flexible"])
    return flex[-1] - inflex[-1], inflex, flex


def is_waiting(design: tuple[float, ...], h_minus_1: float) -> bool:
    """True if `design` (h_0, f_0, ...) does not upgrade at the first decision (h_0 = h_minus_1 [m])."""
    return bool(np.isclose(design[0], h_minus_1))


def extract_optimal_design_at_flexibility(
    rewards: RewardsTable,
    flex_value: float | None = None,
    relative: bool = False,
) -> tuple[float, ...]:
    """Optimal first decision (h_0, f_0, reward) at a fixed flexibility value, h_0 optimized
    (waiting (deferral) included).

    Args:
        rewards: Result of `backward_induction`.
        flex_value: The flexibility value: phi_0 [-], or the secured height h_max_0 [m]. If None,
            the overall optimal design is returned.
        relative: If True, a height-like `flex_value` is the reserve h_max_0 - h_0 [m] instead of
            h_max_0 itself.

    Raises:
        ValueError: If no design has this flexibility value.
    """
    t0 = rewards.t0.reset_index()
    spec = rewards.spec
    column = f"{spec.components[0]}_0"
    if flex_value is None:
        matches = np.ones(len(t0), dtype=bool)
    elif relative:
        targets = [
            spec.resolve_flexibility_state(h_0, flex_value)[0] for h_0 in t0["h_0"]
        ]
        matches = np.isclose(t0[column], targets)
    else:
        matches = np.isclose(t0[column], flex_value)
    candidates = t0[matches]
    if candidates.empty:
        raise ValueError(f"No design has {column} = {flex_value!r}.")
    row = candidates.loc[candidates["reward"].idxmax()]  # type: ignore
    return tuple(float(row[c]) for c in ["h_0", *[f"{c}_0" for c in spec.components], "reward"])  # fmt: skip


def around_optimum(rewards: RewardsTable, spacing: float = 0.5) -> tuple[float, ...]:
    """Three values of h_0 [m] around the optimal one: the optimal h_0 rounded to `spacing` [m], and
    one `spacing` below and above it; each on the height grid and above h_minus_1."""
    grid = rewards.t0.index.get_level_values("h_0").unique().to_numpy()
    center = round(rewards.t0["reward"].idxmax()[0] / spacing) * spacing  # type: ignore
    capacities = []
    for h_0 in (center - spacing, center, center + spacing):
        h_0 = float(grid[np.argmin(np.abs(grid - h_0))])
        if h_0 > rewards.h_minus_1 and h_0 not in capacities:
            capacities.append(h_0)
    return tuple(capacities)


def cost_by_capacity(
    rewards: RewardsTable, flex_value: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost of the first-decision designs against h_0.

    Args:
        rewards: Result of `backward_induction`.
        flex_value: Fixed flexibility value (phi_0 [-] or h_max_0 [m]); if None, the optimal
            flexibility at every h_0.

    Returns:
        (h_0 [m], expected cost [€]), waiting (deferral; h_0 = h_minus_1) included where it exists.
    """
    t0 = rewards.t0.reset_index()
    column = f"{rewards.spec.components[0]}_0"
    if flex_value is not None:
        t0 = t0[np.isclose(t0[column], flex_value)]
    best = t0.groupby("h_0")["reward"].max()
    return best.index.to_numpy(), -best.to_numpy()


def cost_by_flexibility(
    rewards: RewardsTable, h_0: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost of the first-decision designs against the flexibility value.

    Args:
        rewards: Result of `backward_induction`.
        h_0: Fixed first-decision height [m]; if None, the optimal h_0 at every flexibility value
            (waiting (deferral) included).

    Returns:
        (flexibility value [phi_0 in -, or h_max_0 in m], expected cost [€]).
    """
    t0 = rewards.t0.reset_index()
    column = f"{rewards.spec.components[0]}_0"
    if h_0 is not None:
        t0 = t0[np.isclose(t0["h_0"], h_0)]
    best = t0.groupby(column)["reward"].max()
    return best.index.to_numpy(), -best.to_numpy()


def design_summary(
    rewards_by_case: dict[str, RewardsTable], table: bool = True
) -> pd.DataFrame:
    """Table of the optimal inflexible and flexible dike of every case (see
    `extract_optimal_designs`), one row per case.

    Columns: expected cost of both dikes [M€], value of flexibility VoF [M€], relative VoF
    (VoF / expected cost of the inflexible dike) [%], first-decision height h_0 of both dikes [cm],
    flexibility of the flexible dike (phi_0, or h_max_0 in cm) and whether each dike waits.

    Args:
        rewards_by_case: Case name -> result of `backward_induction`.
        table: Whether to print the table (thesis precision, see the module docstring).

    Returns:
        The table as a DataFrame with unrounded values.
    """
    rows = {}
    for case, rewards in rewards_by_case.items():
        vof, inflex, flex = extract_optimal_designs(rewards)
        height_like = rewards.spec.height_like[0]
        rows[case] = {
            "E[cost] inflexible [M€]": -inflex[-1] / EUR_PER_MEUR,
            "E[cost] flexible [M€]": -flex[-1] / EUR_PER_MEUR,
            "VoF [M€]": vof / EUR_PER_MEUR,
            "Relative VoF [%]": vof / -inflex[-1] * 100,
            "h_0 inflexible [cm]": inflex[0] * 100,
            "h_0 flexible [cm]": flex[0] * 100,
            f"{rewards.spec.components[0]}_0 flexible"
            + (" [cm]" if height_like else ""): flex[1] * (100 if height_like else 1),
            "Inflexible waits": is_waiting(inflex, rewards.h_minus_1),
            "Flexible waits": is_waiting(flex, rewards.h_minus_1),
        }
    summary = pd.DataFrame(rows).T
    if table:
        print(tabulate(
            [[case, *(_format_column(column, value) for column, value in row.items())] for case, row in summary.iterrows()], # type: ignore
            headers=["Case", *summary.columns],
            tablefmt="simple",
            disable_numparse=True,
        ))  # fmt: skip
    return summary


def _format_column(column: str, value) -> str:
    """Text of a `design_summary` value with thesis precision."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if "[cm]" in column:
        return f"{value:.0f}"
    if "[M€]" in column or "[%]" in column:
        return f"{value:.1f}"
    return f"{value:.2f}"


# ─── Trajectories ───────────────────────────────────────────────────────────────


def decision_path(
    rewards: RewardsTable,
    trajectory: list[float],
    sea_level_rise_model: SeaLevelRiseModel,
    design: tuple[float, ...],
) -> tuple[list[float], list[tuple[float, ...]]]:
    """Optimal decisions along one sea level rise trajectory, starting with the first decision
    `design` = (h_0, f_0, ...).

    Returns:
        (heights [m], flexibility states), one entry per lattice time step; the last entry repeats the
        last decision.
    """
    h_0, f_0 = design[0], tuple(design[1 : 1 + len(rewards.spec.components)])
    time_steps, _ = sea_level_rise_model.lattice.time_steps()
    heights, flex_states = [h_0], [f_0]
    for t, theta in zip(time_steps[1:-1], trajectory[1:-1]):
        h_i, f_i, _ = _lookup_optimal(rewards.rest, rewards.spec, t, theta, heights[-1], flex_states[-1])  # fmt: skip
        heights.append(h_i)
        flex_states.append(f_i)
    return heights + [heights[-1]], flex_states + [flex_states[-1]]


def compute_realized_reward(
    trajectory: list[float],
    heights: list[float],
    flex_states: list[tuple[float, ...]],
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
) -> float:
    """Discounted reward [€] (negative cost) of one decision sequence along one sampled trajectory.

    Same cost, flood loss and discounting as `backward_induction`, but with the realized next state
    instead of an expectation over the lattice.

    Args:
        trajectory: Sea level rise at every lattice time step [m].
        heights: Dike height after every decision [m] (at least one per decision).
        flex_states: Flexibility state after every decision, parallel to `heights`.
        sea_level_rise_model, risk_model, cost_model, backward_kwargs: As for `backward_induction`.
    """
    spec = cost_model.flexibility_spec
    discount_rate = backward_kwargs["discount_rate"]
    h_minus_1 = backward_kwargs["h_minus_1"]
    time_steps, _ = sea_level_rise_model.lattice.time_steps()
    t_0 = time_steps[0]

    total_reward = 0.0
    for idx in range(len(time_steps) - 1):
        t_i, t_i_plus_1 = time_steps[idx], time_steps[idx + 1]
        h_i, f_i = heights[idx], flex_states[idx]
        if idx == 0:
            cost = cost_model.initial_cost_fn(h_minus_1, *spec.initial_state(h_minus_1), h_i, *f_i)  # type: ignore # fmt: skip
        else:
            h_i_minus_1 = heights[idx - 1]
            f_standing = spec.priced_state(h_i_minus_1, flex_states[idx - 1], h_minus_1)
            time_kwargs = _cost_time_kwargs(cost_model, t_i, t_i_plus_1, t_0)
            cost = cost_model.adapt_cost_fn(h_i_minus_1, *f_standing, h_i, *f_i, **time_kwargs)  # type: ignore # fmt: skip
        loss = risk_model.risk_fn(h_i, trajectory[idx + 1])
        total_reward += -discount_factor(t_i, discount_rate) * cost + sum(-discount_factor(t, discount_rate) * loss for t in range(t_i + 1, t_i_plus_1 + 1))  # fmt: skip
    return total_reward


# ─── Sensitivity analysis ───────────────────────────────────────────────────────


def build_learning_model(
    sea_level_rise_model: SeaLevelRiseModel, model_cls: type[SeaLevelRiseModel]
) -> SeaLevelRiseModel:
    """The sea level rise (learning) model `model_cls` with the calibration (start/end year, time
    step, scenario weights, ...) of `sea_level_rise_model`. Returns `sea_level_rise_model` itself if it
    already is a `model_cls`."""
    if type(sea_level_rise_model) is model_cls:
        return sea_level_rise_model
    shared = {f.name: getattr(sea_level_rise_model, f.name) for f in dataclasses.fields(SeaLevelRiseModel) if f.init}  # fmt: skip
    return model_cls(**shared)


def compare_learning_models(
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    learning_models: dict[str, type[SeaLevelRiseModel]],
) -> dict[str, dict]:
    """Solves the decision problem for other sea level rise (learning) models with the same
    calibration.

    Args:
        sea_level_rise_model, risk_model, cost_model, backward_kwargs: As for `backward_induction`.
        learning_models: Name -> SeaLevelRiseModel class, e.g. {"direct_fit":
            DirectFitSeaLevelRiseModel, "static_grid": StaticGridSeaLevelRiseModel}.

    Returns:
        Name -> {"model", "rewards", "vof", "inflex", "flex"} (see `extract_optimal_designs`).
    """
    results = {}
    for name, model_cls in learning_models.items():
        model = build_learning_model(sea_level_rise_model, model_cls)
        _, rewards = backward_induction(model, risk_model, cost_model, backward_kwargs)
        vof, inflex, flex = extract_optimal_designs(rewards)
        results[name] = {"model": model, "rewards": rewards, "vof": vof, "inflex": inflex, "flex": flex}  # fmt: skip
    return results


def _resolve_param_owner(param_name: str, holders: dict[str, object], backward_kwargs: dict) -> tuple:  # fmt: skip
    """Identifies where the parameter `param_name` lives: ("top", holder) for a field of a submodel,
    ("nested", holder, field) for a field of a submodel's dataclass field (e.g. LandCostModel.inner),
    or ("backward_kwargs",). Raises ValueError unless there is exactly one owner."""
    owners = [name for name, holder in holders.items() if param_name in {f.name for f in dataclasses.fields(holder)}]  # type: ignore # fmt: skip
    nested_owners = [
        (name, f.name)
        for name, holder in holders.items()
        for f in dataclasses.fields(holder)  # type: ignore
        if dataclasses.is_dataclass(nested := getattr(holder, f.name))
        and not isinstance(nested, type)
        and param_name in {nf.name for nf in dataclasses.fields(nested)}
    ]
    in_backward_kwargs = param_name in backward_kwargs
    if len(owners) + len(nested_owners) + in_backward_kwargs != 1:
        found = (
            owners + nested_owners + (["backward_kwargs"] if in_backward_kwargs else [])
        )
        raise ValueError(
            f"Expected exactly one owner of parameter {param_name!r}, found {found}."
        )
    if in_backward_kwargs:
        return ("backward_kwargs",)
    if nested_owners:
        return ("nested", *nested_owners[0])
    return ("top", owners[0])


def _replace_swept_params(
    param_names: tuple[str, ...],
    values: tuple,
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
) -> tuple[SeaLevelRiseModel, RiskModel, CostModel, dict]:
    """Copies of the submodels and `backward_kwargs` with the parameters `param_names` set to `values`.

    All changes to the same object are applied at once, so that fields that are checked against each
    other (e.g. phi_minus_1 and phi_values) can be changed together.
    """
    holders = {"sea_level_rise_model": sea_level_rise_model, "risk_model": risk_model, "cost_model": cost_model}  # fmt: skip
    kwargs_changes: dict = {}
    top_changes: dict[str, dict] = {}
    nested_changes: dict[tuple[str, str], dict] = {}
    for name, value in zip(param_names, values, strict=True):
        location = _resolve_param_owner(name, holders, backward_kwargs)
        if location[0] == "backward_kwargs":
            kwargs_changes[name] = value
        elif location[0] == "nested":
            nested_changes.setdefault(location[1:], {})[name] = value
        else:
            top_changes.setdefault(location[1], {})[name] = value

    for holder_name, changes in top_changes.items():
        holders[holder_name] = dataclasses.replace(holders[holder_name], **changes)  # type: ignore
    for (holder_name, field_name), changes in nested_changes.items():
        holder = holders[holder_name]
        holders[holder_name] = dataclasses.replace(holder, **{field_name: dataclasses.replace(getattr(holder, field_name), **changes)})  # type: ignore # fmt: skip
    return (
        holders["sea_level_rise_model"],  # type: ignore
        holders["risk_model"],  # type: ignore
        holders["cost_model"],  # type: ignore
        {**backward_kwargs, **kwargs_changes},
    )


def sweep_designs(
    param_name: str | tuple[str, ...],
    param_values: list,
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    learning_models: dict[str, type[SeaLevelRiseModel]] | None = None,
) -> dict[str, list]:
    """Solves the decision problem for every value of a parameter (one-at-a-time sensitivity analysis).

    Args:
        param_name: Name of a submodel field or `backward_kwargs` key, e.g. "discount_rate", "D",
            "lambda_enc" or "phi_minus_1" (fields of `LandCostModel.inner` are found too). A tuple of
            names changes several parameters together; each value is then a tuple of the same length.
            The name "learning_model" replaces the sea level rise model by the class
            `learning_models[value]` with the same calibration.
        param_values: Values of the parameter(s).
        sea_level_rise_model, risk_model, cost_model, backward_kwargs: The baseline, as for
            `backward_induction`.
        learning_models: Name -> SeaLevelRiseModel class; needed for "learning_model".

    Returns:
        Dict of lists parallel to `param_values`: "inflex", "flex" (expected reward of the optimal
        inflexible/flexible dike [€]), "inflex_design", "flex_design" ((h_0 [m], f_0) tuples),
        "build" and "wait" (best expected reward of the designs that upgrade / wait at the first
        decision [€]).
    """
    multi = not isinstance(param_name, str)
    param_names = tuple(param_name) if multi else (param_name,)
    swept = {key: [] for key in ("inflex", "flex", "inflex_design", "flex_design", "build", "wait")}  # fmt: skip
    for value in param_values:
        values = dict(
            zip(param_names, tuple(value) if multi else (value,), strict=True)
        )
        learning_model = values.pop("learning_model", None)
        submodels = _replace_swept_params(tuple(values), tuple(values.values()), sea_level_rise_model, risk_model, cost_model, backward_kwargs)  # fmt: skip
        if learning_model is None:
            _, rewards = backward_induction(*submodels)
        else:
            if learning_models is None:
                raise ValueError(
                    'param_name "learning_model" requires learning_models.'
                )
            rewards = compare_learning_models(*submodels, {learning_model: learning_models[learning_model]})[learning_model]["rewards"]  # fmt: skip
        _, inflex, flex = extract_optimal_designs(rewards)
        for name, design in (("inflex", inflex), ("flex", flex)):
            swept[name].append(float(design[-1]))
            swept[f"{name}_design"].append(tuple(float(v) for v in design[:-1]))
        t0 = rewards.t0.reset_index()
        waits = np.isclose(t0["h_0"], rewards.h_minus_1)
        swept["build"].append(float(t0.loc[~waits, "reward"].max()))
        swept["wait"].append(float(t0.loc[waits, "reward"].max()))
    return swept


def sweep_results(swept: dict[str, list]) -> dict[str, np.ndarray]:
    """Expected costs and value of flexibility of a `sweep_designs` result.

    Returns:
        Dict of arrays parallel to the sweep values: "cost_inflex", "cost_flex" (expected cost [€]),
        "vof" (value of flexibility [€]), "relative_vof" (vof / cost_inflex [-]) and "waiting"
        (True where waiting (deferral) at the first decision is optimal).
    """
    cost_inflex = -np.array(swept["inflex"])
    cost_flex = -np.array(swept["flex"])
    vof = cost_inflex - cost_flex
    return {
        "cost_inflex": cost_inflex,
        "cost_flex": cost_flex,
        "vof": vof,
        "relative_vof": vof / cost_inflex,
        "waiting": np.array(swept["wait"]) > np.array(swept["build"]),
    }


def sweep_summary(
    sweeps: dict[str, dict[str, list]], table: bool = True
) -> pd.DataFrame:
    """Range of the relative value of flexibility [%] of several sweeps, sorted by decreasing spread
    (max - min) [percentage points].

    Args:
        sweeps: Sweep name -> result of `sweep_designs`.
        table: Whether to print the table (thesis precision, see the module docstring).

    Returns:
        The table as a DataFrame with unrounded values.
    """
    rows = {}
    for name, swept in sweeps.items():
        relative = sweep_results(swept)["relative_vof"] * 100
        rows[name] = {"Min [%]": relative.min(), "Max [%]": relative.max(), "Spread [p.p.]": np.ptp(relative)}  # fmt: skip
    summary = pd.DataFrame(rows).T.sort_values("Spread [p.p.]", ascending=False)
    if table:
        print(tabulate(
            [[name, *(f"{value:.1f}" for value in row)] for name, row in summary.iterrows()],
            headers=["Sweep", *summary.columns],
            tablefmt="simple",
            disable_numparse=True,
        ))  # fmt: skip
    return summary


def cost_level(c: float, v: float, length_km: float = 10.0) -> float:
    """Cost level c_bar = (c + v * 1 m) / length [M€/km]: the cost of a 1 m upgrade per km of dike.

    Args:
        c: Fixed upgrade cost [€].
        v: Variable upgrade cost [€/m].
        length_km: Length of the dike section [km].

    Remarks:
        - Unlike the rest of the code, the result is in M€/km instead of SI units: it is only used
          to define and display sweep values.
    """
    return (c + v * 1.0) / EUR_PER_MEUR / length_km


def cost_level_pairs(
    c: float, v: float, levels: list[float], length_km: float = 10.0
) -> list[tuple[float, float]]:
    """(c, v) pairs with the cost levels `levels` [M€/km] (see `cost_level`) and the ratio c/v of
    the baseline (c [€], v [€/m])."""
    factor = 1 / cost_level(c, v, length_km)
    return [(c * level * factor, v * level * factor) for level in levels]


def cost_ratio_pairs(
    c: float, v: float, ratios: list[float]
) -> list[tuple[float, float]]:
    """(c, v) pairs with the ratios c/v = `ratios` [m] (math.inf allowed) and the cost of a 1 m upgrade
    c + v * 1 m of the baseline (c [€], v [€/m])."""
    total = c + v * 1.0
    return [(total, 0.0) if math.isinf(r) else (total * r / (r + 1.0), total / (r + 1.0)) for r in ratios]  # fmt: skip


# ─── Plots ──────────────────────────────────────────────────────────────────────


def flexibility_colors(flex_values: tuple[float, ...] | list[float]) -> list[tuple]:
    """One color per flexibility value, from dark (least flexible) to light (most flexible),
    following the rank of each value."""
    distinct = sorted(set(flex_values))
    shades = np.linspace(*FLEXIBILITY_CMAP_RANGE, max(len(distinct), 2))
    cmap = plt.get_cmap(FLEXIBILITY_CMAP)
    return [cmap(shades[distinct.index(value)]) for value in flex_values]


def _design_label(spec: FlexibilitySpec, name: str, design: tuple[float, ...]) -> str:
    """Legend label of a design, e.g. "Flexible dike ($\\tilde{\\varphi}$ = 1)"."""
    symbol = spec.labels[0].strip("$")
    if spec.height_like[0]:
        value = (
            r"h_0"
            if name == "Inflexible"
            else f"{design[1] * 100:.0f}" + r"\,\mathrm{cm}"
        )
    else:
        value = f"{design[1]:.2g}"
    return f"{name} dike (${symbol} = {value}$)"


def _format_flex(spec: FlexibilitySpec, value: float) -> str:
    return f"{value * 100:.0f} cm" if spec.height_like[0] else f"{value:.2f}"


def _trajectory_designs(
    rewards: RewardsTable,
    flexibility: float | None,
    flex_values: tuple[float, ...],
    show_inflexible: bool = True,
) -> dict[str, tuple[tuple[float, ...], str, object]]:
    """Designs shown along a trajectory: name -> (design, legend label, color)."""
    spec = rewards.spec
    flex = extract_optimal_design_at_flexibility(rewards, flexibility)
    designs = {}
    if show_inflexible:
        _, inflex, _ = extract_optimal_designs(rewards)
        designs["Inflexible"] = (inflex, _design_label(spec, "Inflexible", inflex), INFLEXIBLE_DIKE_COLOR)  # fmt: skip
    designs["Flexible"] = (flex, _design_label(spec, "Flexible", flex), FLEXIBLE_DIKE_COLOR)
    sorted_values = sorted(set(flex_values))
    for value, color in zip(sorted_values, flexibility_colors(sorted_values)):
        design = extract_optimal_design_at_flexibility(rewards, value)
        label = f"{spec.labels[0]} = {_format_flex(spec, value)}"
        designs[label] = (design, label, color)  # type: ignore
    return designs  # type: ignore


def _decision_years(sea_level_rise_model: SeaLevelRiseModel) -> tuple[np.ndarray, int, list[int]]:  # fmt: skip
    """(decision years, year of the first decision, lattice time steps); decisions are drawn at the
    middle of the lattice time steps."""
    time_steps, lattice_start_year = sea_level_rise_model.lattice.time_steps()
    start_year = lattice_start_year + (time_steps[0] + time_steps[1]) // 2
    return np.array([start_year + t for t in time_steps]), start_year, time_steps


def _draw_decisions(
    ax1: Axes,
    trajectory: list[float],
    sea_level_rise_model: SeaLevelRiseModel,
    cost_model: CostModel,
    spec: FlexibilitySpec,
    designs: dict,
    paths: dict,
    show_flexibility: bool,
    show_land_availability: bool,
    title: str,
    xlim: tuple[int, int] | None,
) -> tuple[Axes, list[float]]:
    """Draws the lattice and trajectory (left axis) and the capacities of the designs (right axis).
    Returns the right axis and the capacities [cm] to tick on it."""
    lattice = sea_level_rise_model.lattice
    years, start_year, time_steps = _decision_years(sea_level_rise_model)
    mask = np.ones(len(years), dtype=bool) if xlim is None else (years >= xlim[0]) & (years <= xlim[1])  # fmt: skip
    plot_trajectory(lattice, trajectory, show_lattice=True, title=title, xlim=xlim, ax=ax1)  # fmt: skip
    ax1.set_xticks(years[mask])

    ax2 = ax1.twinx()  # Dike capacities
    capacities = []
    for name, (_, label, color) in designs.items():
        heights, flex_states, _ = paths[name]
        ax2.step(years[mask], np.array(heights)[mask] * 100, where="post", linewidth=2.5, color=color, label=label)  # fmt: skip
        capacities.extend(np.array(heights)[mask] * 100)
        if show_flexibility and spec.height_like[0] and name != "Inflexible":
            h_max = np.array([f[0] for f in flex_states])[mask] * 100
            ax2.step(years[mask], h_max, where="post", color=color, linestyle="--", linewidth=1.5, label=f"Secured height {spec.labels[0]} ({label})")  # fmt: skip
            capacities.extend(h_max)

    if isinstance(cost_model, LandCostModel) and show_land_availability:
        # Evaluated on a fine time grid, as the available height decreases nonlinearly over time
        fine_years = np.linspace(years[mask][0], years[mask][-1], 400)
        step_idx = np.clip(np.searchsorted(years, fine_years, side="right") - 1, 0, len(years) - 1)  # fmt: skip
        tau = fine_years - start_year - time_steps[0]
        for name, (_, label, color) in designs.items():
            h_max = np.array([f[0] for f in paths[name][1]])[step_idx]
            h_available = [available_crest_height(h, t, cost_model.h_ref, cost_model.W_available_0, cost_model.lambda_enc, cost_model.kappa_1, cost_model.kappa_2) for h, t in zip(h_max, tau)]  # fmt: skip
            ax2.plot(fine_years, np.array(h_available) * 100, color=color, linestyle=":", linewidth=1.5, label=f"Available crest height ({label})")  # fmt: skip
    ax2.set_ylabel("Dike capacity [cm]")
    return ax2, capacities


def _align_capacity_axes(panels: list[tuple[Axes, Axes, float]]) -> None:
    """Gives all panels (left axis, right axis, offset [cm]) the same scale, such that a value y on
    the left (sea level rise) axis is at y + offset on the right (capacity) axis."""
    panels[0][0].figure.canvas.draw()
    low = min(min(ax1.get_ylim()[0], ax2.get_ylim()[0] - offset) for ax1, ax2, offset in panels)  # fmt: skip
    high = max(max(ax1.get_ylim()[1], ax2.get_ylim()[1] - offset) for ax1, ax2, offset in panels)  # fmt: skip
    for ax1, ax2, offset in panels:
        ax1.set_ylim(low, high)
        ax2.set_ylim(low + offset, high + offset)


def _set_capacity_ticks(
    ax2: Axes, capacities: list[float], priority: list[float]
) -> None:
    """Ticks the capacities the dikes take [cm] on the right axis, dropping a tick that would overlap
    another. The capacities in `priority` (the initial capacities h_0) are placed first."""
    fig = ax2.figure
    fig.canvas.draw()  # Final axes height
    low, high = ax2.get_ylim()
    height_pt = ax2.get_window_extent().height * 72 / fig.dpi
    label_pt = ax2.yaxis.get_ticklabels()[0].get_fontsize() if ax2.yaxis.get_ticklabels() else 10  # fmt: skip
    min_gap = 1.15 * label_pt / height_pt * (high - low)  # type: ignore # [cm]
    ticks: list[float] = []
    for capacity in sorted(set(priority)) + sorted(set(capacities)):
        if low <= capacity <= high and all(
            abs(capacity - tick) >= min_gap for tick in ticks
        ):
            ticks.append(capacity)
    ticks.sort()
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{tick:.0f}" for tick in ticks])


def _upgrades(heights: list[float], h_minus_1: float) -> np.ndarray:
    """Sizes of the upgrades [m] along a decision path (`decision_path`)."""
    steps = np.diff([h_minus_1, *heights])
    return steps[steps > 1e-9]


def _print_decision_table(
    rewards: RewardsTable, designs: dict, paths_by_trajectory: dict[str, dict]
) -> None:
    """Prints the expected cost, (relative) value of flexibility of every design (if the inflexible
    dike is shown) and, per trajectory, its realized cost, number of upgrades and mean upgrade size."""
    spec = rewards.spec
    with_vof = "Inflexible" in designs
    cost_inflex = -designs["Inflexible"][0][-1] if with_vof else 0.0
    headers = ["Design", "h_0 [cm]", spec.components[0] + ("_0 [cm]" if spec.height_like[0] else "_0"), "E[cost] [M€]"]  # fmt: skip
    if with_vof:
        headers += ["VoF [M€]", "Relative VoF [%]"]
    for name in paths_by_trajectory:
        suffix = f" ({name})" if name else ""
        headers += [f"Realized cost{suffix} [M€]", f"Upgrades{suffix}", f"Mean upgrade{suffix} [cm]"]  # fmt: skip
    rows = []
    for name, (design, _, _) in designs.items():
        cost = -design[-1]
        row = [
            name + (" (waits)" if is_waiting(design, rewards.h_minus_1) else ""),
            f"{design[0] * 100:.0f}",
            f"{design[1] * 100:.0f}" if spec.height_like[0] else f"{design[1]:.2f}",
            f"{cost / EUR_PER_MEUR:.1f}",
        ]
        if with_vof:
            row += [f"{(cost_inflex - cost) / EUR_PER_MEUR:.1f}", f"{(cost_inflex - cost) / cost_inflex * 100:.1f}"]  # fmt: skip
        for paths in paths_by_trajectory.values():
            heights, _, realized = paths[name]
            upgrades = _upgrades(heights, rewards.h_minus_1)
            row += [f"{-realized / EUR_PER_MEUR:.1f}", f"{len(upgrades)}", f"{upgrades.mean() * 100:.0f}" if len(upgrades) else "-"]  # fmt: skip
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True))


def _plot_decision_panels(
    panels: dict[str, tuple[RewardsTable, list[float], SeaLevelRiseModel]],
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    flexibility: float | None,
    flex_values: tuple[float, ...],
    show_flexibility: bool,
    show_land_availability: bool,
    title: str | None,
    xlim: tuple[int, int] | None,
    axes: list[Axes] | None,
    show: bool,
    table: bool,
    show_inflexible: bool = True,
) -> Figure | SubFigure:
    """Draws the optimal decisions of one panel per entry of `panels` (name -> (rewards, trajectory,
    sea level rise model)) on common scales and prints one table per distinct `rewards`."""
    designs_by_rewards: dict[int, dict] = {}  # Panels of the same rewards share their designs # fmt: skip
    groups: dict[int, tuple[RewardsTable, dict]] = {}  # id(rewards) -> (rewards, panel name -> paths) # fmt: skip
    for name, (rewards, trajectory, sea_level_rise_model) in panels.items():
        if id(rewards) not in designs_by_rewards:
            designs_by_rewards[id(rewards)] = _trajectory_designs(rewards, flexibility, flex_values, show_inflexible)  # fmt: skip
            groups[id(rewards)] = (rewards, {})
        paths = {}
        for design_name, (design, _, _) in designs_by_rewards[id(rewards)].items():
            heights, flex_states = decision_path(rewards, trajectory, sea_level_rise_model, design)  # fmt: skip
            realized = compute_realized_reward(trajectory, heights, flex_states, sea_level_rise_model, risk_model, cost_model, backward_kwargs)  # fmt: skip
            paths[design_name] = (heights, flex_states, realized)
        groups[id(rewards)][1][name] = paths

    n = len(panels)
    created = axes is None
    if axes is None:
        fig, grid = plt.subplots(1, n, figsize=(10 if n == 1 else 7.5 * n, 6), squeeze=False)  # fmt: skip
        axes = list(grid[0])
    fig = axes[0].figure
    drawn, capacities, priorities = [], [], []
    for ax1, (name, (rewards, trajectory, sea_level_rise_model)) in zip(axes, panels.items()):  # fmt: skip
        designs = designs_by_rewards[id(rewards)]
        paths = groups[id(rewards)][1][name]
        panel_title = name if n > 1 else (title or "Optimal decisions along a sea level rise trajectory")  # fmt: skip
        ax2, panel_capacities = _draw_decisions(ax1, trajectory, sea_level_rise_model, cost_model, rewards.spec, designs, paths, show_flexibility, show_land_availability, panel_title, xlim)  # fmt: skip
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()  # type: ignore
        min_height = min(path[0][0] for path in paths.values()) * 100
        drawn.append((ax1, ax2, min_height - trajectory[0] * 100))  # Aligns the start of both axes # fmt: skip
        capacities.append(panel_capacities)
        priorities.append([design[0] * 100 for design, _, _ in designs.values()])

    legend = {}  # Label -> handle, over all panels
    for ax1, ax2, _ in drawn:
        for ax in (ax1, ax2):
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend.setdefault(label, handle)
    if n == 1:
        land = isinstance(cost_model, LandCostModel) and show_land_availability
        drawn[0][1].legend(list(legend.values()), list(legend), loc="upper center" if land else "upper left")  # fmt: skip
        if created:
            fig.tight_layout()  # type: ignore
    else:
        ncol = 3
        rows = -(-len(legend) // ncol)
        fig.legend(list(legend.values()), list(legend), loc="lower center", ncol=ncol, frameon=False)  # fmt: skip
        if title:
            fig.suptitle(title)
        if created:
            fig.tight_layout(rect=(0, 0.045 * rows + 0.01, 1, 1))  # type: ignore
    _align_capacity_axes(drawn)
    for i, (ax1, ax2, _) in enumerate(drawn):  # Common scales: label the outer axes only # fmt: skip
        if i > 0:
            ax1.set_ylabel("")
        if i < len(drawn) - 1:
            ax2.set_ylabel("")
    for (_, ax2, _), panel_capacities, priority in zip(drawn, capacities, priorities):
        _set_capacity_ticks(ax2, panel_capacities, priority)
    if show and created:
        display(fig)
        plt.close(fig)  # type: ignore
    if table:
        for i, (rewards, paths_by_trajectory) in enumerate(groups.values()):
            if len(groups) > 1:  # One table per panel, headed by its name
                ((name, paths),) = paths_by_trajectory.items()
                print(("\n" if i else "") + name)
                paths_by_trajectory = {"": paths}
            _print_decision_table(
                rewards, designs_by_rewards[id(rewards)], paths_by_trajectory
            )
    return fig


def plot_optimal_decisions_along_trajectories(
    rewards: RewardsTable,
    trajectories: dict[str, list[float]],
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    flexibility: float | None = None,
    flex_values: tuple[float, ...] = (),
    show_flexibility: bool = True,
    show_land_availability: bool = True,
    title: str | None = None,
    xlim: tuple[int, int] | None = None,
    axes: list[Axes] | None = None,
    show: bool = True,
    table: bool = True,
    show_inflexible: bool = True,
) -> Figure | SubFigure:
    """Plots the optimal decisions of the inflexible and the flexible dike along several trajectories
    side by side, on common scales, and prints one table of their expected and realized costs.

    Args:
        rewards: Result of `backward_induction`.
        trajectories: Name -> sea level rise at every lattice time step [m], e.g. {"Low": ...,
            "High": ...} from `sea_level_rise_model.sample_trajectories`. One panel each.
        sea_level_rise_model, risk_model, cost_model, backward_kwargs: As for `backward_induction`.
        flexibility: Flexibility value (phi_0, or h_max_0 [m]) of the design drawn as the flexible
            dike, with h_0 optimized. If None, the flexible dike is the overall optimal design.
        flex_values: Flexibility values of additional designs, drawn in blue next to the inflexible
            and flexible dike (each with h_0 optimized). Independent of `flexibility`: both can be
            given, e.g. flexibility=9.0 and flex_values=(5.0, 7.0).
        show_flexibility: Draw the secured height h_max of the flexible designs (land model only).
        show_land_availability: Draw the available crest height (`available_crest_height`) of every
            design (land model only).
        title: Figure title.
        xlim: Range of years shown.
        axes: Axes to draw on, one per trajectory; a new figure is created if None.
        show: Display the figure immediately (in a notebook), before the table; only for a new
            figure.
        table: Print the table: per design its first decision, expected cost, value of flexibility
            VoF = E[cost(inflexible)] - E[cost(design)], relative VoF (VoF / E[cost(inflexible)]),
            and per trajectory its realized cost, number of upgrades (the first decision included)
            and mean upgrade size.
        show_inflexible: Show the inflexible dike (and the VoF columns of the table). False shows only
            the flexible designs, e.g. for a cost model whose flexibility is pinned to one value (no
            inflexible design exists).

    Remarks:
        - The right (capacity) axis ticks the capacities the dikes take, thinned to avoid overlaps.
    """
    panels = {name: (rewards, trajectory, sea_level_rise_model) for name, trajectory in trajectories.items()}  # fmt: skip
    return _plot_decision_panels(
        panels, risk_model, cost_model, backward_kwargs, flexibility, flex_values, show_flexibility,
        show_land_availability, title, xlim, axes, show, table, show_inflexible,
    )  # fmt: skip


def plot_optimal_decisions_along_trajectory(
    rewards: RewardsTable,
    trajectory: list[float],
    sea_level_rise_model: SeaLevelRiseModel,
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    flexibility: float | None = None,
    flex_values: tuple[float, ...] = (),
    show_flexibility: bool = True,
    show_land_availability: bool = True,
    title: str = "Optimal decisions along a sea level rise trajectory",
    xlim: tuple[int, int] | None = None,
    ax: Axes | None = None,
    show: bool = True,
    table: bool = True,
    show_inflexible: bool = True,
) -> Figure | SubFigure:
    """Plots the optimal decisions of the inflexible and the flexible dike along one trajectory and
    prints their expected and realized costs. Same as `plot_optimal_decisions_along_trajectories`
    with a single trajectory (drawn on `ax`, if given); see there for the other arguments."""
    return plot_optimal_decisions_along_trajectories(
        rewards, {"": trajectory}, sea_level_rise_model, risk_model, cost_model, backward_kwargs,
        flexibility=flexibility, flex_values=flex_values, show_flexibility=show_flexibility,
        show_land_availability=show_land_availability, title=title, xlim=xlim,
        axes=None if ax is None else [ax], show=show, table=table, show_inflexible=show_inflexible,
    )  # fmt: skip


def plot_optimal_decisions_by_model(
    results: dict[str, dict],
    trajectories: dict[str, list[float]],
    risk_model: RiskModel,
    cost_model: CostModel,
    backward_kwargs: dict,
    flexibility: float | None = None,
    flex_values: tuple[float, ...] = (),
    show_flexibility: bool = True,
    show_land_availability: bool = True,
    title: str | None = None,
    xlim: tuple[int, int] | None = None,
    axes: list[Axes] | None = None,
    show: bool = True,
    table: bool = True,
    show_inflexible: bool = True,
) -> Figure | SubFigure:
    """Plots the optimal decisions of several sea level rise (learning) models side by side, each
    along one of its own trajectories, on common scales, and prints one table per model.

    Args:
        results: Name -> {"model", "rewards", ...}, the result of `compare_learning_models` (or a
            subset of it). One panel each.
        trajectories: Name -> sea level rise at every lattice time step [m] of that model, e.g. from
            `results[name]["model"].sample_trajectories`; the names select the panels.
        risk_model, cost_model, backward_kwargs: As for `backward_induction`.
        flexibility, flex_values, show_flexibility, show_land_availability, title, xlim, axes,
            show, table, show_inflexible: As for `plot_optimal_decisions_along_trajectories`.
    """
    panels = {name: (results[name]["rewards"], trajectory, results[name]["model"]) for name, trajectory in trajectories.items()}  # fmt: skip
    return _plot_decision_panels(
        panels, risk_model, cost_model, backward_kwargs, flexibility, flex_values, show_flexibility,
        show_land_availability, title, xlim, axes, show, table, show_inflexible,
    )  # fmt: skip


def plot_cost_by_capacity(
    rewards: RewardsTable,
    flex_values: tuple[float, ...] | None = None,
    optimal: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the expected cost of the first-decision designs against h_0 (`cost_by_capacity`), one
    curve per fixed flexibility value, and marks the optimum of every curve.

    Args:
        rewards: Result of `backward_induction`.
        flex_values: Fixed flexibility values (phi_0, or h_max_0 [m]), one curve each.
        optimal: Also draw the curve with the optimal flexibility at every h_0 (black, dashed).
        title: Plot title.
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = rewards.spec
    values = sorted(set(flex_values or ()))
    curves = [(f"{spec.labels[0]} = {_format_flex(spec, v)}", color, "-", cost_by_capacity(rewards, v)) for v, color in zip(values, flexibility_colors(values))]  # fmt: skip
    if optimal:
        curves.append((f"Optimal {spec.labels[0]}", "black", "--", cost_by_capacity(rewards)))  # type: ignore # fmt: skip
    for label, color, linestyle, (h_0s, costs) in curves:
        ax.plot(h_0s * 100, costs / EUR_PER_MEUR, color=color, linestyle=linestyle, label=label)  # fmt: skip
        best = np.argmin(costs)
        ax.scatter(h_0s[best] * 100, costs[best] / EUR_PER_MEUR, marker="o", s=40, color=color, zorder=4)  # fmt: skip
    ax.set_xlabel("Initial capacity $h_0$ [cm]")
    ax.set_ylabel("Expected cost [M€]")
    ax.set_title(title or "Expected cost by initial capacity")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()
    return fig


def plot_cost_by_flexibility(
    rewards: RewardsTable,
    initial_capacities: tuple[float, ...] | None = None,
    optimal_h_0: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the expected cost of the first-decision designs against the flexibility value
    (`cost_by_flexibility`), one curve per fixed h_0, and marks the optimum of every curve.

    Args:
        rewards: Result of `backward_induction`.
        initial_capacities: Fixed values of h_0 [m]; defaults to `around_optimum(rewards)`.
        optimal_h_0: Also draw the curve with the optimal h_0 at every flexibility value (black,
            dashed; waiting (deferral) included).
        title: Plot title.
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = rewards.spec
    scale = 100 if spec.height_like[0] else 1
    h_0s = sorted(set(initial_capacities or around_optimum(rewards)))
    curves = [(f"$h_0 = {h_0 * 100:.0f}$ cm", color, "-", cost_by_flexibility(rewards, h_0)) for h_0, color in zip(h_0s, flexibility_colors(h_0s))]  # fmt: skip
    if optimal_h_0:
        curves.append(("Optimal $h_0$", "black", "--", cost_by_flexibility(rewards)))  # type: ignore
    for label, color, linestyle, (values, costs) in curves:
        ax.plot(values * scale, costs / EUR_PER_MEUR, color=color, linestyle=linestyle, label=label)  # fmt: skip
        best = np.argmin(costs)
        ax.scatter(values[best] * scale, costs[best] / EUR_PER_MEUR, marker="o", s=40, color=color, zorder=4)  # fmt: skip
    ax.set_xlabel(f"{spec.labels[0]} [cm]" if spec.height_like[0] else spec.labels[0])
    ax.set_ylabel("Expected cost [M€]")
    ax.set_title(title or "Expected cost by flexibility")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()
    return fig


def match_ylim_spans(axes: list[Axes]) -> None:
    """Gives all `axes` the y-range of the one with the largest range, keeping each axis centered on
    its own data, so that slopes compare directly across panels with different cost levels."""
    limits = [ax.get_ylim() for ax in axes]
    span = max(high - low for low, high in limits)
    for ax, (low, high) in zip(axes, limits):
        center = (low + high) / 2
        ax.set_ylim(center - span / 2, center + span / 2)


def _new_axes(ax: Axes | None) -> tuple[Figure | SubFigure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return fig, ax
    return ax.figure, ax


def _format_sweep_value(value) -> str:
    """Compact text of one sweep value (tuples recursively, numbers to 4 significant digits)."""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_format_sweep_value(v) for v in value) + ")"
    if isinstance(value, numbers.Real):
        return f"{value:.4g}"
    return str(value)


def sweep_table(
    swept: dict[str, list],
    param_values: list,
    label: str = "Value",
    display_values: list | None = None,
) -> None:
    """Prints the result of `sweep_designs` as a table, one row per sweep value: expected cost of the
    inflexible and flexible dike, (relative) value of flexibility, h_0 of both dikes and whether
    waiting (deferral) is optimal.

    Args:
        swept: Result of `sweep_designs`.
        param_values: The sweep values passed to `sweep_designs`.
        label: Header of the first column (the parameter).
        display_values: Values shown in the first column (e.g. in display units); defaults to
            `param_values`.
    """
    results = sweep_results(swept)
    shown = param_values if display_values is None else display_values
    rows = [
        [
            _format_sweep_value(value),
            f"{cost_inflex / EUR_PER_MEUR:.1f}",
            f"{cost_flex / EUR_PER_MEUR:.1f}",
            f"{vof / EUR_PER_MEUR:.1f}",
            f"{relative * 100:.1f}",
            f"{inflex[0] * 100:.0f}",
            f"{flex[0] * 100:.0f}",
            "yes" if waiting else "",
        ]
        for value, cost_inflex, cost_flex, vof, relative, inflex, flex, waiting in zip(
            shown,
            results["cost_inflex"],
            results["cost_flex"],
            results["vof"],
            results["relative_vof"],
            swept["inflex_design"],
            swept["flex_design"],
            results["waiting"],
        )
    ]
    headers = [label, "E[cost] inflexible [M€]", "E[cost] flexible [M€]", "VoF [M€]", "Relative VoF [%]", "h_0 inflexible [cm]", "h_0 flexible [cm]", "Waiting optimal"]  # fmt: skip
    print(tabulate(rows, headers=headers, tablefmt="simple", disable_numparse=True))


def plot_sweep(
    swept: dict[str, list],
    param_values: list,
    view: str = "relative_vof",
    x_values: list | None = None,
    xlabel: str = "",
    xscale: str = "linear",
    xticklabels: list[str] | None = None,
    shade_waiting: bool = True,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
    table: bool = True,
) -> Figure | SubFigure:
    """Plots the result of `sweep_designs` and prints it as a table (`sweep_table`).

    Args:
        swept: Result of `sweep_designs`.
        param_values: The sweep values passed to `sweep_designs` (used for the table and, if
            `x_values` is None, the x-axis).
        view: "cost" (expected cost of the inflexible and flexible dike [M€]), "vof" (value of
            flexibility [M€]) or "relative_vof" (value of flexibility / expected cost of the
            inflexible dike [%]).
        x_values: Values on the x-axis (e.g. in display units). If None and the sweep values are not
            numbers, the x-axis is categorical.
        xlabel: Label of the x-axis.
        xscale: Scale of the x-axis ("linear" or "log").
        xticklabels: Labels of the sweep values on the x-axis (e.g. for a categorical axis).
        shade_waiting: Shade the sweep values at which waiting (deferral) at the first decision is optimal.
        ylim: Limits of the y-axis, e.g. to compare several sweeps on one scale.
        title: Plot title.
        ax: Axes to draw on; a new figure is created if None.
        show: Display the figure immediately (in a notebook) before the table.
        table: Print the results as a table (`sweep_table`).
    """
    if view not in ("cost", "vof", "relative_vof"):
        raise ValueError(
            f"Invalid view {view!r}. Choose from 'cost', 'vof', 'relative_vof'."
        )
    results = sweep_results(swept)
    categorical = x_values is None and not all(isinstance(v, numbers.Real) for v in param_values)  # fmt: skip
    if x_values is None:
        x_values = list(range(len(param_values))) if categorical else list(param_values)
    x = np.array(x_values, dtype=float)

    ax_created = ax is None
    fig, ax = _new_axes(ax)
    if shade_waiting and results["waiting"].any():
        # Each sweep value at which waiting (deferral) is optimal is shaded up to halfway to its neighbors
        positions = np.log(x) if xscale == "log" else x
        edges = np.concatenate(([positions[0]], (positions[:-1] + positions[1:]) / 2, [positions[-1]]))  # fmt: skip
        edges = np.exp(edges) if xscale == "log" else edges
        for i in np.flatnonzero(results["waiting"]):
            ax.axvspan(edges[i], edges[i + 1], color=WAITING_REGION_COLOR, zorder=0, label="Waiting optimal" if i == np.flatnonzero(results["waiting"])[0] else None)  # fmt: skip

    if view == "cost":
        ax.plot(x, results["cost_inflex"] / EUR_PER_MEUR, marker="o", color=INFLEXIBLE_DIKE_COLOR, label="Inflexible dike")  # fmt: skip
        ax.plot(x, results["cost_flex"] / EUR_PER_MEUR, marker="o", color=FLEXIBLE_DIKE_COLOR, label="Flexible dike")  # fmt: skip
        ax.set_ylabel("Expected cost [M€]")
    elif view == "vof":
        ax.plot(x, results["vof"] / EUR_PER_MEUR, marker="o", color="black", label="Value of flexibility")  # fmt: skip
        ax.set_ylabel("Value of flexibility [M€]")
    else:
        ax.plot(x, results["relative_vof"] * 100, marker="o", color="black", label="Relative value of flexibility")  # fmt: skip
        ax.set_ylabel("Relative value of flexibility [%]")
        ax.set_ylim(bottom=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    if not categorical:
        ax.set_xscale(xscale)
    if categorical or xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels or [_format_sweep_value(v) for v in param_values], rotation=0 if xticklabels else 45, ha="center" if xticklabels else "right")  # fmt: skip
        ax.minorticks_off()
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()
    if isinstance(fig, Figure):
        fig.tight_layout()
    if show and ax_created:
        display(fig)
        plt.close(fig)  # type: ignore
    if table:
        display_values = xticklabels if xticklabels is not None else (None if categorical else x_values)  # fmt: skip
        sweep_table(swept, param_values, label=xlabel or "Value", display_values=display_values)  # fmt: skip
    return fig
