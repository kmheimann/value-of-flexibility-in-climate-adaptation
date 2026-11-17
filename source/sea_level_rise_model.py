"""
sea_level_rise_model.py — Sea level rise model and learning lattice (thesis Chapter 4).

A sea level rise model fits a quantile function Q_t(prob) of sea level rise to the KNMI projections
(via source.data) and builds a lattice from it: a discrete set of sea level rise states per time step,
with transition probabilities between consecutive time steps. Updating projections while moving through
the lattice represents learning about sea level rise. `decision_analysis.backward_induction` only uses
`model.lattice`.

Sea level rise (learning) models:
    DirectFitSeaLevelRiseModel     Direct fit method (Völz et al. 2023b): a recombining binomial lattice
                                   whose nodes are fitted to the quantile function (thesis baseline).
    StaticGridSeaLevelRiseModel    No learning: every state can follow every state (Völz et al. 2023b).
    LinearGrowthSeaLevelRiseModel  Learning at once: the first transition picks a constant rate of
                                   sea level rise, which is then followed deterministically.

Lattices:
    Lattice, ForwardLattice, BackwardLattice
                                   Lattice data structure: states and transition probabilities, queries
                                   and trajectory sampling. Adapted from the Julia implementation of
                                   Völz et al. (2023b), with the transition probabilities p and 1 - p
                                   of the direct fit corrected (they were swapped).
    build_direct_fit_lattice, build_static_grid_lattice, build_linear_growth_lattice
                                   Build the lattice of each model from a quantile function.

Plots:
    print_lattice, plot_lattice, plot_trajectory, plot_trajectories, plot_conditioned_lattice,
    plot_quantile_fan, plot_static_boxplots, plot_lattice_and_boxplots, plot_boxplot_comparison,
    plot_lattice_boxplot_comparison, and helpers that illustrate the method
    (plot_direct_fit_construction, plot_projections_as_seen_from_future).

Units: sea level rise in m (plots in cm), time in years. Time steps t are years after a model's
start year (`rel_time_steps`).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, fields
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure
from pypalettes import load_palette
from scipy.stats import binom

from source.data import (
    extrapolate_quantiles,
    get_aggregated_quantile_fn_cached,
    load_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ─────────────────────────────────────────────────────────────────────────────


_SLR_LATTICE_CACHE_MAXSIZE = 100  # Number of fitted models kept in memory
_slr_lattice_cache: OrderedDict[tuple, tuple[Callable[[int, float], float], ForwardLattice]] = OrderedDict()  # fmt: skip


def clear_lattice_cache() -> None:
    """Frees the memory of the cache of fitted sea level rise models."""
    _slr_lattice_cache.clear()


def _freeze_field(value):
    """Hashable version of a model field (lists become tuples), used as a cache key."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_field(v) for v in value)
    return value


@dataclass(frozen=True)
class SeaLevelRiseModel(ABC):
    """Abstract sea level rise model: fits the quantile function and builds the lattice on creation.

    Subclasses add their own fields and implement `_build_lattice()`.

    Args:
        start_year: Year of the first lattice time step (the first decision) [yr].
        end_year: Year of the last lattice time step [yr].
        step: Time step of the lattice [yr].
        method: How the SSP scenarios are combined, "mixture" (thesis) or "uncertainty_band" (see
            `data.get_aggregated_quantile_fn`).
        distribution: Distribution fitted per scenario; None for the default (Pearson type IV).
        weights: Scenario weights (SSP1-2.6, SSP2-4.5, SSP5-8.5) [-].
        extrapolate: Extrapolate the KNMI data beyond 2100 (needed for end years after 2100).
        align_to_lattice_midpoint: Fit every lattice time step t to the year start_year + t - step/2,
            i.e. a state represents the sea level rise in the middle of the preceding time step.

    Attributes:
        quantile_fn: Fitted quantile function quantile_fn(t, prob) [m], t in years after the fit start.
        lattice: The lattice (a `ForwardLattice`).

    Remarks:
        - Fitted models are cached in memory, keyed on the class and fields; `clear_lattice_cache()`
          frees the cache.
    """

    start_year: int
    end_year: int
    step: int
    method: str
    distribution: str | None
    weights: list[float] | None
    extrapolate: bool
    align_to_lattice_midpoint: bool = True
    quantile_fn: Callable[[int, float], float] = field(
        init=False, repr=False, compare=False
    )
    lattice: ForwardLattice = field(init=False, repr=False, compare=False)

    @property
    def rel_time_steps(self) -> list[int]:
        return list(range(0, self.end_year - self.start_year + 1, self.step))

    @property
    def fit_start_year(self) -> int:
        return (
            self.start_year - self.step // 2
            if self.align_to_lattice_midpoint
            else self.start_year
        )

    @property
    def fit_end_year(self) -> int:
        return (
            self.end_year - self.step // 2
            if self.align_to_lattice_midpoint
            else self.end_year
        )

    def __post_init__(self) -> None:
        key = (type(self),) + tuple(
            (f.name, _freeze_field(getattr(self, f.name)))
            for f in fields(self)
            if f.compare
        )
        if key in _slr_lattice_cache:
            _slr_lattice_cache.move_to_end(key)
            fn, lattice = _slr_lattice_cache[key]
            object.__setattr__(self, "quantile_fn", fn)
            object.__setattr__(self, "lattice", lattice)
            return

        fn, _ = get_aggregated_quantile_fn_cached(self.fit_start_year, self.fit_end_year, method=self.method, distribution=self.distribution, weights=self.weights, extrapolate=self.extrapolate)  # fmt: skip
        object.__setattr__(self, "quantile_fn", fn)
        lattice = self._build_lattice()
        object.__setattr__(self, "lattice", lattice)

        _slr_lattice_cache[key] = (fn, lattice)
        if len(_slr_lattice_cache) > _SLR_LATTICE_CACHE_MAXSIZE:
            _slr_lattice_cache.popitem(last=False)  # Evict the least-recently-used entry # fmt: skip

    @abstractmethod
    def _build_lattice(self) -> ForwardLattice:
        """Builds the lattice from `quantile_fn` and `rel_time_steps`."""

    def sample_trajectories(
        self, n: int, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> Generator[list[float]]:
        """Samples `n` trajectories (lists of states [m], one per time step) from the lattice.

        Args:
            n: Number of trajectories.
            seed: Seed of a new random generator, for reproducible samples.
            rng: Random generator to use instead of `seed`.
        """
        yield from self.lattice.sample_trajectories(n, seed=seed, rng=rng)


@dataclass(frozen=True)
class DirectFitSeaLevelRiseModel(SeaLevelRiseModel):
    """Direct fit sea level rise model (Völz et al. 2023b; thesis baseline): a recombining binomial
    lattice whose nodes are fitted to the quantile function.

    Args (in addition to `SeaLevelRiseModel`):
        p: Probability of an upward move [-] (thesis: p).
        skip: Grow by two nodes per step (not used in the thesis).
        shift: Number of extra nodes at the first time step (not used in the thesis).
    """

    p: float = 0.5
    skip: bool = False
    shift: int = 0

    def _build_lattice(self) -> ForwardLattice:
        return build_direct_fit_lattice(self.quantile_fn, self.p, self.rel_time_steps, self.fit_start_year, skip=self.skip, shift=self.shift)  # fmt: skip

    def sample_trajectories(
        self, n: int, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> Generator[list[float]]:
        """Same as `SeaLevelRiseModel.sample_trajectories` (same samples), faster for the direct fit
        lattice.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        rel_time_steps = self.rel_time_steps
        lattice = self.lattice
        rvs = rng.binomial(n=1, p=self.p, size=(n, len(rel_time_steps) - 1))  # Pre-sample all binomial outcomes for efficiency # fmt: skip

        def build_trajectory(binomial_outcomes):
            trajectory = [lattice.states(rel_time_steps[0])[0]]  # Start with the root node # fmt: skip
            for t, outcome in zip(rel_time_steps[:-1], binomial_outcomes):
                trajectory.append(lattice.get_transition_probs(t, trajectory[-1], None)[outcome][1])  # fmt: skip
            return trajectory

        yield from map(build_trajectory, rvs)


@dataclass(frozen=True)
class StaticGridSeaLevelRiseModel(SeaLevelRiseModel):
    """Static grid sea level rise model (no learning, Völz et al. 2023b): every state can follow
    every state.

    Args (in addition to `SeaLevelRiseModel`):
        n_nodes_list: Number of nodes per time step; defaults to 1, 2, 3, ...
    """

    n_nodes_list: list[int] | None = None

    def _build_lattice(self) -> ForwardLattice:
        return build_static_grid_lattice(self.quantile_fn, self.rel_time_steps, self.fit_start_year, n_nodes_list=self.n_nodes_list)  # fmt: skip


@dataclass(frozen=True)
class LinearGrowthSeaLevelRiseModel(SeaLevelRiseModel):
    """Linear growth sea level rise model (all learning at once): the first transition picks a
    constant rate of sea level rise, which is then followed deterministically.

    Args (in addition to `SeaLevelRiseModel`):
        p: Probability of an upward move of the matching direct fit lattice [-]; defaults to 0.5.
        n_nodes: Number of rates (nodes after the first time step); defaults to the number of time
            steps.
    """

    p: float | None = None
    n_nodes: int | None = None

    def _build_lattice(self) -> ForwardLattice:
        return build_linear_growth_lattice(self.quantile_fn, self.rel_time_steps, self.fit_start_year, p=self.p, n_nodes=self.n_nodes)  # fmt: skip


class Lattice(ABC):
    """Abstract lattice: discrete sea level rise states per time step with transition probabilities
    between adjacent time steps, stored as

        _data[idx][state][action] = [(probability, adjacent_state), ...]

    with action None for sea level rise. Subclasses fix the direction of the transitions:
    `ForwardLattice` (to the next time step) or `BackwardLattice` (to the previous one).
    """

    direction: ClassVar[Literal["forward", "backward"]]

    def __init__(self, rel_time_steps: list[int], start_year: int) -> None:
        """Args:
            rel_time_steps: Time steps t [yr after start_year], in order. Need not start at 0 (the
                lattices of `ForwardLattice.propagate_observation` keep the original t).
            start_year: Calendar year of t = 0.
        """
        self._data: dict = {}
        self._rel_time_steps = rel_time_steps  # Relative time indices, in years relative to start_year # fmt: skip
        self._start_year = start_year  # Reference year for relative time indices
        self._time_to_idx = {
            t: idx for idx, t in enumerate(rel_time_steps)
        }  # O(1) time lookup

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rel_time_steps={self._rel_time_steps}, start_year={self._start_year})"

    @property
    def _boundary_idx(self) -> int:
        """Index that carries no outgoing transitions (last index forward, first index backward)."""
        return len(self._rel_time_steps) - 1 if self.direction == "forward" else 0

    @property
    def _step(self) -> int:
        """Index offset from a time step to the adjacent one its transitions point to."""
        return 1 if self.direction == "forward" else -1

    @property
    def _traversal_indices(self) -> range:
        """All indices that can carry outgoing transitions, in traversal order."""
        if self.direction == "forward":
            return range(len(self._rel_time_steps) - 1)
        return range(len(self._rel_time_steps) - 1, 0, -1)

    # ── write ─────────────────────────────────────────────────────────────────

    def add_transition(
        self,
        t: int,
        state: float,
        action,
        dist: list[tuple[float, float]],
    ) -> None:
        """Adds the transitions from `state` at time step t.

        Args:
            t: Time step [yr after start_year].
            state: Sea level rise state [m].
            action: Action; None for sea level rise lattices.
            dist: Transitions [(probability, adjacent_state), ...]; the probabilities sum to 1.
        """
        if not np.isclose(sum(prob for prob, _ in dist), 1.0):
            raise ValueError(
                f"Transition probabilities for (t={t}, state={state:.4f}, action={action}) do not sum to 1."
            )
        idx = self._time_to_idx.get(t)
        if idx is None or idx == self._boundary_idx:
            raise ValueError(
                f"Time step {t} is not valid for this {type(self).__name__}."
            )
        if idx not in self._data:
            self._data[idx] = {}
        if state not in self._data[idx]:
            self._data[idx][state] = {}
        if action in self._data[idx][state]:
            raise ValueError(
                f"Transition (t={t}, state={state:.4f}, action={action}) already registered."
            )
        # Sort once on insertion by state value
        self._data[idx][state][action] = sorted(dist, key=lambda x: x[1])

    # ── read ──────────────────────────────────────────────────────────────────

    def time_steps(self) -> tuple[list[int], int]:
        """Returns (rel_time_steps, start_year)."""
        return self._rel_time_steps, self._start_year

    def iter_steps(self) -> Iterator[tuple[int, int]]:
        """Yields (t, t_adjacent) for every time step with transitions, in traversal order; t_adjacent is
        the step its transitions point to (next step forward, previous step backward).
        """
        for idx in self._traversal_indices:
            if idx not in self._data:
                continue  # skip time steps with no registered transitions
            t = self._rel_time_steps[idx]
            yield t, self._rel_time_steps[idx + self._step]

    def states(self, t: int) -> list[float]:
        """States [m] at time step t; if t has no transitions, the states reachable from the adjacent
        step.
        """
        idx = self._time_to_idx.get(t)
        # Check if the time step exists in the lattice
        if idx is None:
            raise ValueError(f"Time step {t} not found in {type(self).__name__}.")

        if idx in self._data:
            return list(self._data[idx])
        fallback_idx = idx - self._step
        if fallback_idx in self._data:
            # If the adjacent time step exists, return states reachable from it
            return self._reachable_states(self._rel_time_steps[fallback_idx])

        raise ValueError(f"Time step {t} not found in {type(self).__name__}.")

    def actions(self, t: int, state: float) -> list:
        """Actions available at (t, state); sea level rise lattices have the single action None."""
        idx = self._time_to_idx.get(t)
        # Check if the time step exists in the lattice
        if idx is None or idx not in self._data:
            raise ValueError(f"Time step {t} not found in {type(self).__name__}.")

        # Check if the state exists at the given time step
        if state not in self._data[idx]:
            raise ValueError(f"State {state} not found at time step {t}.")

        return list(self._data[idx][state])

    # ── internal ──────────────────────────────────────────────────────────────

    def _reachable_states(self, t: int, state: float | None = None) -> list[float]:
        """Unique adjacent states reachable from (t, state), or from all states at t if state is None."""
        idx = self._time_to_idx.get(t)
        # Check if the time step exists in the lattice
        if idx is None or idx not in self._data:
            raise ValueError(f"Time step {t} not found in {type(self).__name__}.")
        if idx == self._boundary_idx:
            return []  # Terminal step; no successors exist

        # If a specific state is provided, check if it exists and return its successors
        if state is not None:
            if state not in self._data[idx]:
                raise ValueError(f"State {state} not found at time step {t}.")
            return [s for dist in self._data[idx][state].values() for _, s in dist]

        # If no specific state is provided, return all unique successors from all states at time t
        seen: set = set()
        result: list = []
        for s in self._data[idx]:
            for s_adjacent in self._reachable_states(t, s):
                key = round(
                    s_adjacent, 10
                )  # deduplicate floats within numerical precision
                if key not in seen:
                    seen.add(key)
                    result.append(s_adjacent)
        return result

    # ── probabilities and sampling ──────────────────────────────────────────────────────────────

    def get_transition_probs(
        self, t: int, state: float, action
    ) -> list[tuple[float, float]]:
        """Transitions from (t, state): [(probability, adjacent_state), ...]."""
        idx = self._time_to_idx.get(t)
        # Check if the time step exists in the lattice
        if idx is None or idx not in self._data:
            raise ValueError(f"Time step {t} not found in {type(self).__name__}.")
        if idx == self._boundary_idx:
            return []  # Terminal step; no successors exist

        # Check if the state exists at the given time step
        if state not in self._data[idx]:
            raise ValueError(f"State {state} not found at time step {t}.")

        # Check if the action exists for the given state at the given time step
        if action not in self._data[idx][state]:
            raise ValueError(f"Action {action} not found for state {state} at time step {t}.")  # fmt: skip

        return self._data[idx][state][action]  # Already sorted on insertion

    def get_marginal_probs(self) -> dict[int, dict[float, float]]:
        """Marginal probability of every state per time step: {t: {state: probability}}."""
        marginal_probs: dict[int, dict[float, float]] = {}

        # Initialize the marginal probabilities for the first time step. min/max(self._data) identifies
        # the populated root that iter_steps() alone can't find, since iter_steps() excludes _boundary_idx.
        start_idx = min(self._data) if self.direction == "forward" else max(self._data)
        t_start = self._rel_time_steps[start_idx]
        marginal_probs[t_start] = {state: 1.0 for state in self.states(t_start)}

        # One step at a time, propagate the marginal probabilities through the lattice
        for t, t_adjacent in self.iter_steps():
            # For each adjacent state, identify the states that can transition to it and calculate the marginal probability
            for adjacent_state in self.states(t_adjacent):
                marginal_prob = 0.0
                for current_state in self.states(t):
                    for prob, s in self.get_transition_probs(t, current_state, None):
                        if s == adjacent_state:
                            marginal_prob += prob * marginal_probs[t][current_state]  # fmt: skip

                if t_adjacent not in marginal_probs:
                    marginal_probs[t_adjacent] = {}
                marginal_probs[t_adjacent][adjacent_state] = marginal_prob

        return marginal_probs

    def get_marginal_quantile_fn(self) -> Callable[[int, float], float]:
        """Quantile function q(t, prob) [m] of the marginal lattice distribution."""
        marginal_probs = self.get_marginal_probs()

        cache = {}
        for t in self._rel_time_steps:
            states = sorted(marginal_probs[t].keys())
            probs = [marginal_probs[t][state] for state in states]
            cumulative_probs = np.cumsum(probs).tolist()
            cache[t] = get_generalized_inverse_fn(states, cumulative_probs)

        def quantile_fn(t: int, prob: float) -> float:
            if t not in cache:
                raise ValueError(f"Time step {t} not found in lattice quantile function.")  # fmt: skip
            return cache[t](prob)

        return quantile_fn

    def sample_adjacent_state(
        self,
        t: int,
        state: float,
        action,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Samples the adjacent state [m] of `state` at time step t.

        Args:
            t: Time step [yr after start_year].
            state: Current state [m].
            action: Action; None for sea level rise lattices.
            seed: Seed of a new random generator, for reproducible samples.
            rng: Random generator to use instead of `seed`.
        """
        dist = self.get_transition_probs(t, state, action)
        if not dist:
            kind = "successors" if self.direction == "forward" else "predecessors"
            raise ValueError(
                f"No {kind} found for (t={t}, state={state}, action={action})."
            )

        # Unzip probabilities and adjacent states
        probs, adjacent_states = zip(*dist)

        # Sample the adjacent state based on the transition probabilities
        if rng is None:
            rng = np.random.default_rng(seed)
        return float(rng.choice(adjacent_states, p=probs))

    def sample_trajectories(
        self, n: int, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> Generator[list[float]]:
        """Samples `n` trajectories (lists of states [m], one per time step), walking the lattice from
        its root in traversal order.

        Args:
            n: Number of trajectories.
            seed: Seed of a new random generator, for reproducible samples.
            rng: Random generator to use instead of `seed`.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        start_idx = min(self._data) if self.direction == "forward" else max(self._data)  # fmt: skip
        for _ in range(n):
            # Generate a trajectory one step at a time, starting from the root of the populated data
            trajectory = [self.states(self._rel_time_steps[start_idx])[0]]

            idx = start_idx
            while idx != self._boundary_idx and idx in self._data:
                t = self._rel_time_steps[idx]
                current_state = trajectory[-1]
                adjacent_state = self.sample_adjacent_state(t, current_state, None, rng=rng)  # fmt: skip
                trajectory.append(adjacent_state)
                idx += self._step

            yield trajectory


class ForwardLattice(Lattice):
    """A Lattice whose transitions point to the next time step."""

    direction: ClassVar[Literal["forward", "backward"]] = "forward"

    def sample_next_state(
        self,
        t: int,
        state: float,
        action,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Alias of `Lattice.sample_adjacent_state`."""
        return self.sample_adjacent_state(t, state, action, seed=seed, rng=rng)

    def propagate_observation(
        self,
        observed_year: int,
        observed_state: float,
    ) -> tuple[ForwardLattice, BackwardLattice, float]:
        """Conditions the lattice on an observation: the closest state at `observed_year` is propagated
        forward with the transition probabilities, and backward with Bayes' rule on the marginal
        probabilities.

        Args:
            observed_year: Time step of the observation [yr after start_year].
            observed_state: Observed sea level rise [m].

        Returns:
            (forward_lattice, backward_lattice, closest_state): a ForwardLattice from observed_year to
            the end, a BackwardLattice from the start to observed_year (both keep this lattice's t and
            start_year), and the lattice state closest to observed_state [m].

        Remarks:
            Requires a single root state (`get_marginal_probs` does not normalize several roots, e.g.
            a direct fit lattice with shift > 0).
        """
        rel_time_steps, start_year = self.time_steps()

        if observed_year not in rel_time_steps:
            raise ValueError("observed_year is not part of the lattice time horizon.")
        if len(self.states(rel_time_steps[0])) > 1:
            raise ValueError(
                "propagate_observation requires a single-root lattice; get_marginal_probs does not "
                "yet normalize a multi-node root (e.g. direct fit with shift > 0)."
            )
        observed_idx = rel_time_steps.index(observed_year)

        # Find closest state at observed_year
        states_at_obs = self.states(observed_year)
        closest_observed_state = min(
            states_at_obs, key=lambda s: abs(s - observed_state)
        )

        marginal_probs = self.get_marginal_probs()

        # Forward propagation lattice — spans [observed_year, ..., end]
        forward_lattice = ForwardLattice(rel_time_steps[observed_idx:], start_year)
        forward_lattice._data[0] = {
            closest_observed_state: {}
        }  # Seed the forward lattice with the closest observed state at the observed year so it is available in forward_lattice.states()

        for idx in range(observed_idx, self._boundary_idx):
            t = rel_time_steps[idx]

            # Collect nodes that can be reached in the original lattice starting from a state that is present in the forward lattice
            for state in forward_lattice.states(t):
                dist = list(self.get_transition_probs(t, state, None))
                forward_lattice.add_transition(t, state, None, dist)

        # Backward propagation lattice — spans [start, ..., observed_year]
        backward_lattice = BackwardLattice(
            rel_time_steps[: observed_idx + 1], start_year
        )
        backward_lattice._data[observed_idx] = {closest_observed_state: {}}

        for idx in range(observed_idx, 0, -1):
            t = rel_time_steps[idx]
            t_prev = rel_time_steps[idx - 1]

            # Collect nodes in the original lattice that can reach the present state in the backward lattice
            for state in backward_lattice.states(t):
                dist = []
                for prev_state in self.states(t_prev):
                    for prob, next_state in self.get_transition_probs(t_prev, prev_state, None):  # fmt: skip
                        if next_state == state:
                            back_prob = prob * marginal_probs[t_prev][prev_state] / marginal_probs[t][state]  # Bayes' rule # fmt: skip
                            dist.append((back_prob, prev_state))
                backward_lattice.add_transition(t, state, None, dist)

        return forward_lattice, backward_lattice, closest_observed_state


def build_direct_fit_lattice(
    quantile_fn: Callable[[int, float], float],
    p: float,
    rel_time_steps: list[int],
    start_year: int,
    *,
    skip: bool = False,
    shift: int = 0,
) -> ForwardLattice:
    """Builds the direct fit lattice (thesis baseline): a recombining binomial lattice whose node j
    at step m lies at quantile_fn(t, prob), with prob the binomial midpoint probability of node j
    (`_binomial_midpoint_prob`); each node moves up with probability p and down with 1 - p.

    Args:
        quantile_fn: Quantile function quantile_fn(t, prob) [m], t in years after start_year.
        p: Probability of an upward move [-].
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        skip: Grow by two nodes per step, so that every step has an odd number of nodes.
        shift: Extra nodes at the first time step (shift + 1 nodes); not combinable with skip.
    """
    if skip and shift > 0:
        raise ValueError("'skip' and 'shift > 0' cannot be used together.")

    lattice = ForwardLattice(rel_time_steps, start_year)
    m = shift  # binomial order at the current step; number of nodes = m + 1

    for idx in range(len(rel_time_steps) - 1):
        t = rel_time_steps[idx]
        t_next = rel_time_steps[idx + 1]
        m_next = m + 2 if skip else m + 1

        for j in range(m + 1):
            # Current node position
            pos_current = quantile_fn(t, _binomial_midpoint_prob(j, m, p))

            # Successor 1: same rank j in the next step (corresponds to "downward" move)
            pos_succ1 = quantile_fn(t_next, _binomial_midpoint_prob(j, m_next, p))

            # Successor 2: rank j+1 in the next step (corresponds to "upward" move)
            pos_succ2 = quantile_fn(t_next, _binomial_midpoint_prob(j + 1, m_next, p))

            if skip:
                # Not used in the thesis and not checked: the probabilities may be swapped, as they
                # were in the original implementation for skip=False
                # Three-outcome case; the second and third outcome share pos_succ2
                # (faithfully translated from the reference implementation)
                dist = [
                    (p**2, pos_succ1),
                    (2 * p * (1 - p), pos_succ2),
                    ((1 - p) ** 2, pos_succ2),
                ]
            else:
                # Successor 2 is the higher node, reached with probability p (corrected: the original
                # implementation swapped p and 1 - p)
                dist = [(1 - p, pos_succ1), (p, pos_succ2)]

            lattice.add_transition(t, pos_current, None, dist)

        m = m_next

    return lattice


def build_static_grid_lattice(
    quantile_fn: Callable[[int, float], float],
    rel_time_steps: list[int],
    start_year: int,
    *,
    n_nodes_list: list[int] | None = None,
) -> ForwardLattice:
    """Builds the static grid lattice (no learning): the N nodes of a time step lie at the
    (k + 0.5) / N quantiles, and each connects to every node of the next step with probability 1 / N.

    Args:
        quantile_fn: Quantile function quantile_fn(t, prob) [m], t in years after start_year.
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        n_nodes_list: Number of nodes per time step, starting with 1; defaults to 1, 2, 3, ...
    """
    if n_nodes_list is None:
        n_nodes_list = list(range(1, len(rel_time_steps) + 1))

    lattice = ForwardLattice(rel_time_steps, start_year)

    for idx in range(len(rel_time_steps) - 1):
        t = rel_time_steps[idx]
        t_next = rel_time_steps[idx + 1]
        n = n_nodes_list[idx]
        n_next = n_nodes_list[idx + 1]
        prob = 1.0 / n_next

        succ_positions = [
            quantile_fn(t_next, (k + 0.5) / n_next) for k in range(n_next)
        ]
        dist = [(prob, pos) for pos in succ_positions]

        if idx == 0:
            # First time step: single node at the median
            position = quantile_fn(t, 0.5)
            lattice.add_transition(t, position, None, dist)
        else:
            # General case: n uniformly-spaced nodes
            for k in range(n):
                position = quantile_fn(t, (k + 0.5) / n)
                lattice.add_transition(t, position, None, dist)

    return lattice


def build_linear_growth_lattice(
    quantile_fn: Callable[[int, float], float],
    rel_time_steps: list[int],
    start_year: int,
    *,
    p: float | None = None,
    n_nodes: int | None = None,
) -> ForwardLattice:
    """Builds the linear growth lattice (all learning at once): the first transition picks one of
    `n_nodes` constant rates of sea level rise, which later transitions follow deterministically.
    The rates are the final-step quantiles at the binomial midpoint probabilities (as in the direct
    fit), divided by the time horizon.

    Args:
        quantile_fn: Quantile function quantile_fn(t, prob) [m], t in years after start_year.
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        p: Probability of an upward move of the matching direct fit lattice [-]; defaults to 0.5.
        n_nodes: Number of rates (>= 2); defaults to the number of time steps.
    """
    if p is None:
        p = 0.5
    if n_nodes is None:
        n_nodes = len(rel_time_steps)

    if n_nodes < 2:
        raise ValueError("'n_nodes' must be at least 2.")
    if len(rel_time_steps) < 2:
        raise ValueError("'rel_time_steps' must contain at least two entries.")

    t_start = rel_time_steps[0]
    t_end = rel_time_steps[-1]
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError("'rel_time_steps' must be strictly increasing.")

    lattice = ForwardLattice(rel_time_steps, start_year)

    root_state = quantile_fn(t_start, 0.5)  # single root node at the median

    probs = [_binomial_midpoint_prob(j, n_nodes - 1, p) for j in range(n_nodes)]
    rates = [(quantile_fn(t_end, prob) - root_state) / duration for prob in probs]

    first_dist = [
        (
            float(binom.pmf(j, n_nodes - 1, p)),
            root_state + rate * (rel_time_steps[1] - t_start),
        )
        for j, rate in enumerate(rates)
    ]
    lattice.add_transition(t_start, root_state, None, first_dist)

    for idx in range(1, len(rel_time_steps) - 1):
        t = rel_time_steps[idx]
        t_next = rel_time_steps[idx + 1]
        elapsed = t - t_start
        next_elapsed = t_next - t_start

        for rate in rates:
            state = root_state + rate * elapsed
            next_state = root_state + rate * next_elapsed
            lattice.add_transition(t, state, None, [(1.0, next_state)])

    return lattice


class BackwardLattice(Lattice):
    """A Lattice whose transitions point to the previous time step; the result of conditioning on an
    observation (see `ForwardLattice.propagate_observation`).
    """

    direction: ClassVar[Literal["forward", "backward"]] = "backward"

    def sample_previous_state(
        self,
        t: int,
        state: float,
        action,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Alias of `Lattice.sample_adjacent_state`."""
        return self.sample_adjacent_state(t, state, action, seed=seed, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _plot_box_like(
    ax,
    x: float,
    q01: float,
    q25: float,
    q50: float,
    q75: float,
    q99: float,
    *,
    color: str,
    label: str,
    cap_width: float,
    box_width: float,
    alpha: float = 0.35,
) -> None:
    # Whiskers
    ax.plot([x, x], [q01, q99], color=color, linewidth=1.2, label="")
    # Caps
    ax.plot(
        [x - cap_width, x + cap_width], [q01, q01], color=color, linewidth=1.2, label=""
    )
    ax.plot(
        [x - cap_width, x + cap_width], [q99, q99], color=color, linewidth=1.2, label=""
    )
    # Fill Box
    ax.fill_between(
        [x - box_width, x + box_width],
        [q25, q25],
        [q75, q75],
        color=color,
        alpha=alpha,
        label=label,
    )
    # Outline Box
    box_x = [x - box_width, x + box_width, x + box_width, x - box_width, x - box_width]
    box_y = [q25, q25, q75, q75, q25]
    ax.plot(box_x, box_y, color=color, linewidth=1.2, label="")
    # Median
    ax.plot(
        [x - box_width, x + box_width],
        [q50, q50],
        color="black",
        linewidth=1.2,
        label="",
    )


def _binomial_midpoint_prob(j: int, m: int, p: float) -> float:
    """Probability that represents node j of a Binomial(m, p) tree: the midpoint of its CDF interval
    [F(j - 1), F(j)], so that the nodes cover the full probability range.

    Args:
        j: Node rank, 0 <= j <= m.
        m: Binomial order (m + 1 nodes).
        p: Probability of an upward move [-].
    """
    upper = float(binom.cdf(j, m, p)) if m > 0 else 1.0
    lower = float(binom.cdf(j - 1, m, p)) if j > 0 else 0.0
    return (upper + lower) / 2.0


def get_generalized_inverse_fn(states: list[float], cumulative_probs: list[float]) -> Callable[[float], float]:  # fmt: skip
    """Quantile function (generalized inverse) of a discrete distribution with the sorted `states` and
    their cumulative probabilities `cumulative_probs`: q(prob) = smallest state with CDF >= prob."""
    # Check that the cumulative probabilities are valid
    if (
        cumulative_probs[0] < 0
        or not np.isclose(cumulative_probs[-1], 1)
        or any(
            cumulative_probs[i] < cumulative_probs[i - 1]
            for i in range(1, len(cumulative_probs))
        )
    ):
        raise ValueError(
            "Cumulative probabilities must be non-decreasing, non-negative, and end at 1."
        )

    def generalized_inverse(prob: float) -> float:
        if prob < 0 or prob > 1:
            raise ValueError(f"Probability level {prob} is out of bounds [0, 1].")

        # return generalized inverse (infimum of the set of states with cumulative probability >= prob)
        return next((state for state, cp in zip(states, cumulative_probs) if cp >= prob), states[-1])  # fmt:skip

    return generalized_inverse


# ─────────────────────────────────────────────────────────────────────────────
# Inspection and Visualization
# ─────────────────────────────────────────────────────────────────────────────


def _as_lattice(lattice: Lattice | SeaLevelRiseModel) -> Lattice:
    """Unwrap a SeaLevelRiseModel to its `.lattice`; pass a Lattice through unchanged."""
    return lattice.lattice if isinstance(lattice, SeaLevelRiseModel) else lattice


def _as_forward_lattice(lattice: ForwardLattice | SeaLevelRiseModel) -> ForwardLattice:
    """Unwrap a SeaLevelRiseModel to its `.lattice`; pass a ForwardLattice through unchanged."""
    return lattice.lattice if isinstance(lattice, SeaLevelRiseModel) else lattice


def print_lattice(lattice: Lattice | SeaLevelRiseModel) -> None:
    """Prints every state and its transitions per time step."""
    lattice = _as_lattice(lattice)
    for t, _ in lattice.iter_steps():
        print(f"t = {t}")
        try:
            ordered_states = sorted(lattice.states(t))
        except TypeError:
            ordered_states = lattice.states(t)  # unsortable state type; use as-is

        for state in ordered_states:
            print(f"  state = {state * 100:8.2f} cm")
            for action in lattice.actions(t, state):
                dist = lattice.get_transition_probs(t, state, action)
                entries = ",  ".join(f"p={prob:.3f} → {s * 100:8.2f} cm" for prob, s in dist)  # fmt: skip
                print(f"    action={action}:  {entries}")


def _set_sea_level_rise_plot_style(
    ax: plt.axes.Axes,
    title: str,
    start_year: int,
    rel_time_steps: list[int],
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Applies the shared axes style of the sea level rise plots (title, year ticks, labels,
    limits).
    """
    # Title and labels
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")

    # Ticks and Lims
    years = [start_year + t for t in rel_time_steps]
    if xlim is not None:
        years = [year for year in years if xlim[0] <= year <= xlim[1]]
    ax.set_xticks(years)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    _, labels = ax.get_legend_handles_labels()
    active_labels = [label for label in labels if label and not label.startswith("_")]
    if active_labels:
        ax.legend(loc="upper left")

    # Layout
    ax.figure.tight_layout()


def plot_lattice(
    lattice: Lattice | SeaLevelRiseModel,
    *,
    title: str = "Direct Fit Lattice",
    label: str = "",
    color: str = "dimgrey",
    alpha: float = 1.0,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the lattice: its states and one line per transition.

    Args:
        lattice: Lattice, or a SeaLevelRiseModel (its `.lattice` is used).
        title, label, color, alpha: Title, legend label and line style.
        xlim, ylim: Axis limits (years, sea level rise [m]).
        ax: Axes to draw on; a new figure is created if None.
    """
    lattice = _as_lattice(lattice)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    rel_time_steps, start_year = lattice.time_steps()
    first_edge = True

    for t, t_adjacent in lattice.iter_steps():
        yr = start_year + t
        yr_adjacent = start_year + t_adjacent
        if xlim is not None and (yr < xlim[0] or yr > xlim[1]):
            continue  # skip time steps outside the specified x-axis limits

        for state in lattice.states(t):
            for _, adjacent_state in lattice.get_transition_probs(t, state, None):
                ax.plot(
                    [yr, yr_adjacent],
                    [state * 100, adjacent_state * 100],  # Convert m to cm for plotting
                    color=color,
                    linewidth=1.2,
                    marker="o",
                    markersize=3,
                    alpha=alpha,
                    label=label if first_edge else "",
                )
                first_edge = False

    # Style
    _set_sea_level_rise_plot_style(ax, title, start_year, rel_time_steps, xlim, ylim)

    return fig


def plot_quantile_fan(
    quantile_fn: Callable[[int, float], float],
    rel_time_steps: list[int],
    start_year: int,
    quantile_fn_2: Callable[[int, float], float] | None = None,
    *,
    name: str = "Quantile Function 1",
    name_2: str = "Quantile Function 2",
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots a fan chart of a quantile function, optionally next to a second one.

    Args:
        quantile_fn: Quantile function quantile_fn(t, prob) [m], t in years after start_year.
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        quantile_fn_2: Second quantile function to compare with.
        name, name_2: Legend labels.
        ax: Axes to draw on; a new figure is created if None.
    """
    years = [start_year + t for t in rel_time_steps]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if quantile_fn_2 is None:
        title = name
        q50 = [100.0 * float(quantile_fn(t, 0.5)) for t in rel_time_steps]
        q05 = [100.0 * float(quantile_fn(t, 0.05)) for t in rel_time_steps]
        q95 = [100.0 * float(quantile_fn(t, 0.95)) for t in rel_time_steps]

        ax.plot(years, q50, color="darkgoldenrod", linewidth=3, label="Median")
        ax.fill_between(
            years,
            q05,
            q95,
            color="goldenrod",
            alpha=0.5,
            label="5% - 95% percentile",
        )

    else:
        title = f"{name} vs. {name_2}"

        # Source 1
        q50_1 = [100.0 * float(quantile_fn(t, 0.5)) for t in rel_time_steps]
        q05_1 = [100.0 * float(quantile_fn(t, 0.05)) for t in rel_time_steps]
        q95_1 = [100.0 * float(quantile_fn(t, 0.95)) for t in rel_time_steps]

        ax.plot(
            years,
            q50_1,
            color="darkgoldenrod",
            linewidth=3,
            label=f"{name} Median",
        )
        ax.fill_between(
            years,
            q05_1,
            q95_1,
            color="goldenrod",
            alpha=0.5,
            label=f"{name} 5% - 95% percentile",
        )

        # Source 2
        q50_2 = [100.0 * float(quantile_fn_2(t, 0.5)) for t in rel_time_steps]
        q05_2 = [100.0 * float(quantile_fn_2(t, 0.05)) for t in rel_time_steps]
        q95_2 = [100.0 * float(quantile_fn_2(t, 0.95)) for t in rel_time_steps]

        ax.plot(
            years,
            q50_2,
            color="tab:blue",
            linewidth=3,
            label=f"{name_2} Median",
        )
        ax.fill_between(
            years,
            q05_2,
            q95_2,
            color="tab:blue",
            alpha=0.5,
            label=f"{name_2} 5% - 95% percentile",
        )

    # Improved tick formatting
    ax.set_xticks([start_year + t for t in rel_time_steps])
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.2)

    # Improve spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Sea level rise [cm]", fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)

    _, labels = ax.get_legend_handles_labels()
    active_labels = [label for label in labels if label and not label.startswith("_")]
    if active_labels:
        ax.legend(loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_static_boxplots(
    quantile_fn: Callable[[int, float], float],
    rel_time_steps: list[int],
    start_year: int,
    *,
    title: str = "",
    label: str = "Static scenario",
    color: str = load_palette("Acanthurus_olivaceus")[0],
    alpha: float = 1.0,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots a boxplot of the quantile function per time step (the static scenario).

    Args:
        quantile_fn: Quantile function quantile_fn(t, prob) [m], t in years after start_year.
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        title, label, color, alpha: Title, legend label and style.
        ax: Axes to draw on; a new figure is created if None.
    """
    years = [start_year + t for t in rel_time_steps]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if len(years) < 2:
        step = 1.0
    else:
        step = float(
            min(abs(b - a) for a, b in itertools.pairwise(years) if b != a) or 1.0
        )

    cap_width = step * 0.1
    box_width = step * 0.3

    first = True
    for year, t in zip(years, rel_time_steps, strict=True):
        q01 = 100.0 * float(quantile_fn(t, 0.01))
        q25 = 100.0 * float(quantile_fn(t, 0.25))
        q50 = 100.0 * float(quantile_fn(t, 0.5))
        q75 = 100.0 * float(quantile_fn(t, 0.75))
        q99 = 100.0 * float(quantile_fn(t, 0.99))
        _plot_box_like(
            ax,
            float(year),
            q01,
            q25,
            q50,
            q75,
            q99,
            color=color,
            label=label if first else "",
            cap_width=cap_width,
            box_width=box_width,
            alpha=alpha,
        )
        first = False

    # Improved tick formatting
    ax.set_xticks([start_year + t for t in rel_time_steps])
    ax.tick_params(axis="both", which="major")

    # Improve spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")
    ax.set_title(title)

    _, labels = ax.get_legend_handles_labels()
    active_labels = [label for label in labels if label and not label.startswith("_")]
    if active_labels:
        ax.legend(loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_lattice_and_boxplots(
    model: SeaLevelRiseModel,
    *,
    title: str = "Learning vs Static Scenario Lattice and Boxplots",
    label: str = "Learning scenario",
    alpha: float = 1.0,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the lattice of `model` on top of the boxplots of its quantile function.

    Args:
        model: Sea level rise model.
        title, label, alpha: Title, legend label and transparency.
        xlim, ylim: Axis limits (years, sea level rise [m]).
        ax: Axes to draw on; a new figure is created if None.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    rel_time_steps, start_year = model.lattice.time_steps()

    plot_static_boxplots(
        model.quantile_fn,
        rel_time_steps,
        start_year,
        label="Static scenario",
        color=load_palette("Acanthurus_olivaceus")[0],
        alpha=1.0,
        ax=ax,
    )

    plot_lattice(
        model.lattice,
        title=title,
        label=label,
        color="dimgrey",
        alpha=alpha,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique_by_label = {}
    for handle, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in unique_by_label:
            unique_by_label[lab] = handle

    if unique_by_label:
        ax.legend(unique_by_label.values(), unique_by_label.keys(), loc="upper left")

    return fig


def plot_boxplot_comparison(
    quantile_fn_1: Callable[[int, float], float],
    quantile_fn_2: Callable[[int, float], float],
    rel_time_steps: list[int],
    start_year: int,
    *,
    name_1: str = "Quantile Function 1",
    name_2: str = "Quantile Function 2",
    color_1: str = load_palette("bucks_city2")[0],
    color_2: str = load_palette("bucks_city2")[5],
    title: str | None = None,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the boxplots of two quantile functions side by side per time step.

    Args:
        quantile_fn_1, quantile_fn_2: Quantile functions quantile_fn(t, prob) [m].
        rel_time_steps: Time steps t [yr after start_year], in order.
        start_year: Calendar year of t = 0.
        name_1, name_2, color_1, color_2: Legend labels and colors.
        title: Title; generated from the names if None.
        ax: Axes to draw on; a new figure is created if None.
    """
    years = [start_year + t for t in rel_time_steps]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if len(years) < 2:
        step = 1.0
    else:
        step = float(
            min(abs(b - a) for a, b in itertools.pairwise(years) if b != a) or 1.0
        )

    cap_width = step * 0.1
    box_width = step * 0.2
    offset = box_width * 1.1
    label_1 = name_1 or "Scenario 1"
    label_2 = name_2 or "Scenario 2"

    first_1 = True
    first_2 = True
    for year, t in zip(years, rel_time_steps, strict=True):
        box_1 = [
            100.0 * float(quantile_fn_1(t, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
        ]
        box_2 = [
            100.0 * float(quantile_fn_2(t, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
        ]
        _plot_box_like(
            ax,
            float(year) - offset,
            *box_1,
            color=color_1,
            label=label_1 if first_1 else "",
            cap_width=cap_width,
            box_width=box_width,
            alpha=1.0,
        )
        _plot_box_like(
            ax,
            float(year) + offset,
            *box_2,
            color=color_2,
            label=label_2 if first_2 else "",
            cap_width=cap_width,
            box_width=box_width,
            alpha=1.0,
        )
        first_1 = False
        first_2 = False

    # Improved tick formatting
    ax.set_xticks([start_year + t for t in rel_time_steps])
    ax.tick_params(axis="both", which="major")

    # Improve spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")
    ax.set_title(title if title is not None else f"{label_1} vs. {label_2}")  # fmt: skip

    handles, labels = ax.get_legend_handles_labels()
    unique_by_label = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique_by_label:
            unique_by_label[label] = handle

    if unique_by_label:
        ax.legend(unique_by_label.values(), unique_by_label.keys(), loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_lattice_boxplot_comparison(
    model: SeaLevelRiseModel,
    *,
    title: str = "Learning vs Static Scenario Boxplot Comparison",
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the boxplots of the lattice distribution of `model` and of its quantile function side
    by side per time step.
    """
    quantile_fn_static = model.quantile_fn
    quantile_fn_learning = model.lattice.get_marginal_quantile_fn()
    rel_time_steps, start_year = model.lattice.time_steps()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    return plot_boxplot_comparison(
        quantile_fn_static,
        quantile_fn_learning,
        rel_time_steps=rel_time_steps,
        start_year=start_year,
        name_1="Static Scenario",
        name_2="Learning Scenario",
        title=title,
        ax=ax,
    )


def plot_conditioned_lattice(
    lattice: ForwardLattice | SeaLevelRiseModel,
    observed_year: int,
    observed_state: float,
    *,
    plot_distributions: bool = False,
    title: str = "Conditioned Lattice",
    color: str = "dodgerblue",
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots the lattice conditioned on an observation (see `ForwardLattice.propagate_observation`).

    Args:
        lattice: Lattice, or a SeaLevelRiseModel (its `.lattice` is used).
        observed_year: Time step of the observation [yr after start_year].
        observed_state: Observed sea level rise [m].
        plot_distributions: Add boxplots of the conditional distribution per time step.
        title, color: Title and color of the conditioned lattice.
        ax: Axes to draw on; a new figure is created if None.
    """
    lattice = _as_forward_lattice(lattice)
    forward_lattice, backward_lattice, closest_observed_state = lattice.propagate_observation(observed_year, observed_state)  # fmt:skip
    full_rel_time_steps, start_year = (
        lattice.time_steps()
    )  # the original, unconditioned lattice's own full horizon

    print(f"Closest observed state at observed year: {closest_observed_state * 100: .2f} cm")  # fmt: skip

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Optional: plot the conditional distributions as boxplots, on top of the conditional lattice
    if plot_distributions:
        forward_quantile_fn = forward_lattice.get_marginal_quantile_fn()
        backward_quantile_fn = backward_lattice.get_marginal_quantile_fn()

        def conditional_quantile_fn(t, prob):
            return (
                forward_quantile_fn(t, prob)
                if t >= observed_year
                else backward_quantile_fn(t, prob)
            )

        # Plot the conditional distributions as boxplots
        plot_static_boxplots(conditional_quantile_fn, full_rel_time_steps, start_year, label="Conditional distribution", color=load_palette("Acanthurus_olivaceus")[3], alpha=1.0, ax=ax)  # fmt: skip

    # Background unconditioned lattice in light gray
    plot_lattice(lattice, label="Unconditioned lattice", color="lightgray", alpha=0.55, ax=ax)  # fmt: skip

    # Plot the conditioned lattice paths starting from the observed state
    plot_lattice(forward_lattice, label="Conditioned lattice", color=color, alpha=0.75, ax=ax)  # fmt: skip
    plot_lattice(backward_lattice, label="", color=color, alpha=0.75, ax=ax)  # fmt: skip

    # Mark the discrete observation
    ax.plot(
        [start_year + observed_year],
        [closest_observed_state * 100],
        marker="o",
        markersize=8,
        color="red",
        label=f"Observed: {closest_observed_state * 100:.2f} cm",
    )

    # Improved tick formatting
    ax.set_xticks([start_year + t for t in full_rel_time_steps])
    ax.tick_params(axis="both", which="major")

    # Improve spine visibility
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    unique_by_label = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique_by_label:
            unique_by_label[label] = handle

    if unique_by_label:
        ax.legend(unique_by_label.values(), unique_by_label.keys(), loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_trajectory(
    lattice: Lattice | SeaLevelRiseModel,
    trajectory: list[float],
    *,
    show_lattice: bool = True,
    title: str = "Sample Trajectory",
    label: str = "Sample trajectory",
    color: str = "dodgerblue",
    alpha: float = 1.0,
    xlim: tuple[int, int] | None = None,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots one sampled trajectory.

    Args:
        lattice: Lattice, or a SeaLevelRiseModel (its `.lattice` is used).
        trajectory: States [m], one per time step.
        show_lattice: Draw the lattice in light gray behind the trajectory.
        title, label, color, alpha: Title, legend label and line style.
        xlim: x-axis limits (years).
        ax: Axes to draw on; a new figure is created if None.
    """
    lattice = _as_lattice(lattice)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    rel_time_steps, lattice_start_year = lattice.time_steps()
    years = [lattice_start_year + t for t in rel_time_steps]

    if xlim is not None:
        mask = (np.array(years) >= xlim[0]) & (np.array(years) <= xlim[1])
    else:
        mask = np.ones(len(years), dtype=bool)

    if show_lattice:
        plot_lattice(
            lattice,
            title=title,
            label="Sea level rise model",
            color="lightgrey",
            alpha=0.55,
            xlim=xlim,
            ax=ax,
        )

    ax.plot(
        np.array(years)[mask],
        np.array(trajectory)[mask] * 100.0,  # Convert to cm for plotting
        color=color,
        alpha=alpha,
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=label,
    )

    # Style
    _set_sea_level_rise_plot_style(ax, title, lattice_start_year, rel_time_steps, xlim)

    return fig


def plot_trajectories(
    lattice: Lattice | SeaLevelRiseModel,
    trajectories: list[list[float]],
    *,
    title: str = "Sampled Trajectories",
    label: str = "Sampled trajectories",
    color: str = "dodgerblue",
    alpha: float = 0.5,
    ax: plt.axes.Axes = None,
) -> plt.figure.Figure:
    """Plots several sampled trajectories.

    Args:
        lattice: Lattice, or a SeaLevelRiseModel (its `.lattice` is used).
        trajectories: Trajectories, each a list of states [m], one per time step.
        title, label, color, alpha: Title, legend label and line style.
        ax: Axes to draw on; a new figure is created if None.
    """
    lattice = _as_lattice(lattice)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    plot_lattice(
        lattice,
        title=title,
        label="Direct fit lattice",
        color="lightgrey",
        alpha=0.55,
        ax=ax,
    )

    rel_time_steps, start_year = lattice.time_steps()
    years = [start_year + t for t in rel_time_steps]

    for trajectory in trajectories:
        ax.plot(
            years,
            [state * 100 for state in trajectory],  # Convert to cm for plotting
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            alpha=alpha,
            label=label if label else "",
        )
        label = ""  # Only show the label for the first trajectory

    handles, labels = ax.get_legend_handles_labels()
    unique_by_label = {}
    for handle, lab in zip(handles, labels):
        if lab and not lab.startswith("_") and lab not in unique_by_label:
            unique_by_label[lab] = handle

    if unique_by_label:
        ax.legend(unique_by_label.values(), unique_by_label.keys(), loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Static plotting helpers (these are only used for demonstration purposes and are not part of the core functionality)
# ─────────────────────────────────────────────────────────────────────────────


def plot_data_as_static_boxplots(
    start_year: int,
    end_year: int,
    step: int,
    ssp: str,
    ax: plt.axes.Axes = None,
    extrapolate: bool = False,
) -> plt.figure.Figure:
    """Plots a boxplot of the KNMI projections of one SSP scenario per year.

    Args:
        start_year, end_year, step: Years shown.
        ssp: Scenario, "ssp126", "ssp245" or "ssp585".
        ax: Axes to draw on; a new figure is created if None.
        extrapolate: Use the data extrapolated to 2300 (`data.extrapolate_quantiles`) instead of the
            raw data.
    """

    data = load_data()
    if extrapolate:
        data = extrapolate_quantiles(data)

    if start_year not in data or end_year not in data:
        raise ValueError(f"Data for years {start_year} or {end_year} not found.")

    years = range(start_year, end_year + 1, step)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    cap_width = step * 0.1
    box_width = step * 0.3

    for year in years:
        q, p = data[year][ssp]
        q01 = 100.0 * float(q[p == 0.01][0])
        q17 = 100.0 * float(q[p == 0.17][0])
        q50 = 100.0 * float(q[p == 0.5][0])
        q83 = 100.0 * float(q[p == 0.83][0])
        q99 = 100.0 * float(q[p == 0.99][0])
        _plot_box_like(
            ax,
            float(year),
            q01,
            q17,
            q50,
            q83,
            q99,
            color=load_palette("Acanthurus_olivaceus")[0],
            label="",
            cap_width=cap_width,
            box_width=box_width,
            alpha=1.0,
        )

    # Improved tick formatting
    ax.set_xticks(years)
    ax.tick_params(axis="both", which="major")

    # Enhanced title and labels with larger fonts
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea level rise [cm]")
    ax.set_title(f"Sea Level Rise Projections as Point-wise Distributions: {ssp.upper()}")  # fmt:skip

    _, labels = ax.get_legend_handles_labels()
    active_labels = [label for label in labels if label and not label.startswith("_")]
    if active_labels:
        ax.legend(loc="upper left")

    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_projections_as_seen_from_future(
    start_year: int,
    end_year: int,
    step: int = 10,
    method: str = "mixture",
    distribution: str | None = None,
    weights: list[float] | None = (0.25, 0.5, 0.25),  # type: ignore
    observed_year: int | None = None,
    observed_state: float | None = None,
    extrapolate: bool = False,
) -> Figure | SubFigure:
    """Illustrates learning: a direct fit lattice conditioned on an observation (see
    `plot_conditioned_lattice`).

    Args:
        start_year, step, method, distribution, weights, extrapolate: Sea level rise model, see
            `SeaLevelRiseModel`.
        end_year: Last year of the lattice.
        observed_year: Time step of the observation [yr after start_year]; defaults to the middle.
        observed_state: Observed sea level rise [m]; defaults to the middle state.
    """
    model = DirectFitSeaLevelRiseModel(
        start_year=start_year,
        end_year=end_year,
        step=step,
        method=method,
        distribution=distribution,
        weights=weights,
        extrapolate=extrapolate,
        align_to_lattice_midpoint=False,
        p=0.5,
        skip=False,
        shift=0,
    )
    lattice = model.lattice

    if observed_year is None or observed_state is None:
        middle_year = (start_year + end_year) // 2 - start_year
        observed_year = middle_year // step * step
        observed_state = lattice.states(observed_year)[
            len(lattice.states(observed_year)) // 2
        ]  # type: ignore

    return plot_conditioned_lattice(
        model,
        observed_year=observed_year,
        observed_state=observed_state,
        plot_distributions=True,
        title="Sea Level Rise Projections as Seen from the Future",
    )


def plot_direct_fit_construction(model: DirectFitSeaLevelRiseModel) -> Figure:
    """Illustrates the direct fit method on a small lattice (thesis Section 4.3), two panels with a
    shared y-axis. Left: the lattice, with its states labeled. Right: the quantile function at the
    last time step, the lattice's own (discrete) quantile function, and the quantiles at the
    midpoints of the probability intervals, which are the states of the last time step.

    Args:
        model: Direct fit sea level rise model with a single root and few (about three) time steps,
            e.g. with end_year = start_year + 2 * step.
    """
    lattice = model.lattice
    rel_time_steps, start_year = lattice.time_steps()
    t_last = rel_time_steps[-1]
    year_last = start_year + t_last
    m = len(rel_time_steps) - 1  # Binomial order at the last time step
    midpoints = [_binomial_midpoint_prob(j, m, model.p) for j in range(m + 1)]
    values = [model.quantile_fn(t_last, p) * 100 for p in midpoints]  # m -> cm
    marginal = lattice.get_marginal_probs()[t_last]
    bounds = [0.0, *np.cumsum([marginal[s] for s in sorted(marginal)]).tolist()]

    fig, (ax_lattice, ax_q) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left: the lattice with labeled states
    plot_lattice(lattice, title="Direct fit lattice", ax=ax_lattice)
    for t in rel_time_steps:
        for k, state in enumerate(sorted(lattice.states(t))):
            ax_lattice.annotate(
                rf"$\theta_{{{start_year + t}}}^{{{k}}}$",
                (start_year + t, state * 100),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    years = [start_year + t for t in rel_time_steps]
    ax_lattice.set_xlim(years[0] - 3, years[-1] + 3)

    # Right: the quantile function and the lattice's quantile function
    probs = np.linspace(0.005, 0.995, 199)
    ax_q.plot(probs, [model.quantile_fn(t_last, float(p)) * 100 for p in probs], color="dodgerblue", label=f"Quantile function $Q_{{{year_last}}}$")  # fmt: skip
    ax_q.step(bounds, [values[0], *values], where="pre", color="dimgrey", label="Lattice quantile function")  # fmt: skip
    ax_q.plot(midpoints, values, color="dodgerblue", marker="o", linestyle="none", zorder=5, label="Quantile at interval midpoint")  # fmt: skip
    for bound in bounds[1:-1]:
        ax_q.axvline(bound, color="0.75", linewidth=0.8, linestyle="--", zorder=0)
    for midpoint, value in zip(midpoints, values, strict=True):  # Link every state to its quantile
        guide = {"color": "0.6", "linewidth": 0.8, "linestyle": ":", "zorder": 1}
        ax_lattice.plot([year_last, years[-1] + 3], [value, value], **guide)
        ax_q.plot([0, midpoint], [value, value], **guide)
    ax_q.set_xlim(0, 1)
    ax_q.set_xticks(sorted({*bounds, 0.5}))
    ax_q.set_xlabel("Probability level")
    ax_q.set_title(f"Quantiles in {year_last}")
    ax_q.tick_params(axis="y", labelleft=False)
    ax_q.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax_q.legend(loc="upper left")
    fig.tight_layout()
    return fig
