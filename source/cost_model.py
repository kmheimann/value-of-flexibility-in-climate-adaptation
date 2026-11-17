"""
cost_model.py — Cost of upgrading a dike (thesis Chapter 5).

A cost model prices every upgrade of a dike, from its capacity h_{i-1} before to h_i after the
upgrade. `decision_analysis.backward_induction` calls it for every candidate decision.

Cost models:
    RetentionCostModel  Retention flexibility model (thesis Sec. 5.2):
                        c_a(h_{i-1}, h_i) = c + v * (h_i - phi * h_{i-1}).
                        The retention flexibility phi (thesis: \\tilde{\\varphi}) is chosen at the
                        first decision.
    LandCostModel       Land availability model (thesis Sec. 5.3): wraps a RetentionCostModel (with
                        a fixed phi) and adds the cost of the land under the growing dike footprint.
                        The secured height h_max is chosen at the first decision.

Flexibility state:
    Each cost model has a `flexibility_spec` (a `FlexibilitySpec`) that tells `backward_induction`
    which flexibility values exist and how they evolve: `RetentionSpec` for phi, `HeightReserveSpec`
    for h_max. `backward_induction` carries the flexibility state as a tuple next to the dike height.

Land geometry (used by LandCostModel):
    w_footprint, open_land_width, land_acquisition_cost, available_crest_height.

Plots:
    plot_cost_initial_vs_h0, plot_cost_initial_vs_flexibility,
    plot_cost_adapt_vs_hiplus1, plot_cost_adapt_vs_flexibility, plot_open_land_width.

Units: heights and widths in m, time in years, costs in €. Plots show heights in cm and costs in M€.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

EUR_PER_MEUR = 1e6  # Display unit of costs in plots [€ per M€]


class CostModel(ABC):
    """Abstract cost model: prices the first decision and every later upgrade of a dike.

    Every height is absolute, i.e. measured from the ground (the existing dike has capacity h_minus_1).
    `f_*` arguments are flexibility-state tuples (see `FlexibilitySpec`); every spec in this module has
    a single component, so each cost function takes four positional arguments.

    Attributes:
        time_aware: If True, `backward_induction` also passes the keyword-only arguments t_i and
            t_i_plus_1 (years since the first decision t_0) to `adapt_cost_fn`.
    """

    time_aware: ClassVar[bool] = False

    @property
    @abstractmethod
    def flexibility_spec(self) -> "FlexibilitySpec":
        """The flexibility state this cost model expects (see `FlexibilitySpec`)."""

    @abstractmethod
    def initial_cost_fn(
        self, h_minus_1: float, f_minus_1: float, h_0: float, f_0: float
    ) -> float:
        """Cost of the first decision [€]: upgrading the existing dike (h_minus_1, f_minus_1) [m] to
        (h_0, f_0) [m]."""

    @abstractmethod
    def adapt_cost_fn(
        self, h_i: float, f_i: float, h_i_plus_1: float, f_i_plus_1: float
    ) -> float:
        """Cost of a later upgrade [€] from (h_i, f_i) to (h_i_plus_1, f_i_plus_1), heights in m.

        Note that this function uses the "before/after" index pair (i, i+1), while the thesis and
        `backward_induction` write the same upgrade equivalently as (i-1, i).
        """


class FlexibilitySpec(ABC):
    """Describes the flexibility state that `backward_induction` carries next to the dike height.

    `backward_induction` does not know what the state means. It only asks this object which states
    exist (`state_candidates`), which can be chosen (`action_candidates`) and which designs count as
    inflexible (`design_masks`).

    Attributes:
        components: Names of the state's components, e.g. ("phi",) or ("h_max",).
        height_like: Per component, whether it is a height [m] (shown in cm in plots).
        labels: Per component, the thesis symbol as a matplotlib LaTeX string.
    """

    components: ClassVar[tuple[str, ...]]
    height_like: ClassVar[tuple[bool, ...]]
    labels: ClassVar[tuple[str, ...]]

    @abstractmethod
    def initial_state(self, h_minus_1: float) -> tuple[float, ...]:
        """Flexibility state of the existing dike with capacity h_minus_1 [m]."""

    @abstractmethod
    def state_candidates(
        self, h_i_minus_1: float, h_alternatives: list[float]
    ) -> list[tuple[float, ...]]:
        """Every flexibility state a dike of height h_i_minus_1 [m] can have after the first decision.
        `h_alternatives` is the height grid [m]."""

    @abstractmethod
    def action_candidates(
        self,
        h_i_minus_1: float,
        f_i_minus_1: tuple[float, ...],
        h_i: float,
        h_alternatives: list[float],
        initial: bool,
    ) -> list[tuple[float, ...]]:
        """Every flexibility state that can follow the state (h_i_minus_1, f_i_minus_1) when the dike
        is upgraded to h_i [m]. `initial` is True at the first decision, the only one at which the
        flexibility is chosen freely."""

    def priced_state(
        self, h_i: float, f_i: tuple[float, ...], h_minus_1: float
    ) -> tuple[float, ...]:
        """Flexibility state of the structure that is actually standing at height h_i [m], i.e. the
        "before" state that an upgrade is priced with. By default the state `f_i` itself."""
        return f_i

    @abstractmethod
    def resolve_flexibility_state(
        self, base_height: float, value: float
    ) -> tuple[float, ...]:
        """Turns one plot value into a flexibility state (plots only). For a height-like state the
        value is an offset above `base_height` [m]; otherwise it is the state itself. The value 0 is
        always the inflexible design."""

    @abstractmethod
    def default_plot_values(self) -> list[float]:
        """Default flexibility values drawn as separate curves in plots."""

    @abstractmethod
    def default_plot_range(self) -> list[float]:
        """Default flexibility values on a plot's x-axis."""

    @abstractmethod
    def design_masks(self, t0: pd.DataFrame) -> dict[str, pd.Series]:
        """Boolean masks over the first-decision designs (`RewardsTable.t0` after `.reset_index()`)
        for the "inflexible" and "flexible" design classes.

        The classes are nested: the inflexible class holds the designs without flexibility, the
        flexible class holds every design. So the flexible design is the overall optimum and the value
        of flexibility is never negative. Waiting (deferral; h_0 = h_minus_1) belongs to both classes.
        """


class HeightReserveSpec(FlexibilitySpec):
    """Flexibility as a secured height h_max [m] (land availability model).

    h_max is chosen freely at the first decision (h_max_0 >= h_0, also without upgrading). Afterwards
    it follows the dike: h_max_i = max(h_max_{i-1}, h_i).
    """

    components: ClassVar[tuple[str, ...]] = ("h_max",)
    height_like: ClassVar[tuple[bool, ...]] = (True,)
    labels: ClassVar[tuple[str, ...]] = (r"$h_\mathrm{max}$",)

    def initial_state(self, h_minus_1: float) -> tuple[float, ...]:
        return (h_minus_1,)

    def state_candidates(
        self, h_i_minus_1: float, h_alternatives: list[float]
    ) -> list[tuple[float, ...]]:
        return [(h_max,) for h_max in h_alternatives if h_max >= h_i_minus_1]

    def action_candidates(
        self, h_i_minus_1, f_i_minus_1, h_i, h_alternatives, initial
    ) -> list[tuple[float, ...]]:
        (h_max_i_minus_1,) = f_i_minus_1
        if initial:
            return [(h_max,) for h_max in h_alternatives if h_max >= max(h_max_i_minus_1, h_i)]  # fmt: skip
        return [(max(h_max_i_minus_1, h_i),)]

    def resolve_flexibility_state(
        self, base_height: float, value: float
    ) -> tuple[float, ...]:
        return (base_height + value,)

    def default_plot_values(self) -> list[float]:
        return [0.0, 0.5, 1.0, 1.5]

    def default_plot_range(self) -> list[float]:
        return np.linspace(0.0, 3.0, 100).tolist()

    def design_masks(self, t0: pd.DataFrame) -> dict[str, pd.Series]:
        # Inflexible: no reservation beyond the dike itself
        return {"inflexible": t0["h_max_0"] == t0["h_0"], "flexible": t0["h_max_0"] >= t0["h_0"]}  # fmt: skip


class RetentionSpec(FlexibilitySpec):
    """Flexibility as a retention fraction phi [-] (retention flexibility model).

    phi is chosen at the first decision and then held constant. A design that waits at the first
    decision (h_0 = h_minus_1) also chooses phi; it applies from the first upgrade on. Until then the
    existing dike is standing, so the first upgrade is priced with phi_minus_1 (`priced_state`).

    Args:
        phi_values: Grid of retention values [-], each in [0, 1]. Must contain 0 (the inflexible
            design) and phi_minus_1.
        phi_minus_1: Retention of the existing dike [-].
    """

    components: ClassVar[tuple[str, ...]] = ("phi",)
    height_like: ClassVar[tuple[bool, ...]] = (False,)
    labels: ClassVar[tuple[str, ...]] = (r"$\tilde{\varphi}$",)

    def __init__(self, phi_values: list[float], phi_minus_1: float = 0.0) -> None:
        phi_values = [
            float(p) for p in phi_values
        ]  # Avoids np.float64 in states and results
        phi_minus_1 = float(phi_minus_1)
        if not all(0.0 <= p <= 1.0 for p in phi_values):
            raise ValueError("Every value in phi_values must be between 0 and 1.")
        if phi_minus_1 not in phi_values:
            raise ValueError("phi_minus_1 must be one of phi_values.")
        self.phi_values = phi_values
        self.phi_minus_1 = phi_minus_1

    def initial_state(self, h_minus_1: float) -> tuple[float, ...]:
        return (self.phi_minus_1,)

    def state_candidates(
        self, h_i_minus_1: float, h_alternatives: list[float]
    ) -> list[tuple[float, ...]]:
        return [(phi,) for phi in self.phi_values]

    def action_candidates(
        self, h_i_minus_1, f_i_minus_1, h_i, h_alternatives, initial
    ) -> list[tuple[float, ...]]:
        if initial:
            return [(phi,) for phi in self.phi_values]
        return [f_i_minus_1]

    def priced_state(
        self, h_i: float, f_i: tuple[float, ...], h_minus_1: float
    ) -> tuple[float, ...]:
        # Heights never decrease, so h_i = h_minus_1 means the existing dike is still standing
        return (self.phi_minus_1,) if h_i == h_minus_1 else f_i

    def resolve_flexibility_state(
        self, base_height: float, value: float
    ) -> tuple[float, ...]:
        return (value,)

    def default_plot_values(self) -> list[float]:
        return [0.0, 0.25, 0.5, 0.75, 1.0]

    def default_plot_range(self) -> list[float]:
        return np.linspace(0.0, 1.0, 100).tolist()

    def design_masks(self, t0: pd.DataFrame) -> dict[str, pd.Series]:
        # Inflexible: no retention
        return {"inflexible": t0["phi_0"] == 0.0, "flexible": t0["phi_0"] >= 0.0}


# ─── Land geometry (thesis Sec. 5.3) ────────────────────────────────────────────


def w_footprint(delta_h: float, kappa_1: float, kappa_2: float) -> float:
    """Land-side footprint width of a dike raised by delta_h above the existing dike [m]:
    w(delta_h) = kappa_1 * delta_h + kappa_2 * delta_h**2.

    Args:
        delta_h: Height above the existing dike [m]; negative values count as 0.
        kappa_1: Linear dike width coefficient (slope) [-].
        kappa_2: Quadratic dike width coefficient (additional width for macro-stability and piping) [1/m].
    """
    delta_h = max(0.0, delta_h)
    return kappa_1 * delta_h + kappa_2 * delta_h**2


def _invert_w_footprint(width: float, kappa_1: float, kappa_2: float) -> float:
    """Inverse of `w_footprint`: the height increase [m] whose footprint is `width` [m] (0 for
    width <= 0)."""
    if width <= 0:
        return 0.0
    if kappa_2 == 0:
        return width / kappa_1
    return (-kappa_1 + (kappa_1**2 + 4 * kappa_2 * width) ** 0.5) / (2 * kappa_2)


def open_land_width(
    h_max_i: float,
    tau_i: float,
    h_ref: float,
    W_available_0: float,
    lambda_enc: float,
    kappa_1: float,
    kappa_2: float,
) -> float:
    """Open land width that is still available [m]:

        W_available_i = max(0, W_available_0 - w(h_max_i - h_ref) - lambda_enc * tau_i)

    Args:
        h_max_i: Secured height [m].
        tau_i: Years since the first decision t_0 [yr].
        h_ref: Capacity of the existing dike h_minus_1 [m].
        W_available_0: Initial open land width [m].
        lambda_enc: Land encroachment rate [m/yr].
        kappa_1, kappa_2: Footprint coefficients, see `w_footprint`.

    Remarks:
        - Valid only for a non-decreasing h_max, which `backward_induction` guarantees (dikes are
          never lowered).
    """
    return max(0.0, W_available_0 - w_footprint(h_max_i - h_ref, kappa_1, kappa_2) - lambda_enc * tau_i)  # fmt: skip


def land_acquisition_cost(
    land_width: float, open_width: float, C_open: float, C_built: float
) -> float:
    """Cost of buying `land_width` [m] of land [€]: first from the `open_width` [m] of open land at
    C_open [€/m], the rest from built-up land at C_built [€/m]."""
    return C_open * min(land_width, open_width) + C_built * max(0.0, land_width - open_width)  # fmt: skip


def available_crest_height(
    h_max_i: float,
    tau_i: float,
    h_ref: float,
    W_available_0: float,
    lambda_enc: float,
    kappa_1: float,
    kappa_2: float,
) -> float:
    """Highest crest height that can still be built on secured or open land [m] (plots only):

        h_available_i = max(h_max_i, h_ref + w^-1(W_available_0 - lambda_enc * tau_i))

    It decreases over time as open land is built up, but never below the secured height h_max_i.
    Arguments as in `open_land_width`.
    """
    h_ceiling = h_ref + _invert_w_footprint(W_available_0 - lambda_enc * tau_i, kappa_1, kappa_2)  # fmt: skip
    return max(h_max_i, h_ceiling)


# ─── Cost models ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetentionCostModel(CostModel):
    """Retention flexibility model (thesis Sec. 5.2):

        c_a(h_{i-1}, h_i) = c + v * (h_i - phi * h_{i-1})   if h_i > h_{i-1}, else 0.

    The retention phi (thesis: \\tilde{\\varphi}) is the part of the existing structure that is kept in
    an upgrade: phi = 0 rebuilds the whole dike (inflexible), phi = 1 pays only for the added height.
    phi is not a parameter but a decision: `backward_induction` chooses it from `phi_values` at the
    first decision. The first upgrade acts on the existing dike and is priced with phi_minus_1.

    Args:
        c: Fixed upgrade cost [€].
        v: Variable upgrade cost [€/m].
        phi_minus_1: Retention of the existing dike [-].
        phi_values: Grid of retention values for the new dike [-]; must contain 0 and phi_minus_1.
    """

    c: float
    v: float
    phi_minus_1: float = 0.0
    phi_values: tuple[float, ...] = tuple(float(phi) for phi in np.linspace(0, 1, 21))

    def __post_init__(self) -> None:
        _ = self.flexibility_spec  # Validates phi_values and phi_minus_1

    @property
    def flexibility_spec(self) -> FlexibilitySpec:
        return RetentionSpec(phi_values=list(self.phi_values), phi_minus_1=self.phi_minus_1)  # fmt: skip

    def _c_generic(
        self, h_i: float, phi_i: float, h_i_plus_1: float, phi_i_plus_1: float
    ) -> float:
        """Upgrade cost [€] from height h_i [m] with retention phi_i [-] to h_i_plus_1 [m].
        phi_i_plus_1 is not used: only the retention of the upgraded structure matters."""
        if not h_i_plus_1 >= h_i:
            raise ValueError("h_i_plus_1 must be greater than or equal to h_i.")
        if h_i < 0:
            raise ValueError("h_i must be non-negative.")
        if not (0.0 <= phi_i <= 1.0 and 0.0 <= phi_i_plus_1 <= 1.0):
            raise ValueError("phi_i and phi_i_plus_1 must be between 0 and 1.")
        if h_i_plus_1 == h_i:
            return 0.0
        return self.c + self.v * (h_i_plus_1 - phi_i * h_i)

    def initial_cost_fn(
        self, h_minus_1: float, phi_minus_1: float, h_0: float, phi_0: float
    ) -> float:
        """Cost of the first decision [€], see `_c_generic`."""
        return self._c_generic(h_minus_1, phi_minus_1, h_0, phi_0)

    def adapt_cost_fn(
        self, h_i: float, phi_i: float, h_i_plus_1: float, phi_i_plus_1: float
    ) -> float:
        """Cost of a later upgrade [€], see `_c_generic`."""
        return self._c_generic(h_i, phi_i, h_i_plus_1, phi_i_plus_1)


@dataclass(frozen=True)
class LandCostModel(CostModel):
    """Land availability model (thesis Sec. 5.3): the cost of the wrapped retention
    model `inner` plus the cost of the land under the growing dike footprint.

    Flexibility is the secured height h_max: the land for the footprint up to h_max is reserved by a
    legal zone at the first decision. Reserving open land is free; it stays open (it is exempt from
    encroachment) and is bought at C_open once the dike grows onto it. Unreserved land is bought from
    the remaining open land at C_open, and beyond it from built-up land at C_built. Open land shrinks
    over time at the encroachment rate lambda_enc.

    Args:
        inner: Retention cost model whose phi is fixed to one value, i.e. a
            `RetentionCostModel` with `phi_values=(phi_minus_1,)`. Land reservation is then
            the only flexibility.
        W_available_0: Initial open land width [m].
        lambda_enc: Land encroachment rate [m/yr].
        C_open: Cost of open land [€/m].
        C_built: Cost of built-up land [€/m].
        kappa_1: Linear dike width coefficient [-].
        kappa_2: Quadratic dike width coefficient [1/m].
        h_ref: Capacity of the existing dike [m]; must equal `backward_kwargs["h_minus_1"]`.

    Remarks:
        - Land costs are per meter of land width along the whole dike section.
        - h_max may also be raised beyond the open land; that land is then bought at C_built when it
          is reserved (and charged C_open again when used). This option is more expensive than buying
          the land when needed, so it is never optimal and does not affect the results (the thesis
          caps h_max_0 at the open land instead).
    """

    time_aware: ClassVar[bool] = True

    inner: CostModel
    W_available_0: float
    lambda_enc: float
    C_open: float
    C_built: float
    kappa_1: float
    kappa_2: float
    h_ref: float

    def __post_init__(self) -> None:
        if (
            len(self.inner.flexibility_spec.state_candidates(self.h_ref, [self.h_ref]))
            != 1
        ):
            raise ValueError(
                "LandCostModel requires `inner` to have a single flexibility value, e.g. "
                "RetentionCostModel(..., phi_values=(phi_minus_1,))."
            )

    @property
    def flexibility_spec(self) -> FlexibilitySpec:
        return HeightReserveSpec()

    def _f_inner(self) -> tuple[float, ...]:
        """The fixed flexibility state of `inner`."""
        return self.inner.flexibility_spec.initial_state(self.h_ref)

    def _c_land(
        self,
        h_i: float,
        h_max_i: float,
        h_i_plus_1: float,
        h_max_i_plus_1: float,
        tau: float,
    ) -> float:
        """Land cost [€] of the transition (h_i, h_max_i) -> (h_i_plus_1, h_max_i_plus_1), tau years
        after t_0, with three parts:

            1. W_res: reserved land the dike grows onto, bought at C_open.
            2. W_new: unreserved land the dike grows onto, bought from the open land at C_open and
               beyond it at C_built.
            3. Land newly reserved beyond the remaining open land, bought at C_built (not in the
               thesis; never optimal, see the class docstring).
        """

        def w(h: float) -> float:
            return w_footprint(h - self.h_ref, self.kappa_1, self.kappa_2)

        open_width = open_land_width(h_max_i, tau, self.h_ref, self.W_available_0, self.lambda_enc, self.kappa_1, self.kappa_2)  # fmt: skip
        w_res = max(0.0, w(min(h_i_plus_1, h_max_i)) - w(h_i))
        w_new = max(0.0, w(h_i_plus_1) - w(max(h_i, h_max_i)))
        w_reserved = max(0.0, w(h_max_i_plus_1) - w(max(h_i_plus_1, h_max_i)))
        open_left = max(0.0, open_width - w_new)
        return (
            self.C_open * w_res
            + land_acquisition_cost(w_new, open_width, self.C_open, self.C_built)
            + self.C_built * max(0.0, w_reserved - open_left)
        )

    def initial_cost_fn(
        self, h_minus_1: float, h_max_minus_1: float, h_0: float, h_max_0: float
    ) -> float:
        """Cost of the first decision [€]: the dike cost of `inner` plus the land cost of building h_0
        and reserving up to h_max_0 (heights in m)."""
        f_inner = self._f_inner()
        dike_cost = self.inner.initial_cost_fn(h_minus_1, *f_inner, h_0, *f_inner)  # type: ignore
        return dike_cost + self._c_land(h_minus_1, h_max_minus_1, h_0, h_max_0, 0.0)

    def adapt_cost_fn(
        self,
        h_i: float,
        h_max_i: float,
        h_i_plus_1: float,
        h_max_i_plus_1: float,
        *,
        t_i: float = 0.0,
        t_i_plus_1: float | None = None,
    ) -> float:
        """Cost of a later upgrade [€]: the dike cost of `inner` plus the land cost (heights in m).

        Args:
            t_i: Years since the first decision t_0 at the time of the upgrade [yr].
            t_i_plus_1: Not used; accepted because `backward_induction` passes it to every
                time-aware cost model.
        """
        f_inner = self._f_inner()
        dike_cost = self.inner.adapt_cost_fn(h_i, *f_inner, h_i_plus_1, *f_inner)  # type: ignore
        return dike_cost + self._c_land(h_i, h_max_i, h_i_plus_1, h_max_i_plus_1, t_i)


# ─── Plots ──────────────────────────────────────────────────────────────────────


def _flex_curve_label(spec: FlexibilitySpec, base_symbol: str, value: float) -> str:
    """Legend label of a curve at the flexibility plot value `value` (0 = inflexible design)."""
    symbol = spec.labels[0].strip("$")
    if value == 0.0:
        reference = base_symbol if spec.height_like[0] else "0"
        return f"Inflexible dike (${symbol} = {reference}$)"
    if spec.height_like[0]:
        return f"${symbol} = {base_symbol} + {value * 100:.0f}$ cm"
    return f"${symbol} = {value:.2f}$"


def _auto_updated_flex_state(
    spec: FlexibilitySpec, f_i: tuple[float, ...], h_i: float, h_i_plus_1: float
) -> tuple[float, ...]:
    """Flexibility state after a later upgrade from h_i to h_i_plus_1 [m], as in `backward_induction`."""
    (candidate,) = spec.action_candidates(h_i, f_i, h_i_plus_1, [], initial=False)
    return candidate


def _plot_time_kwargs(cost_model: CostModel, t_i: float | None) -> dict:
    """Keyword arguments t_i/t_i_plus_1 [yr] for `adapt_cost_fn` of a time-aware cost model."""
    if not cost_model.time_aware or t_i is None:
        return {}
    return {"t_i": t_i, "t_i_plus_1": t_i}


def _new_axes(ax: Axes | None) -> tuple[Figure | SubFigure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        return fig, ax
    return ax.figure, ax


def plot_cost_initial_vs_h0(
    cost_model: CostModel,
    h_0_range: np.ndarray | None = None,
    flex_values: list[float] | None = None,
    h_minus_1: float = 3.0,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the cost of the first decision against h_0, one curve per flexibility value.

    Args:
        cost_model: Cost model to evaluate.
        h_0_range: Values of h_0 [m]. Defaults to h_minus_1 + 0.1 to 4 m.
        flex_values: Flexibility values, one curve each: phi_0 directly, or the reserve h_max_0 - h_0
            [m]. Defaults to `flexibility_spec.default_plot_values()`.
        h_minus_1: Capacity of the existing dike [m].
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = cost_model.flexibility_spec
    if h_0_range is None:
        h_0_range = h_minus_1 + np.linspace(0.1, 4.0, 100)
    if flex_values is None:
        flex_values = spec.default_plot_values()

    f_minus_1 = spec.initial_state(h_minus_1)
    for value in flex_values:
        costs = [cost_model.initial_cost_fn(h_minus_1, *f_minus_1, h_0, *spec.resolve_flexibility_state(h_0, value)) for h_0 in h_0_range]  # type: ignore # fmt: skip
        style = {"color": "black", "linewidth": 2.5} if value == 0.0 else {}
        ax.plot(h_0_range * 100, np.array(costs) / EUR_PER_MEUR, label=_flex_curve_label(spec, "h_0", value), **style)  # fmt: skip

    ax.set_xlabel("$h_0$ [cm]")
    ax.set_ylabel("Initial cost [M€]")
    ax.legend()
    return fig


def plot_cost_initial_vs_flexibility(
    cost_model: CostModel,
    h_0_values: list[float] | None = None,
    flex_range: list[float] | None = None,
    h_minus_1: float = 3.0,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the cost of the first decision against the flexibility value, one curve per h_0.

    Args:
        cost_model: Cost model to evaluate.
        h_0_values: Values of h_0 [m], one curve each. Defaults to h_minus_1 + 1, 2, 3, 4 m.
        flex_range: Flexibility values on the x-axis: phi_0 directly, or the reserve h_max_0 - h_0
            [m]. Defaults to `flexibility_spec.default_plot_range()`.
        h_minus_1: Capacity of the existing dike [m].
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = cost_model.flexibility_spec
    if flex_range is None:
        flex_range = spec.default_plot_range()
    if h_0_values is None:
        h_0_values = [h_minus_1 + x for x in (1.0, 2.0, 3.0, 4.0)]

    f_minus_1 = spec.initial_state(h_minus_1)
    symbol = spec.labels[0].strip("$")
    x_values = np.array(flex_range) * (100 if spec.height_like[0] else 1)
    for h_0 in h_0_values:
        costs = [cost_model.initial_cost_fn(h_minus_1, *f_minus_1, h_0, *spec.resolve_flexibility_state(h_0, value)) for value in flex_range]  # type: ignore # fmt: skip
        ax.plot(
            x_values,
            np.array(costs) / EUR_PER_MEUR,
            label=f"$h_0 = {h_0 * 100:.0f}$ cm",
        )
        ax.plot(0.0, costs[0] / EUR_PER_MEUR, marker="o", color="black", zorder=3, label="Inflexible dike" if h_0 == h_0_values[0] else None)  # fmt: skip

    ax.set_xlabel(f"${symbol} - h_0$ [cm]" if spec.height_like[0] else spec.labels[0])
    ax.set_ylabel("Initial cost [M€]")
    ax.legend()
    return fig


def plot_cost_adapt_vs_hiplus1(
    cost_model: CostModel,
    h_i: float = 3.0,
    h_i_plus_1_range: np.ndarray | None = None,
    flex_values: list[float] | None = None,
    t_i: float | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the cost of a later upgrade from h_i against the new height h_i_plus_1, one curve per
    flexibility value.

    Args:
        cost_model: Cost model to evaluate.
        h_i: Height before the upgrade [m].
        h_i_plus_1_range: Heights after the upgrade [m]. Defaults to h_i + 0.01 to 3 m.
        flex_values: Flexibility values before the upgrade, one curve each: phi directly, or the
            reserve h_max - h_i [m]. Defaults to `flexibility_spec.default_plot_values()`.
        t_i: Years since the first decision [yr]; only used by a time-aware model (land model).
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = cost_model.flexibility_spec
    if h_i_plus_1_range is None:
        h_i_plus_1_range = np.linspace(h_i + 0.01, h_i + 3.0, 100)
    if flex_values is None:
        flex_values = spec.default_plot_values()

    time_kwargs = _plot_time_kwargs(cost_model, t_i)
    for value in flex_values:
        f_i = spec.resolve_flexibility_state(h_i, value)
        costs = [cost_model.adapt_cost_fn(h_i, *f_i, h_i_plus_1, *_auto_updated_flex_state(spec, f_i, h_i, h_i_plus_1), **time_kwargs) for h_i_plus_1 in h_i_plus_1_range]  # type: ignore # fmt: skip
        style = {"color": "black", "linewidth": 2.5} if value == 0.0 else {}
        ax.plot(h_i_plus_1_range * 100, np.array(costs) / EUR_PER_MEUR, label=_flex_curve_label(spec, "h_i", value), **style)  # fmt: skip

    ax.set_xlabel("$h_{i+1}$ [cm]")
    ax.set_ylabel("Upgrade cost [M€]")
    ax.legend()
    return fig


def plot_cost_adapt_vs_flexibility(
    cost_model: CostModel,
    h_i: float = 3.0,
    flex_range: list[float] | None = None,
    deltas: list[float] | None = None,
    t_i: float | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the cost of a later upgrade from h_i against the flexibility value, one curve per upgrade
    size h_i_plus_1 - h_i.

    Args:
        cost_model: Cost model to evaluate.
        h_i: Height before the upgrade [m].
        flex_range: Flexibility values before the upgrade on the x-axis: phi directly, or the reserve
            h_max - h_i [m]. Defaults to `flexibility_spec.default_plot_range()`.
        deltas: Upgrade sizes h_i_plus_1 - h_i [m], each > 0. Defaults to 0.5, 1, 1.5, 2 m.
        t_i: Years since the first decision [yr]; only used by a time-aware model (land model).
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    spec = cost_model.flexibility_spec
    if flex_range is None:
        flex_range = spec.default_plot_range()
    if deltas is None:
        deltas = [0.5, 1.0, 1.5, 2.0]

    time_kwargs = _plot_time_kwargs(cost_model, t_i)
    x_values = (h_i + np.array(flex_range)) * 100 if spec.height_like[0] else np.array(flex_range)  # fmt: skip
    x_0 = h_i * 100 if spec.height_like[0] else 0.0
    for i, delta in enumerate(deltas):
        h_i_plus_1 = h_i + delta
        costs = []
        for value in flex_range:
            f_i = spec.resolve_flexibility_state(h_i, value)
            f_i_plus_1 = _auto_updated_flex_state(spec, f_i, h_i, h_i_plus_1)
            costs.append(cost_model.adapt_cost_fn(h_i, *f_i, h_i_plus_1, *f_i_plus_1, **time_kwargs))  # type: ignore # fmt: skip
        costs = np.array(costs) / EUR_PER_MEUR
        ax.plot(x_values, costs, label=f"$h_{{i+1}} = h_i + {delta * 100:.0f}$ cm")
        ax.plot(x_0, costs[0], marker="o", color="black", zorder=3, label="Inflexible dike" if i == 0 else None)  # fmt: skip

    ax.set_xlabel(f"{spec.labels[0]} [cm]" if spec.height_like[0] else spec.labels[0])
    ax.set_ylabel("Upgrade cost [M€]")
    ax.legend()
    return fig


def plot_open_land_width(
    land_models: dict[str, LandCostModel],
    years: np.ndarray | None = None,
    h_max: float | None = None,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the open land width over time (`open_land_width`), one curve per land model.

    Args:
        land_models: Name -> LandCostModel, e.g. one per land-use scenario.
        years: Years since the first decision on the x-axis [yr]; defaults to 0-160.
        h_max: Secured height [m]; defaults to h_ref (no reservation).
        ax: Axes to draw on; a new figure is created if None.
    """
    fig, ax = _new_axes(ax)
    years = np.linspace(0, 160, 161) if years is None else np.asarray(years)
    for name, m in land_models.items():
        secured = m.h_ref if h_max is None else h_max
        widths = [open_land_width(secured, tau, m.h_ref, m.W_available_0, m.lambda_enc, m.kappa_1, m.kappa_2) for tau in years]  # fmt: skip
        ax.plot(years, widths, label=rf"{name} ($\lambda_\mathrm{{enc}}$ = {m.lambda_enc:g} m/yr)")
    ax.set_xlabel("Years after the first decision [yr]")
    ax.set_ylabel("Open land width [m]")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()
    return fig
