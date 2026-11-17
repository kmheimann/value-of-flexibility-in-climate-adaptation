"""
risk_model.py — Expected annual flood loss of a dike (thesis Appendix A6).

    RiskModel          Abstract risk model: `risk_fn(h, theta)` gives the expected annual flood loss [€]
                       of a dike with capacity h [m] at sea level rise theta [m].
    StandardRiskModel  The thesis risk model: only overtopping, with an exponentially distributed
                       extreme water level and a constant damage.
    plot_risk          Expected annual flood loss against the dike capacity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

EUR_PER_MEUR = 1e6  # Display unit of costs [€ per M€]


class RiskModel(ABC):
    """Abstract risk model, used by `decision_analysis.backward_induction`."""

    @abstractmethod
    def risk_fn(self, h_i: float, state_i_plus_1: float) -> float:
        """Expected annual flood loss [€] of a dike with capacity h_i [m] (absolute) at the sea level
        rise state_i_plus_1 [m]."""


@dataclass(frozen=True)
class StandardRiskModel(RiskModel):
    """Risk model of the thesis (adopted from Jongejan et al. 2026):

        r(h, theta) = D * exp(-(h - r * theta - a) / b)   if h - r * theta >= a, else D.

    Args:
        r: Scaling factor of sea level rise, accounting for waves [-].
        a: Location parameter of the exponential extreme water level [m].
        b: Scale parameter of the exponential extreme water level [m].
        D: Flood damage [€].
    """

    r: float
    a: float
    b: float
    D: float

    def risk_fn(self, h_i: float, state_i_plus_1: float) -> float:
        """Expected annual flood loss [€] of a dike with capacity h_i [m] at sea level rise
        state_i_plus_1 [m]."""
        if h_i < state_i_plus_1 * self.r + self.a:
            failure_prob = 1.0  # The extreme water level is at least a
        else:
            failure_prob = float(
                np.exp(-(h_i - state_i_plus_1 * self.r - self.a) / self.b)
            )
        return failure_prob * self.D


def plot_risk(
    risk_model: RiskModel,
    heights: np.ndarray | list[float] | None = None,
    states: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5),
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plots the expected annual flood loss [M€/yr] against the dike capacity on a log scale, one
    curve per sea level rise state.

    Args:
        risk_model: Risk model to evaluate.
        heights: Dike capacities h [m] on the x-axis; defaults to 3-8 m.
        states: Sea level rise states theta [m], one curve each.
        ax: Axes to draw on; a new figure is created if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    heights = np.linspace(3.0, 8.0, 200) if heights is None else np.asarray(heights)
    for theta in states:
        losses = [risk_model.risk_fn(h, theta) / EUR_PER_MEUR for h in heights]
        ax.plot(heights * 100, losses, label=rf"$\theta$ = {theta * 100:.0f} cm")
    ax.set_yscale("log")
    ax.set_xlabel("Dike capacity $h$ [cm]")
    ax.set_ylabel("Expected annual flood loss [M€/yr]")
    ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()
    return fig
