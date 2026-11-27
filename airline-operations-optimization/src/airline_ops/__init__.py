"""Aircraft routing, crew pairing and disruption simulation for a synthetic European hub airline.

The package builds a synthetic daily schedule around one hub, plans it with four
methods (greedy, sequential, integrated and robust), checks every plan against an
independent validator, and simulates the plans under stochastic delays and
disruption scenarios.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
