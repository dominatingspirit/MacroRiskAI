"""Macroeconomic data providers (swappable source).

The pipeline never generates or reads macro data directly. It asks a
:class:`MacroProvider` for a quarterly macro panel via
``get_macro_panel(quarters)``. Every provider returns the **same schema**:

    join_key columns (Year, Quarter) + configured variables
    + macro_source (str) + macro_is_synthetic (bool)

This contract is what makes the synthetic placeholder swappable for real
history: switching ``macro.provider`` in the config from ``synthetic`` to
``historical`` (and supplying a CSV) changes the data source with **no change
to the assembler or any downstream ML code**.

Providers
---------
``SyntheticMacroProvider``
    Generates statistically plausible, temporally continuous, economically
    coherent quarterly series (India-like). DEVELOPMENT ONLY.

``HistoricalMacroProvider``
    Reads a real historical macro CSV and returns the same schema. Ready for
    when real data is available; requires only that the file contain the
    join_key + configured variables.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MacroProvider(ABC):
    """Abstract macro data source. All providers share one output schema."""

    def __init__(self, config: dict[str, Any]):
        self.macro_cfg = config["macro"]
        self.join_key: list[str] = self.macro_cfg["join_key"]
        self.variables: list[str] = self.macro_cfg["variables"]
        prov = self.macro_cfg["provenance"]
        self.source_col = prov["source_column"]
        self.flag_col = prov["synthetic_flag_column"]

    @abstractmethod
    def get_macro_panel(self, quarters: pd.DataFrame) -> pd.DataFrame:
        """Return a macro panel for the requested quarters.

        Parameters
        ----------
        quarters:
            DataFrame with at least the join_key columns and ``time_index``,
            one row per distinct quarter, chronologically ordered.
        """

    # Shared post-condition enforcement so every provider is guaranteed to
    # emit the canonical schema.
    def _finalize(self, df: pd.DataFrame, *, source_label: str, is_synthetic: bool) -> pd.DataFrame:
        missing = set(self.join_key) | set(self.variables)
        absent = missing - set(df.columns)
        if absent:
            raise ValueError(f"Macro provider output missing columns: {sorted(absent)}")
        df = df[self.join_key + self.variables].copy()
        df[self.source_col] = source_label
        df[self.flag_col] = is_synthetic
        return df


class SyntheticMacroProvider(MacroProvider):
    """Generate coherent synthetic quarterly macro series (development only)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.params = self.macro_cfg["synthetic_params"]
        self.source_label = self.macro_cfg["provenance"]["synthetic_source_label"]
        self.seed = config["synthetic"]["random_seed"]

    def get_macro_panel(self, quarters: pd.DataFrame) -> pd.DataFrame:
        q = quarters.sort_values("time_index").drop_duplicates("time_index").reset_index(drop=True)
        n = len(q)
        rng = np.random.default_rng(self.seed)
        p = self.params

        repo = self._simulate_repo(n, rng, p["repo_rate"])
        reverse_repo = self._simulate_reverse_repo(repo, rng, p)
        cpi_infl = self._simulate_cpi_inflation(n, rng, p["cpi_inflation"])
        cpi_index = self._compound_index(cpi_infl, p["cpi_index_start"])
        oil = self._simulate_oil(n, rng, p["oil_price"])
        wpi = self._simulate_wpi(cpi_infl, oil, rng, p)
        fx = self._simulate_fx(n, oil, rng, p["exchange_rate"])

        out = q[self.join_key].copy()
        out["CPI_Combined_Index"] = np.round(cpi_index, 2)
        out["CPI_Inflation_Rate"] = np.round(cpi_infl, 2)
        out["WPI"] = np.round(wpi, 2)
        out["Repo_Rate"] = np.round(repo, 2)
        out["Reverse_Repo_Rate"] = np.round(reverse_repo, 2)
        out["oil_price"] = np.round(oil, 2)
        out["exchange_rate"] = np.round(fx, 2)
        return self._finalize(out, source_label=self.source_label, is_synthetic=True)

    # ------------------------------------------------------------------ #
    # Individual series. Each is a smooth (AR / bounded random-walk) process
    # so consecutive quarters never jump unrealistically.
    # ------------------------------------------------------------------ #
    def _simulate_repo(self, n: int, rng, cfg: dict[str, Any]) -> np.ndarray:
        vals = np.empty(n)
        vals[0] = cfg["start"]
        step_round = cfg["round"]
        for t in range(1, n):
            step = rng.normal(0, cfg["step_std"])
            # Round steps to policy-realistic increments and cap gradualness.
            step = round(step / step_round) * step_round
            step = float(np.clip(step, -0.25, 0.25))
            vals[t] = np.clip(vals[t - 1] + step, cfg["min"], cfg["max"])
        return vals

    def _simulate_reverse_repo(self, repo: np.ndarray, rng, p: dict[str, Any]) -> np.ndarray:
        noise = rng.normal(0, p["corridor_noise_std"], size=len(repo))
        rev = repo - p["repo_corridor"] + noise
        # Reverse repo must stay strictly below the repo rate.
        return np.minimum(rev, repo - 0.10)

    def _simulate_cpi_inflation(self, n: int, rng, cfg: dict[str, Any]) -> np.ndarray:
        vals = np.empty(n)
        vals[0] = cfg["start"]
        phi, mu = cfg["ar_phi"], cfg["mean"]
        for t in range(1, n):
            vals[t] = mu + phi * (vals[t - 1] - mu) + rng.normal(0, cfg["noise_std"])
            vals[t] = np.clip(vals[t], cfg["min"], cfg["max"])
        return vals

    def _compound_index(self, inflation: np.ndarray, start: float) -> np.ndarray:
        """Grow an index by the quarterly share of annual inflation."""
        idx = np.empty(len(inflation))
        idx[0] = start
        for t in range(1, len(inflation)):
            idx[t] = idx[t - 1] * (1 + inflation[t] / 100.0 / 4.0)
        return idx

    def _simulate_oil(self, n: int, rng, cfg: dict[str, Any]) -> np.ndarray:
        vals = np.empty(n)
        vals[0] = cfg["start"]
        for t in range(1, n):
            pct = rng.normal(0, cfg["pct_change_std"])
            pct = float(np.clip(pct, -cfg["max_abs_pct_change"], cfg["max_abs_pct_change"]))
            vals[t] = np.clip(vals[t - 1] * (1 + pct), cfg["min"], cfg["max"])
        return vals

    def _simulate_wpi(self, cpi_infl: np.ndarray, oil: np.ndarray, rng, p: dict[str, Any]) -> np.ndarray:
        """WPI index whose quarterly inflation loads on CPI inflation and oil.

        Positive CPI loading guarantees a positive CPI–WPI relationship; a
        modest oil loading reflects wholesale prices' sensitivity to energy.
        """
        n = len(cpi_infl)
        oil_pct = np.zeros(n)
        oil_pct[1:] = np.diff(oil) / oil[:-1] * 100.0
        wpi = np.empty(n)
        wpi[0] = p["wpi_start"]
        for t in range(1, n):
            wpi_infl = (
                p["wpi_cpi_beta"] * cpi_infl[t]
                + p["wpi_oil_beta"] * oil_pct[t]
                + rng.normal(0, p["wpi_noise_std"])
            )
            wpi[t] = wpi[t - 1] * (1 + wpi_infl / 100.0 / 4.0)
        return wpi

    def _simulate_fx(self, n: int, oil: np.ndarray, rng, cfg: dict[str, Any]) -> np.ndarray:
        """INR/USD with mild depreciation drift and positive oil sensitivity."""
        vals = np.empty(n)
        vals[0] = cfg["start"]
        oil_pct = np.zeros(n)
        oil_pct[1:] = np.diff(oil) / oil[:-1]
        for t in range(1, n):
            change = (
                cfg["drift"]
                + cfg["oil_beta"] * oil_pct[t] * vals[t - 1]
                + rng.normal(0, cfg["noise_std"])
            )
            change = float(np.clip(change, -cfg["max_abs_change"], cfg["max_abs_change"]))
            vals[t] = np.clip(vals[t - 1] + change, cfg["min"], cfg["max"])
        return vals


class HistoricalMacroProvider(MacroProvider):
    """Read real historical macro data (drop-in replacement for synthetic).

    Not used until real data is supplied. Kept here so the swap is a config
    change only.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.path = Path(self.macro_cfg["historical"]["path"])

    def get_macro_panel(self, quarters: pd.DataFrame) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Historical macro file not found: {self.path}. Provide it or set "
                f"macro.provider back to 'synthetic'."
            )
        df = pd.read_csv(self.path)
        return self._finalize(df, source_label=str(self.path.name), is_synthetic=False)


def get_macro_provider(config: dict[str, Any]) -> MacroProvider:
    """Factory: instantiate the provider named in ``config['macro']['provider']``."""
    name = config["macro"]["provider"]
    providers = {
        "synthetic": SyntheticMacroProvider,
        "historical": HistoricalMacroProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown macro provider '{name}'. Options: {list(providers)}")
    return providers[name](config)
