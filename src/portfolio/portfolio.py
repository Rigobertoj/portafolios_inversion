"""Core portfolio entity built by composition over prepared market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..research.assets_research import AssetsResearch


@dataclass
class Portfolio:
    """
    Represent a weighted portfolio built from asset prices and returns.

    The class is intentionally lightweight: it stores aligned price and return
    data, validates the portfolio weights, and exposes the canonical series
    that downstream analysis layers can reuse.
    """

    prices: pd.DataFrame
    weights: Iterable[float]
    returns: Optional[pd.DataFrame] = None
    tickers: Optional[Iterable[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    price_field: str = "Close"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_prices = self._normalize_prices(self.prices)
        resolved_tickers = self._resolve_tickers(
            tickers=self.tickers,
            prices=normalized_prices,
        )
        normalized_prices = normalized_prices.loc[:, resolved_tickers]

        normalized_returns = self._normalize_returns(
            returns=self.returns,
            prices=normalized_prices,
            tickers=resolved_tickers,
        )

        self.prices = normalized_prices
        self.returns = normalized_returns
        self.tickers = resolved_tickers
        self.weights = self._validate_weights(self.weights, len(resolved_tickers))
        self.start = self.start or pd.Timestamp(normalized_prices.index.min()).date().isoformat()
        self.end = self.end or (
            pd.Timestamp(normalized_prices.index.max()) + pd.Timedelta(days=1)
        ).date().isoformat()
        self.price_field = self._validate_price_field(self.price_field)
        self.name = self.name or "portfolio"

    @classmethod
    def from_research(
        cls,
        research: AssetsResearch,
        weights: Iterable[float],
        *,
        tickers: Optional[Iterable[str]] = None,
        name: Optional[str] = None,
    ) -> "Portfolio":
        """Build a portfolio from an `AssetsResearch` instance."""
        resolved_tickers = list(research.tickers if tickers is None else tickers)
        prices = research.get_prices(resolved_tickers)
        returns = research.get_returns(resolved_tickers)
        return cls(
            prices=prices,
            returns=returns,
            tickers=resolved_tickers,
            weights=weights,
            start=research.start,
            end=research.end,
            price_field=research.price_field,
            name=name,
        )

    @staticmethod
    def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        normalized = prices.sort_index().dropna().copy()
        if normalized.empty:
            raise ValueError("prices must contain at least one complete observation.")
        if len(normalized.columns) == 0:
            raise ValueError("prices must contain at least one asset column.")
        return normalized

    @staticmethod
    def _resolve_tickers(
        tickers: Optional[Iterable[str]],
        prices: pd.DataFrame,
    ) -> list[str]:
        if tickers is None:
            resolved = list(prices.columns)
        else:
            resolved = AssetsResearch._normalize_tickers(tickers)

        if not resolved:
            raise ValueError("tickers must contain at least one symbol.")

        missing = [ticker for ticker in resolved if ticker not in prices.columns]
        if missing:
            raise ValueError(f"prices is missing configured tickers: {missing}")

        return resolved

    @staticmethod
    def _normalize_returns(
        *,
        returns: Optional[pd.DataFrame],
        prices: pd.DataFrame,
        tickers: list[str],
    ) -> pd.DataFrame:
        if returns is None:
            normalized = prices.pct_change().dropna()
        else:
            if not isinstance(returns, pd.DataFrame):
                raise TypeError("returns must be a pandas DataFrame.")
            normalized = returns.sort_index().dropna().copy()
            missing = [ticker for ticker in tickers if ticker not in normalized.columns]
            if missing:
                raise ValueError(f"returns is missing configured tickers: {missing}")
            normalized = normalized.loc[:, tickers]

        if normalized.empty:
            raise ValueError("returns must contain at least one observation.")

        return normalized

    @staticmethod
    def _validate_weights(weights: Iterable[float], n_assets: int) -> np.ndarray:
        vector = np.asarray(weights, dtype=float)
        if vector.ndim != 1:
            raise ValueError("weights must be a 1D iterable.")
        if len(vector) != n_assets:
            raise ValueError("weights length must match the number of tickers.")
        if not np.isclose(vector.sum(), 1.0):
            raise ValueError("weights must sum to 1.")
        return vector

    @staticmethod
    def _validate_price_field(price_field: str) -> str:
        if not isinstance(price_field, str) or not price_field.strip():
            raise ValueError("price_field must be a non-empty string.")
        return price_field.strip()

    @property
    def weight(self) -> np.ndarray:
        """Backward-friendly alias around the portfolio weight vector."""
        return self.weights.copy()

    def asset_prices(self) -> pd.DataFrame:
        """Return the aligned asset price table."""
        return self.prices.loc[:, self.tickers].copy()

    def asset_returns(self) -> pd.DataFrame:
        """Return the aligned asset return table."""
        return self.returns.loc[:, self.tickers].copy()

    def portfolio_returns(self) -> pd.Series:
        """Return the weighted portfolio daily return series."""
        values = self.asset_returns().to_numpy(dtype=float) @ self.weights
        return pd.Series(values, index=self.returns.index, name=self.name)

    def wealth_index(self, initial_value: float = 1.0) -> pd.Series:
        """Return the compounded wealth path of the portfolio."""
        if initial_value <= 0.0:
            raise ValueError("initial_value must be greater than zero.")
        returns = self.portfolio_returns()
        wealth = initial_value * (1.0 + returns).cumprod()
        wealth.name = self.name
        return wealth

    def realized_return(self) -> float:
        """Return the effective realized return of the portfolio path."""
        returns = self.portfolio_returns()
        return float((1.0 + returns).prod() - 1.0)

    def update_weights(self, weights: Iterable[float]) -> None:
        """Validate and persist a new portfolio allocation in place."""
        self.weights = self._validate_weights(weights, len(self.tickers))

    def with_weights(
        self,
        weights: Iterable[float],
        *,
        name: Optional[str] = None,
    ) -> "Portfolio":
        """Return a new portfolio instance with the same data and new weights."""
        return Portfolio(
            prices=self.asset_prices(),
            returns=self.asset_returns(),
            tickers=self.tickers,
            weights=weights,
            start=self.start,
            end=self.end,
            price_field=self.price_field,
            name=self.name if name is None else name,
        )


__all__ = [
    "Portfolio",
]
