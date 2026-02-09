# === CHUNK INDEPENDIENTE: CorrelationPortfolioSelector ===
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class CorrelationPortfolioSelector:
    """
    Selector escalable para construir portafolios de minima correlacion por grupos.

    Flujo:
    1) Rankea activos dentro de cada grupo usando su correlacion promedio con pares.
    2) Toma los mejores candidatos por grupo (top_k_per_group).
    3) Construye un portafolio multi-grupo y vuelve a filtrar por minima correlacion.
    4) Permite actualizar un portafolio existente con nuevos candidatos.

    Un grupo puede representar sector, factor de riesgo, industria, region, etc.
    """

    start_date: str
    end_date: Optional[str] = None
    price_field: str = "Close"
    use_absolute_corr: bool = True
    min_coverage: float = 0.80

    prices_by_group: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict)
    returns_by_group: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict)
    ranking_by_group: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    @staticmethod
    def _as_list(tickers: Iterable[str]) -> List[str]:
        if isinstance(tickers, str):
            return [tickers.upper()]
        return [str(t).upper() for t in tickers]

    def _download_prices(self, tickers: Sequence[str]) -> pd.DataFrame:
        unique_tickers = list(dict.fromkeys(self._as_list(tickers)))
        if not unique_tickers:
            return pd.DataFrame()

        raw = yf.download(
            unique_tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if self.price_field not in raw.columns.get_level_values(0):
                raise ValueError(
                    f"price_field '{self.price_field}' no existe en la descarga."
                )
            prices = raw[self.price_field].copy()
        else:
            if self.price_field not in raw.columns:
                raise ValueError(
                    f"price_field '{self.price_field}' no existe en la descarga."
                )
            prices = raw[[self.price_field]].copy()
            prices.columns = [unique_tickers[0]]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=unique_tickers[0])

        prices = prices.sort_index().dropna(how="all")
        return prices

    def _apply_coverage_filter(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return prices
        coverage = prices.notna().mean()
        keep = coverage[coverage >= self.min_coverage].index.tolist()
        return prices[keep]

    def _compute_corr_scores(
        self,
        corr: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Score por ticker = correlacion promedio con sus pares.
        Entre menor sea el score, menor dependencia del resto del grupo.
        """
        if corr.shape[1] < 2:
            return pd.Series(dtype=float)

        scores: Dict[str, float] = {}

        for ticker in corr.columns:
            peers = [col for col in corr.columns if col != ticker]
            values = corr.loc[peers, ticker].dropna()
            if values.empty:
                scores[ticker] = np.nan
                continue

            if self.use_absolute_corr:
                values = values.abs()

            if weights:
                peer_weights = np.array(
                    [max(float(weights.get(peer, 0.0)), 0.0) for peer in values.index],
                    dtype=float,
                )
                if np.allclose(peer_weights.sum(), 0.0):
                    score = float(values.mean())
                else:
                    score = float(
                        np.average(values.to_numpy(dtype=float), weights=peer_weights)
                    )
            else:
                score = float(values.mean())

            scores[ticker] = score

        return pd.Series(scores, name="corr_score").dropna().sort_values()

    def rank_within_groups(
        self,
        grouped_tickers: Dict[str, Sequence[str]],
        intra_group_weights: Optional[Dict[str, Dict[str, float]]] = None,
        top_k: int = 1,
    ) -> pd.DataFrame:
        """Genera ranking intra-grupo de menor a mayor correlacion promedio."""
        if top_k < 1:
            raise ValueError("top_k debe ser >= 1.")

        self.prices_by_group = {}
        self.returns_by_group = {}
        ranking_rows: List[pd.DataFrame] = []

        for group, tickers in grouped_tickers.items():
            tickers_clean = self._as_list(tickers)
            if len(tickers_clean) < 2:
                continue

            prices = self._download_prices(tickers_clean)
            prices = self._apply_coverage_filter(prices)
            if prices.shape[1] < 2:
                continue

            returns = prices.pct_change(fill_method=None).dropna(how="all")
            if returns.shape[1] < 2:
                continue

            corr = returns.corr()
            group_weights = None
            if intra_group_weights is not None:
                group_weights = intra_group_weights.get(group)

            scores = self._compute_corr_scores(corr, weights=group_weights)
            if scores.empty:
                continue

            rank_df = scores.reset_index()
            rank_df.columns = ["ticker", "corr_score"]
            rank_df["group"] = group
            rank_df["rank_in_group"] = np.arange(1, len(rank_df) + 1)
            rank_df["selected"] = rank_df["rank_in_group"] <= top_k

            ranking_rows.append(
                rank_df[["group", "ticker", "corr_score", "rank_in_group", "selected"]]
            )
            self.prices_by_group[group] = prices
            self.returns_by_group[group] = returns

        if not ranking_rows:
            raise ValueError(
                "No se pudo generar ranking. Revisa tickers, fechas o cobertura minima."
            )

        self.ranking_by_group = (
            pd.concat(ranking_rows, ignore_index=True)
            .sort_values(["group", "rank_in_group"])
            .reset_index(drop=True)
        )
        return self.ranking_by_group.copy()

    def build_multigroup_portfolio(
        self,
        top_k_per_group: int = 1,
        final_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Construye el portafolio final tomando candidatos de baja correlacion por grupo.

        Regla base:
        - Toma el mejor candidato por grupo (o de los top_k por grupo, segun ranking).
        - Vuelve a rankear esos candidatos por correlacion mutua para obtener el set final.
        """
        if self.ranking_by_group.empty:
            raise ValueError("Primero ejecuta rank_within_groups(...).")
        if top_k_per_group < 1:
            raise ValueError("top_k_per_group debe ser >= 1.")

        candidate_rows = self.ranking_by_group[
            self.ranking_by_group["rank_in_group"] <= top_k_per_group
        ]
        if candidate_rows.empty:
            raise ValueError("No hay candidatos con el top_k_per_group configurado.")

        per_group = (
            candidate_rows.sort_values(["group", "rank_in_group"])
            .groupby("group", as_index=False)
            .first()[["group", "ticker", "corr_score"]]
        )

        candidate_tickers = list(dict.fromkeys(per_group["ticker"].tolist()))
        if len(candidate_tickers) < 2:
            raise ValueError(
                "Se requieren al menos 2 tickers para construir correlacion multi-grupo."
            )

        prices = self._download_prices(candidate_tickers)
        prices = self._apply_coverage_filter(prices)
        returns = prices.pct_change(fill_method=None).dropna(how="all")
        corr = returns.corr()

        portfolio_scores = self._compute_corr_scores(corr).rename("portfolio_corr_score")
        if portfolio_scores.empty:
            raise ValueError("No se pudieron calcular scores del portafolio multi-grupo.")

        if final_size is None:
            final_size = len(portfolio_scores)
        final_size = max(1, min(final_size, len(portfolio_scores)))
        final_tickers = portfolio_scores.head(final_size).index.tolist()

        final_corr = corr.loc[final_tickers, final_tickers]
        final_mean_offdiag_corr = np.nan
        if len(final_tickers) > 1:
            tri_idx = np.tril_indices(len(final_tickers), k=-1)
            vals = final_corr.to_numpy()[tri_idx]
            if self.use_absolute_corr:
                vals = np.abs(vals)
            final_mean_offdiag_corr = float(np.nanmean(vals))

        return {
            "per_group_selection": per_group,
            "candidate_tickers": candidate_tickers,
            "candidate_corr_matrix": corr,
            "candidate_scores": portfolio_scores,
            "final_tickers": final_tickers,
            "final_corr_matrix": final_corr,
            "final_mean_offdiag_corr": final_mean_offdiag_corr,
        }

    def update_portfolio(
        self,
        current_tickers: Sequence[str],
        candidate_tickers: Sequence[str],
        max_new_assets: int = 1,
    ) -> Dict[str, Any]:
        """Actualiza un portafolio agregando candidatos con menor correlacion al actual."""
        if max_new_assets < 1:
            raise ValueError("max_new_assets debe ser >= 1.")

        current = list(dict.fromkeys(self._as_list(current_tickers)))
        candidates = [t for t in self._as_list(candidate_tickers) if t not in current]

        if not candidates:
            return {
                "updated_tickers": current,
                "new_tickers": [],
                "candidate_scores": pd.Series(dtype=float),
            }

        prices = self._download_prices(current + candidates)
        prices = self._apply_coverage_filter(prices)
        returns = prices.pct_change(fill_method=None).dropna(how="all")
        corr = returns.corr()

        if self.use_absolute_corr:
            corr = corr.abs()

        valid_current = [t for t in current if t in corr.columns]
        score_dict: Dict[str, float] = {}

        for ticker in candidates:
            if ticker not in corr.columns:
                continue
            peers = valid_current if valid_current else [c for c in corr.columns if c != ticker]
            values = corr.loc[peers, ticker].dropna()
            if not values.empty:
                score_dict[ticker] = float(values.mean())

        candidate_scores = pd.Series(
            score_dict, name="corr_with_portfolio"
        ).sort_values()

        new_tickers = candidate_scores.head(max_new_assets).index.tolist()
        updated_tickers = current + [t for t in new_tickers if t not in current]

        return {
            "updated_tickers": updated_tickers,
            "new_tickers": new_tickers,
            "candidate_scores": candidate_scores,
            "corr_matrix": corr,
        }

    def run_pipeline(
        self,
        grouped_tickers: Dict[str, Sequence[str]],
        top_k_in_group: int = 1,
        final_size: Optional[int] = None,
        intra_group_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Pipeline end-to-end: ranking intra-grupo + portafolio multi-grupo."""
        ranking = self.rank_within_groups(
            grouped_tickers=grouped_tickers,
            intra_group_weights=intra_group_weights,
            top_k=top_k_in_group,
        )
        portfolio = self.build_multigroup_portfolio(
            top_k_per_group=top_k_in_group,
            final_size=final_size,
        )
        return {"group_ranking": ranking, **portfolio}