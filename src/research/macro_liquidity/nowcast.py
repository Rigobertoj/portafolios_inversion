"""Current-state nowcasting for macro/liquidity research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .policy import MacroLiquidityPolicy
from .regimes import CurrentRegimeSnapshot, classify_liquidity_regime, classify_macro_regime
from .transforms import build_score_frame


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _score(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    return float(value) if pd.notna(value) else default


@dataclass(frozen=True)
class NowcastResult:
    """Current macro/liquidity diagnosis with evidence quality metrics."""

    current: CurrentRegimeSnapshot
    scores: pd.DataFrame
    block_scores: pd.DataFrame
    confidence_components: pd.DataFrame
    contributors: pd.DataFrame
    methodology: dict[str, object]

    def report(self) -> dict[str, pd.DataFrame]:
        """Return notebook-friendly tables."""

        return {
            "scores": self.scores,
            "block_scores": self.block_scores,
            "confidence_components": self.confidence_components,
            "contributors": self.contributors,
        }


class MacroLiquidityNowcaster:
    """Diagnose the current macro/liquidity state from engineered features."""

    def __init__(self, policy: Optional[MacroLiquidityPolicy] = None) -> None:
        self.policy = policy or MacroLiquidityPolicy()

    @staticmethod
    def _block_scores(features: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if features.empty:
            return pd.DataFrame()
        for keys, group in features.groupby(["date", "engine", "block"], dropna=False):
            date, engine, block = keys
            values = pd.to_numeric(group["signed_zscore"], errors="coerce")
            weights = pd.to_numeric(group["weight"], errors="coerce").abs()
            valid = values.notna() & weights.notna() & (weights > 0)
            score = np.nan
            if valid.any():
                score = float((values[valid] * weights[valid]).sum() / weights[valid].sum())
            rows.append(
                {
                    "date": date,
                    "engine": engine,
                    "block": block,
                    "score": score,
                    "coverage": float(values.notna().mean()),
                    "diffusion": float((values > 0).mean()),
                    "available_indicators": int(values.notna().sum()),
                    "total_indicators": int(group["series"].nunique()),
                }
            )
        return pd.DataFrame(rows).sort_values(["date", "engine", "block"])

    def _confidence_components(
        self,
        features: pd.DataFrame,
        scores: pd.DataFrame,
        block_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        if scores.empty:
            return pd.DataFrame()
        latest = scores.sort_values("date").iloc[-1]
        latest_date = pd.Timestamp(latest["date"])
        latest_features = features[features["date"] == latest_date]
        latest_blocks = block_scores[block_scores["date"] == latest_date]

        coverage = _clip01(
            float(
                np.nanmean(
                    [
                        _score(latest, "macro_coverage", 0.0),
                        _score(latest, "liquidity_coverage", 0.0),
                    ]
                )
            )
        )

        signed = pd.to_numeric(latest_features["signed_zscore"], errors="coerce").dropna()
        agreement = 0.0
        if not signed.empty:
            positive_share = float((signed > 0).mean())
            agreement = 2.0 * abs(positive_share - 0.5)

        if len(scores) >= 6:
            recent = scores.sort_values("date").tail(6)
            volatility = recent[["macro_score", "liquidity_score"]].std(ddof=0).mean()
            stability = _clip01(1.0 / (1.0 + float(volatility if pd.notna(volatility) else 0.0)))
        else:
            stability = 0.50

        if "publication_lag_days" in latest_features.columns and not latest_features.empty:
            avg_lag = pd.to_numeric(latest_features["publication_lag_days"], errors="coerce").mean()
            freshness = _clip01(1.0 - float(avg_lag if pd.notna(avg_lag) else 0.0) / 90.0)
        else:
            freshness = 0.75

        macro_score = _score(latest, "macro_score", 0.0)
        liquidity_score = _score(latest, "liquidity_score", 0.0)
        contradiction = (
            abs(macro_score - liquidity_score) >= self.policy.confidence.contradiction_threshold
            and np.sign(macro_score) != np.sign(liquidity_score)
        )
        narrative = 0.55 if contradiction else 1.0

        weights = self.policy.confidence
        confidence = _clip01(
            coverage * weights.coverage_weight
            + agreement * weights.agreement_weight
            + stability * weights.stability_weight
            + freshness * weights.freshness_weight
            + narrative * weights.narrative_weight
        )

        return pd.DataFrame(
            [
                {
                    "date": latest_date,
                    "coverage": coverage,
                    "agreement": agreement,
                    "stability": stability,
                    "freshness": freshness,
                    "narrative_coherence": narrative,
                    "macro_liquidity_contradiction": bool(contradiction),
                    "confidence": confidence,
                    "block_count": int(latest_blocks["block"].nunique()),
                }
            ]
        )

    @staticmethod
    def _contributors(features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()
        latest_date = features["date"].max()
        latest = features[features["date"] == latest_date].copy()
        latest["contribution"] = (
            pd.to_numeric(latest["signed_zscore"], errors="coerce")
            * pd.to_numeric(latest["weight"], errors="coerce").abs()
        )
        return (
            latest[
                [
                    "date",
                    "engine",
                    "block",
                    "series",
                    "signal",
                    "signed_zscore",
                    "weight",
                    "contribution",
                ]
            ]
            .sort_values("contribution", key=lambda values: values.abs(), ascending=False)
            .reset_index(drop=True)
        )

    def fit_transform(self, features: pd.DataFrame) -> NowcastResult:
        """Return current-regime diagnosis from a feature panel."""

        scores = build_score_frame(features)
        if scores.empty:
            raise ValueError("features are empty; cannot nowcast macro/liquidity regime.")

        block_scores = self._block_scores(features)
        confidence_components = self._confidence_components(features, scores, block_scores)
        confidence = float(confidence_components["confidence"].iloc[0])
        latest = scores.sort_values("date").iloc[-1]
        current = CurrentRegimeSnapshot(
            date=pd.Timestamp(latest["date"]),
            macro_regime=classify_macro_regime(latest),
            liquidity_regime=classify_liquidity_regime(_score(latest, "liquidity_score")),
            macro_score=_score(latest, "macro_score"),
            liquidity_score=_score(latest, "liquidity_score"),
            confidence=confidence,
            macro_coverage=_score(latest, "macro_coverage", 0.0),
            liquidity_coverage=_score(latest, "liquidity_coverage", 0.0),
        )
        return NowcastResult(
            current=current,
            scores=scores,
            block_scores=block_scores,
            confidence_components=confidence_components,
            contributors=self._contributors(features),
            methodology=self.policy.methodology_summary(),
        )

