"""PascalCase compatibility shim for legacy modules that subclass `AssetsResearch`."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Iterable, Optional

from ..research.assets_research import AssetsResearch as ResearchAssetsResearch, yf


@dataclass
class AssetsResearch(ResearchAssetsResearch):
    """
    Lightweight compatibility subclass around the new research implementation.

    Legacy modules in `asset_allocation` still inherit from the PascalCase
    import path. Re-declaring the constructor fields here preserves the old
    dataclass inheritance behavior while delegating all real logic to
    `src.research.assets_research.AssetsResearch`.
    """

    tickers: InitVar[Iterable[str]]
    start: InitVar[str]
    end: InitVar[Optional[str]] = None
    price_field: InitVar[str] = "Close"

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
    ) -> None:
        ResearchAssetsResearch.__init__(
            self,
            tickers=tickers,
            start=start,
            end=end,
            price_field=price_field,
        )


__all__ = [
    "AssetsResearch",
    "yf",
]
