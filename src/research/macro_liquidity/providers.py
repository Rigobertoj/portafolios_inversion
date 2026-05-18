"""Provider credentials, metadata, and data-access adapters."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Any, Iterable, Mapping, Optional, Protocol
from urllib.parse import urlparse

import pandas as pd
import requests

from .catalog import EconomicSeriesSpec


class MissingCredentialError(RuntimeError):
    """Raised when a provider requires a credential that is not configured."""


@dataclass(frozen=True)
class ApiConnectionSpec:
    """Endpoint and credential metadata for one public data provider."""

    provider: str
    env_var: Optional[str]
    requires_key: bool
    base_url: str
    notes: str = ""


@dataclass(frozen=True)
class ProviderFetchError:
    """Error captured when a provider fails in tolerant registry mode."""

    provider: str
    series: tuple[str, ...]
    error_type: str
    message: str


def default_api_connection_specs() -> tuple[ApiConnectionSpec, ...]:
    """Return supported provider metadata without embedding private keys."""

    return (
        ApiConnectionSpec("FRED", "FRED_API_KEY", True, "https://api.stlouisfed.org/fred", "Official API requires a key."),
        ApiConnectionSpec("BEA", "BEA_API_KEY", True, "https://apps.bea.gov/api/data", "Free key after registration."),
        ApiConnectionSpec("BLS", "BLS_API_KEY", False, "https://api.bls.gov/publicAPI", "Some endpoints work without a key; v2 benefits from registration."),
        ApiConnectionSpec("CENSUS", "CENSUS_API_KEY", False, "https://api.census.gov/data", "Key recommended for higher limits."),
        ApiConnectionSpec("FED_DDP", None, False, "https://www.federalreserve.gov/datadownload", "CSV/XML/SDMX downloads."),
        ApiConnectionSpec("NY_FED", None, False, "https://markets.newyorkfed.org/api", "Markets and reference-rate data."),
        ApiConnectionSpec("TREASURY", None, False, "https://api.fiscaldata.treasury.gov/services/api/fiscal_service", "Fiscal Data API."),
        ApiConnectionSpec("EIA", "EIA_API_KEY", True, "https://api.eia.gov/v2", "Official API requires a key."),
        ApiConnectionSpec("BIS", None, False, "https://stats.bis.org/api/v2", "SDMX-style data access."),
        ApiConnectionSpec("INEGI", "INEGI_TOKEN", True, "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml", "Token required for indicator API."),
        ApiConnectionSpec("BANXICO", "BANXICO_TOKEN", True, "https://www.banxico.org.mx/SieAPIRest/service/v1", "Token required for SIE API."),
    )


IMPLEMENTED_API_PROVIDERS = frozenset({"BEA", "BLS", "FED_DDP", "FRED", "TREASURY"})


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _provider_params(spec: EconomicSeriesSpec) -> dict[str, Any]:
    return dict(getattr(spec, "provider_params", {}) or {})


def _api_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if not key.startswith("_")}


def _clean_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _period_to_timestamp(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if not text:
        return pd.NaT

    if "Q" in text and len(text) >= 6:
        year, quarter = text.upper().split("Q", 1)
        try:
            return pd.Period(f"{int(year)}Q{int(quarter)}", freq="Q").to_timestamp(how="end").normalize()
        except ValueError:
            return pd.to_datetime(text, errors="coerce")

    if "M" in text and len(text) >= 7:
        year, month = text.upper().split("M", 1)
        try:
            return pd.Period(f"{int(year)}-{int(month):02d}", freq="M").to_timestamp(how="end").normalize()
        except ValueError:
            return pd.to_datetime(text, errors="coerce")

    if len(text) == 4 and text.isdigit():
        return pd.Period(text, freq="Y").to_timestamp(how="end").normalize()

    return pd.to_datetime(text, errors="coerce")


def _year_bounds(start: Optional[str], end: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    start_year = pd.Timestamp(start).year if start is not None else None
    end_year = pd.Timestamp(end).year if end is not None else None
    return start_year, end_year


def _build_year_list(start: Optional[str], end: Optional[str]) -> Optional[str]:
    start_year, end_year = _year_bounds(start, end)
    if start_year is None and end_year is None:
        return None
    if start_year is None:
        start_year = end_year
    if end_year is None:
        end_year = start_year
    return ",".join(str(year) for year in range(start_year, end_year + 1))


class ProviderCredentials:
    """Resolve provider tokens from explicit values or environment variables.

    Credentials are intentionally independent from provider implementations.
    A provider asks this object for the token it needs; it does not know whether
    that token came from `os.environ`, a notebook-injected mapping, or a test
    fixture.
    """

    def __init__(
        self,
        env: Optional[Mapping[str, str]] = None,
        tokens: Optional[Mapping[str, str]] = None,
        connection_specs: Iterable[ApiConnectionSpec] = default_api_connection_specs(),
    ) -> None:
        self.env = dict(os.environ if env is None else env)
        self.tokens = {provider.upper(): token for provider, token in (tokens or {}).items()}
        self.connection_specs = {spec.provider.upper(): spec for spec in connection_specs}

    def token_for(self, provider: str) -> Optional[str]:
        """Return the configured token for `provider`, if any."""

        provider_name = provider.upper()
        explicit_token = self.tokens.get(provider_name)
        if isinstance(explicit_token, str) and explicit_token.strip():
            return explicit_token.strip()

        spec = self.connection_specs.get(provider_name)
        if spec is None or spec.env_var is None:
            return None
        env_token = self.env.get(spec.env_var)
        return env_token.strip() if isinstance(env_token, str) and env_token.strip() else None

    def has_token(self, provider: str) -> bool:
        """Return whether a non-empty token is configured for `provider`."""

        return self.token_for(provider) is not None

    def require_token(self, provider: str) -> str:
        """Return a token or raise a clear credential error."""

        token = self.token_for(provider)
        if token is None:
            spec = self.connection_specs.get(provider.upper())
            env_var = spec.env_var if spec is not None else f"{provider.upper()}_API_KEY"
            raise MissingCredentialError(
                f"{provider} requires a credential. Configure {env_var} outside the repo."
            )
        return token


class ProviderConfig:
    """Provider metadata plus credential readiness report.

    This class is a compatibility facade for notebooks that already use
    `ProviderConfig(env=...)`. Credential lookup is delegated to
    `ProviderCredentials`.
    """

    def __init__(
        self,
        env: Optional[Mapping[str, str]] = None,
        credentials: Optional[ProviderCredentials] = None,
        connection_specs: Iterable[ApiConnectionSpec] = default_api_connection_specs(),
        implemented_providers: Iterable[str] = IMPLEMENTED_API_PROVIDERS,
    ) -> None:
        self.connection_specs = {spec.provider.upper(): spec for spec in connection_specs}
        self.credentials = credentials or ProviderCredentials(
            env=env,
            connection_specs=self.connection_specs.values(),
        )
        self.implemented_providers = {provider.upper() for provider in implemented_providers}

    def token_for(self, provider: str) -> Optional[str]:
        """Return the token configured for `provider`, if any."""

        return self.credentials.token_for(provider)

    def is_available(self, provider: str) -> bool:
        """Return whether credentials are available for `provider`.

        This preserves the original meaning used by existing notebooks. Use
        `is_ready_for_fetch` when you need downloader readiness.
        """

        spec = self.connection_specs.get(provider.upper())
        if spec is None:
            return False
        return not spec.requires_key or self.credentials.has_token(provider)

    def is_implemented(self, provider: str) -> bool:
        """Return whether this package has a downloader for `provider`."""

        return provider.upper() in self.implemented_providers

    def is_ready_for_fetch(self, provider: str) -> bool:
        """Return whether a provider is implemented and credential-ready."""

        return self.is_implemented(provider) and self.is_available(provider)

    def availability_report(self) -> pd.DataFrame:
        """Return a tabular report of credentials and provider readiness."""

        rows = []
        for spec in self.connection_specs.values():
            token_configured = self.credentials.has_token(spec.provider)
            credential_available = (not spec.requires_key) or token_configured
            provider_implemented = self.is_implemented(spec.provider)
            rows.append(
                {
                    "provider": spec.provider,
                    "provider_registered": True,
                    "provider_implemented": provider_implemented,
                    "requires_key": spec.requires_key,
                    "env_var": spec.env_var,
                    "token_configured": token_configured,
                    "credential_available": credential_available,
                    "ready_for_fetch": provider_implemented and credential_available,
                    "available": credential_available,
                    "base_url": spec.base_url,
                    "notes": spec.notes,
                }
            )
        return pd.DataFrame(rows)

    def require_token(self, provider: str) -> str:
        """Return a token or raise a clear credential error."""

        return self.credentials.require_token(provider)


class EconomicSeriesProvider(Protocol):
    """Provider contract for economic series."""

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return columns `date`, `series`, and `value`."""


def normalize_series_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a long-form economic series frame."""

    required = {"date", "series", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"series frame must contain columns: {sorted(missing)}")

    normalized = frame.loc[:, ["date", "series", "value"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["series"] = normalized["series"].astype(str)
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "series", "value"])
    return normalized.sort_values(["series", "date"]).reset_index(drop=True)


class LocalSeriesProvider:
    """Use an existing long-form DataFrame as the economic data source."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = normalize_series_frame(frame)

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        names = {spec.name for spec in specs}
        frame = self.frame[self.frame["series"].isin(names)].copy()
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        return frame.reset_index(drop=True)


class FredApiProvider:
    """Fetch FRED observations with a detached API key."""

    observations_url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        timeout: int = 30,
        credentials: Optional[ProviderCredentials] = None,
    ) -> None:
        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        self.config = config or ProviderConfig(credentials=credentials)
        self.credentials = self.config.credentials
        self.timeout = timeout

    def _fetch_one(
        self,
        spec: EconomicSeriesSpec,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        api_key = self.credentials.require_token("FRED")
        params = {
            "series_id": spec.provider_code,
            "api_key": api_key,
            "file_type": "json",
        }
        if start is not None:
            params["observation_start"] = start
        if end is not None:
            params["observation_end"] = end

        response = requests.get(self.observations_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])
        frame = pd.DataFrame(observations)
        if frame.empty:
            return pd.DataFrame(columns=["date", "series", "value"])
        frame["series"] = spec.name
        return frame.rename(columns={"date": "date"})[["date", "series", "value"]]

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = [
            self._fetch_one(spec, start, end)
            for spec in specs
            if spec.provider.upper() == "FRED" and spec.provider_code
        ]
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.concat(frames, ignore_index=True))


class BeaApiProvider:
    """Fetch BEA data API observations using provider-specific parameters."""

    data_url = "https://apps.bea.gov/api/data"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        timeout: int = 30,
        credentials: Optional[ProviderCredentials] = None,
    ) -> None:
        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        self.config = config or ProviderConfig(credentials=credentials)
        self.credentials = self.config.credentials
        self.timeout = timeout

    def _fetch_one(
        self,
        spec: EconomicSeriesSpec,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        params = _provider_params(spec)
        api_params = {
            "UserID": self.credentials.require_token("BEA"),
            "method": "GetData",
            "ResultFormat": "JSON",
        }
        api_params.update(_api_params(params))

        if "DatasetName" not in api_params:
            raise ValueError("BEA specs require provider_params['DatasetName'].")
        if "Year" not in api_params:
            api_params["Year"] = _build_year_list(start, end) or "X"

        response = requests.get(self.data_url, params=api_params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("BEAAPI", {}).get("Results", {})
        if "Error" in results:
            raise ValueError(f"BEA API error for {spec.name}: {results['Error']}")

        records = results.get("Data", [])
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=["date", "series", "value"])

        date_field = params.get("_date_field", "TimePeriod")
        value_field = params.get("_value_field", "DataValue")
        if date_field not in frame.columns or value_field not in frame.columns:
            raise ValueError(
                f"BEA response for {spec.name} must include {date_field!r} and {value_field!r}."
            )

        output = pd.DataFrame(
            {
                "date": frame[date_field].map(_period_to_timestamp),
                "series": spec.name,
                "value": _clean_numeric(frame[value_field]),
            }
        )
        return output

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = [
            self._fetch_one(spec, start, end)
            for spec in specs
            if spec.provider.upper() == "BEA"
        ]
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.concat(frames, ignore_index=True))


class BlsApiProvider:
    """Fetch BLS public API time series."""

    timeseries_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        timeout: int = 30,
        credentials: Optional[ProviderCredentials] = None,
    ) -> None:
        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        self.config = config or ProviderConfig(credentials=credentials)
        self.credentials = self.config.credentials
        self.timeout = timeout

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        bls_specs = [spec for spec in specs if spec.provider.upper() == "BLS" and spec.provider_code]
        if not bls_specs:
            return pd.DataFrame(columns=["date", "series", "value"])

        payload: dict[str, Any] = {"seriesid": [spec.provider_code for spec in bls_specs]}
        start_year, end_year = _year_bounds(start, end)
        if start_year is not None:
            payload["startyear"] = str(start_year)
        if end_year is not None:
            payload["endyear"] = str(end_year)

        token = self.credentials.token_for("BLS")
        if token is not None:
            payload["registrationkey"] = token

        response = requests.post(
            self.timeseries_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in {None, "REQUEST_SUCCEEDED"}:
            raise ValueError(f"BLS API error: {payload.get('message')}")

        name_by_code = {spec.provider_code: spec.name for spec in bls_specs}
        rows = []
        for series in payload.get("Results", {}).get("series", []):
            series_name = name_by_code.get(series.get("seriesID"))
            if series_name is None:
                continue
            for observation in series.get("data", []):
                rows.append(
                    {
                        "date": _period_to_timestamp(
                            f"{observation.get('year')}{observation.get('period')}"
                        ),
                        "series": series_name,
                        "value": observation.get("value"),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.DataFrame(rows))


class TreasuryFiscalDataProvider:
    """Fetch Treasury Fiscal Data API endpoints."""

    base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        timeout: int = 30,
        credentials: Optional[ProviderCredentials] = None,
    ) -> None:
        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        self.config = config or ProviderConfig(credentials=credentials)
        self.credentials = self.config.credentials
        self.timeout = timeout

    def _fetch_one(
        self,
        spec: EconomicSeriesSpec,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        params = _provider_params(spec)
        endpoint = str(params.get("_endpoint") or spec.provider_code)
        if not endpoint:
            raise ValueError("Treasury specs require provider_code or provider_params['_endpoint'].")

        date_field = params.get("_date_field", "record_date")
        value_field = params.get("_value_field", "value")
        fields = params.get("_fields") or params.get("fields") or f"{date_field},{value_field}"
        query_params: dict[str, Any] = {
            "fields": fields,
            "format": params.get("_format", "json"),
            "page[size]": params.get("_page_size", "5000"),
        }

        filters = []
        if start is not None:
            filters.append(f"{date_field}:gte:{start}")
        if end is not None:
            filters.append(f"{date_field}:lte:{end}")
        extra_filter = params.get("_filter") or params.get("filter")
        if extra_filter:
            filters.append(str(extra_filter))
        if filters:
            query_params["filter"] = ",".join(filters)

        sort = params.get("_sort") or params.get("sort") or date_field
        if sort:
            query_params["sort"] = sort

        url = endpoint if _is_url(endpoint) else f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = requests.get(url, params=query_params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", [])
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=["date", "series", "value"])
        if date_field not in frame.columns or value_field not in frame.columns:
            raise ValueError(
                f"Treasury response for {spec.name} must include {date_field!r} and {value_field!r}."
            )

        return pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_field], errors="coerce"),
                "series": spec.name,
                "value": _clean_numeric(frame[value_field]),
            }
        )

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = [
            self._fetch_one(spec, start, end)
            for spec in specs
            if spec.provider.upper() == "TREASURY"
        ]
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.concat(frames, ignore_index=True))


class FedDdpProvider:
    """Fetch Federal Reserve Data Download Program CSV packages."""

    download_url = "https://www.federalreserve.gov/datadownload/Download.aspx"

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        timeout: int = 30,
        credentials: Optional[ProviderCredentials] = None,
    ) -> None:
        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        self.config = config or ProviderConfig(credentials=credentials)
        self.credentials = self.config.credentials
        self.timeout = timeout

    def _request_params(
        self,
        spec: EconomicSeriesSpec,
        start: Optional[str],
        end: Optional[str],
    ) -> tuple[str, Optional[dict[str, Any]]]:
        params = _provider_params(spec)
        source = str(params.get("_url") or spec.provider_code)
        if _is_url(source):
            return source, None

        query_params: dict[str, Any] = {
            "rel": params.get("_release", params.get("rel", "H15")),
            "series": source,
            "filetype": params.get("_filetype", "csv"),
            "label": params.get("_label", "include"),
            "layout": params.get("_layout", "seriescolumn"),
            "type": params.get("_type", "package"),
        }
        if start is not None:
            query_params["from"] = start
        if end is not None:
            query_params["to"] = end
        if start is None and end is None and "_last_obs" in params:
            query_params["lastObs"] = params["_last_obs"]
        return self.download_url, query_params

    @staticmethod
    def _read_ddp_csv(text: str) -> pd.DataFrame:
        lines = text.splitlines()
        header_index = 0
        for index, line in enumerate(lines):
            first_cell = line.split(",", 1)[0].strip().lower()
            if first_cell in {"date", "time period"}:
                header_index = index
                break
        return pd.read_csv(StringIO("\n".join(lines[header_index:])))

    def _fetch_one(
        self,
        spec: EconomicSeriesSpec,
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        params = _provider_params(spec)
        url, request_params = self._request_params(spec, start, end)
        response = requests.get(url, params=request_params, timeout=self.timeout)
        response.raise_for_status()
        frame = self._read_ddp_csv(response.text)
        if frame.empty:
            return pd.DataFrame(columns=["date", "series", "value"])

        date_column = params.get("_date_column")
        if date_column is None:
            date_column = "Time Period" if "Time Period" in frame.columns else frame.columns[0]
        value_column = params.get("_value_column")
        if value_column is None:
            candidates = [column for column in frame.columns if column != date_column]
            if len(candidates) != 1:
                raise ValueError(
                    f"Fed DDP spec {spec.name} requires provider_params['_value_column']."
                )
            value_column = candidates[0]
        if date_column not in frame.columns or value_column not in frame.columns:
            raise ValueError(
                f"Fed DDP response for {spec.name} must include {date_column!r} and {value_column!r}."
            )

        return pd.DataFrame(
            {
                "date": pd.to_datetime(frame[date_column], errors="coerce"),
                "series": spec.name,
                "value": _clean_numeric(frame[value_column]),
            }
        )

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        frames = [
            self._fetch_one(spec, start, end)
            for spec in specs
            if spec.provider.upper() == "FED_DDP"
        ]
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.concat(frames, ignore_index=True))


class ProviderRegistry:
    """Route economic series specs to provider implementations."""

    def __init__(
        self,
        providers: Mapping[str, EconomicSeriesProvider],
        strict: bool = False,
        continue_on_error: bool = False,
    ) -> None:
        self.providers = {name.upper(): provider for name, provider in providers.items()}
        self.strict = strict
        self.continue_on_error = continue_on_error
        self.last_errors: list[ProviderFetchError] = []

    @classmethod
    def default_us(
        cls,
        config: Optional[ProviderConfig] = None,
        credentials: Optional[ProviderCredentials] = None,
        timeout: int = 30,
        strict: bool = False,
        continue_on_error: bool = False,
    ) -> "ProviderRegistry":
        """Build the default U.S. macro/liquidity provider registry."""

        if config is not None and credentials is not None:
            raise ValueError("Pass either config or credentials, not both.")
        shared_config = config or ProviderConfig(credentials=credentials)
        return cls(
            {
                "BEA": BeaApiProvider(config=shared_config, timeout=timeout),
                "BLS": BlsApiProvider(config=shared_config, timeout=timeout),
                "FED_DDP": FedDdpProvider(config=shared_config, timeout=timeout),
                "FRED": FredApiProvider(config=shared_config, timeout=timeout),
                "TREASURY": TreasuryFiscalDataProvider(config=shared_config, timeout=timeout),
            },
            strict=strict,
            continue_on_error=continue_on_error,
        )

    def error_report(self) -> pd.DataFrame:
        """Return provider errors captured by the most recent tolerant fetch."""

        rows = [
            {
                "provider": error.provider,
                "series": ", ".join(error.series),
                "error_type": error.error_type,
                "message": error.message,
            }
            for error in self.last_errors
        ]
        return pd.DataFrame(rows, columns=["provider", "series", "error_type", "message"])

    def fetch(
        self,
        specs: Iterable[EconomicSeriesSpec],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        self.last_errors = []
        grouped: dict[str, list[EconomicSeriesSpec]] = defaultdict(list)
        for spec in specs:
            provider_name = spec.provider.upper()
            if provider_name:
                grouped[provider_name].append(spec)

        frames = []
        missing = []
        for provider_name, provider_specs in grouped.items():
            provider = self.providers.get(provider_name)
            if provider is None:
                missing.append(provider_name)
                continue
            try:
                frame = provider.fetch(provider_specs, start=start, end=end)
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                self.last_errors.append(
                    ProviderFetchError(
                        provider=provider_name,
                        series=tuple(spec.name for spec in provider_specs),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue
            if not frame.empty:
                frames.append(frame)

        if missing and self.strict:
            raise ValueError(f"No provider registered for: {sorted(set(missing))}")
        if not frames:
            return pd.DataFrame(columns=["date", "series", "value"])
        return normalize_series_frame(pd.concat(frames, ignore_index=True))
