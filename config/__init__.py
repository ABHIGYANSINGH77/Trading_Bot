"""Configuration loader and validator."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG = CONFIG_DIR / "settings.yaml"
TEMPLATE_CONFIG = CONFIG_DIR / "settings_template.yaml"


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from YAML file.

    Falls back to template if settings.yaml doesn't exist.
    """
    if path:
        config_path = Path(path)
    elif DEFAULT_CONFIG.exists():
        config_path = DEFAULT_CONFIG
    elif TEMPLATE_CONFIG.exists():
        config_path = TEMPLATE_CONFIG
    else:
        raise FileNotFoundError("No configuration file found. Copy settings_template.yaml to settings.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create directories if they don't exist
    cache_dir = config.get("data", {}).get("cache_dir", "./data/cache")
    os.makedirs(cache_dir, exist_ok=True)

    log_file = config.get("logging", {}).get("log_file", "./logs/quantbot.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    return config


@dataclass
class BrokerConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: str = ""
    timeout: int = 30


@dataclass
class RiskConfig:
    max_position_pct: float = 0.20
    max_portfolio_leverage: float = 1.0
    max_daily_drawdown_pct: float = 0.02
    max_total_drawdown_pct: float = 0.10
    max_open_orders: int = 10
    max_correlation: float = 0.85


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_pct: float = 0.001
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"


def parse_broker_config(config: dict) -> BrokerConfig:
    broker = config.get("broker", {})
    return BrokerConfig(**{k: v for k, v in broker.items() if k in BrokerConfig.__dataclass_fields__})


def parse_risk_config(config: dict) -> RiskConfig:
    risk = config.get("risk", {})
    return RiskConfig(**{k: v for k, v in risk.items() if k in RiskConfig.__dataclass_fields__})


def parse_backtest_config(config: dict) -> BacktestConfig:
    bt = config.get("backtest", {})
    return BacktestConfig(**{k: v for k, v in bt.items() if k in BacktestConfig.__dataclass_fields__})
