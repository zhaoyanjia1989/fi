"""
GARP 分析器核心模块
"""
from .parser import parse_percentage
from .calculator import calculate_garp, determine_growth_rate
from .data_fetcher import fetch_stock_data, StockData
from .exchange_rate_fetcher import fetch_cny_usd_rate, fetch_usd_cny_rate, fetch_all_rates

__all__ = [
    'parse_percentage',
    'calculate_garp',
    'determine_growth_rate',
    'fetch_stock_data',
    'StockData',
    'fetch_cny_usd_rate',
    'fetch_usd_cny_rate',
    'fetch_all_rates',
]

