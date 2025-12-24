"""
GARP 分析器核心模块
"""
from .parser import parse_percentage
from .calculator import calculate_garp, determine_growth_rate
from .data_fetcher import fetch_stock_data, StockData

__all__ = [
    'parse_percentage',
    'calculate_garp',
    'determine_growth_rate',
    'fetch_stock_data',
    'StockData',
]

