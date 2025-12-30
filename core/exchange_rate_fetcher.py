"""
汇率获取模块
从 Yahoo Finance 获取最新汇率
"""
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import yfinance as yf
from config import CACHE_TTL_SECONDS

logger = logging.getLogger("GARP_Analyzer.exchange_rate")

# 汇率缓存
_rate_cache: Optional[Dict[str, Optional[float]]] = None
_cache_timestamp: Optional[datetime] = None


def fetch_cny_usd_rate(use_cache: bool = True) -> Optional[float]:
    """
    获取 CNY 兑 USD 的最新汇率 (1 CNY = ? USD)
    
    Args:
        use_cache: 是否使用缓存
    
    Returns:
        CNY/USD 汇率，如果获取失败返回 None
    """
    if use_cache:
        rates = fetch_all_rates()
        return rates.get('cny_usd')
    
    try:
        ticker = yf.Ticker('CNYUSD=X')
        info = ticker.info
        rate = info.get('regularMarketPrice') or info.get('previousClose')
        if rate:
            logger.debug(f"获取 CNY/USD 汇率: {rate}")
            return float(rate)
        else:
            logger.warning("CNY/USD 汇率数据为空")
            return None
    except Exception as e:
        logger.error(f"获取 CNY/USD 汇率失败: {e}", exc_info=True)
        return None


def fetch_usd_cny_rate(use_cache: bool = True) -> Optional[float]:
    """
    获取 USD 兑 CNY 的最新汇率 (1 USD = ? CNY)
    
    Args:
        use_cache: 是否使用缓存
    
    Returns:
        USD/CNY 汇率，如果获取失败返回 None
    """
    if use_cache:
        rates = fetch_all_rates()
        return rates.get('usd_cny')
    
    try:
        ticker = yf.Ticker('USDCNY=X')
        info = ticker.info
        rate = info.get('regularMarketPrice') or info.get('previousClose')
        if rate:
            logger.debug(f"获取 USD/CNY 汇率: {rate}")
            return float(rate)
        else:
            logger.warning("USD/CNY 汇率数据为空")
            return None
    except Exception as e:
        logger.error(f"获取 USD/CNY 汇率失败: {e}", exc_info=True)
        return None


def _fetch_all_rates_from_api() -> Dict[str, Optional[float]]:
    """
    从 API 获取所有相关汇率（内部函数，不检查缓存）
    
    Returns:
        包含所有汇率的字典
    """
    rates = {}
    
    # CNY/USD
    cny_usd = fetch_cny_usd_rate(use_cache=False)
    rates['cny_usd'] = cny_usd
    
    # USD/CNY
    usd_cny = fetch_usd_cny_rate(use_cache=False)
    rates['usd_cny'] = usd_cny
    
    # 如果 USD/CNY 获取失败，尝试从 CNY/USD 计算
    if usd_cny is None and cny_usd is not None and cny_usd > 0:
        rates['usd_cny'] = 1.0 / cny_usd
        logger.info(f"通过 CNY/USD 计算 USD/CNY: {rates['usd_cny']}")
    
    # 如果 CNY/USD 获取失败，尝试从 USD/CNY 计算
    if cny_usd is None and usd_cny is not None and usd_cny > 0:
        rates['cny_usd'] = 1.0 / usd_cny
        logger.info(f"通过 USD/CNY 计算 CNY/USD: {rates['cny_usd']}")
    
    # HKD/CNY (可选)
    try:
        ticker = yf.Ticker('HKDCNY=X')
        info = ticker.info
        hkd_cny = info.get('regularMarketPrice') or info.get('previousClose')
        if hkd_cny:
            rates['hkd_cny'] = float(hkd_cny)
            # 计算 CNY/HKD
            if hkd_cny > 0:
                rates['cny_hkd'] = 1.0 / float(hkd_cny)
    except Exception as e:
        logger.debug(f"获取 HKD/CNY 汇率失败: {e}")
    
    # USD/HKD (可选)
    try:
        ticker = yf.Ticker('USDHKD=X')
        info = ticker.info
        usd_hkd = info.get('regularMarketPrice') or info.get('previousClose')
        if usd_hkd:
            rates['usd_hkd'] = float(usd_hkd)
            # 计算 HKD/USD
            if usd_hkd > 0:
                rates['hkd_usd'] = 1.0 / float(usd_hkd)
    except Exception as e:
        logger.debug(f"获取 USD/HKD 汇率失败: {e}")
    
    return rates


def fetch_all_rates(force_refresh: bool = False) -> Dict[str, Optional[float]]:
    """
    获取所有相关汇率（带缓存）
    
    Args:
        force_refresh: 是否强制刷新缓存
    
    Returns:
        包含所有汇率的字典:
        - cny_usd: 1 CNY = ? USD
        - usd_cny: 1 USD = ? CNY
        - cny_hkd: 1 CNY = ? HKD (如果可用)
        - hkd_cny: 1 HKD = ? CNY (如果可用)
        - usd_hkd: 1 USD = ? HKD (如果可用)
        - hkd_usd: 1 HKD = ? USD (如果可用)
    """
    global _rate_cache, _cache_timestamp
    
    # 检查缓存是否有效
    if not force_refresh and _rate_cache is not None and _cache_timestamp is not None:
        elapsed = (datetime.now() - _cache_timestamp).total_seconds()
        if elapsed < CACHE_TTL_SECONDS:
            logger.debug(f"使用缓存的汇率数据 (缓存时间: {elapsed:.1f}秒, TTL: {CACHE_TTL_SECONDS}秒)")
            return _rate_cache.copy()
        else:
            logger.debug(f"缓存已过期 (已过 {elapsed:.1f}秒，TTL: {CACHE_TTL_SECONDS}秒)，重新获取")
    
    # 从 API 获取新数据
    rates = _fetch_all_rates_from_api()
    
    # 更新缓存
    _rate_cache = rates.copy()
    _cache_timestamp = datetime.now()
    
    return rates

