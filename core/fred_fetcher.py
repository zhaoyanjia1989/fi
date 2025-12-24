"""
FRED (Federal Reserve Economic Data) 数据获取模块
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("GARP_Analyzer.fred")

# FRED API 配置
FRED_API_KEY = "f20587014612fe02052fd203f38f0c9a"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# 美联储相关指标
FED_INDICATORS = {
    "SOFR": "SOFR",
    "IORB": "IORB",
    "EFFR": "EFFR",
    "DFEDTARU": "利率上限",
    "DFEDTARL": "利率下限",
    "WTREGEN": "TGA余额",
    "WRESBAL": "银行准备金",
}


@dataclass
class FredObservation:
    """单个观测值"""
    date: str
    value: Optional[float]


@dataclass
class FredSeries:
    """指标数据序列"""
    series_id: str
    name: str
    observations: List[FredObservation]
    
    @property
    def latest_value(self) -> Optional[float]:
        """获取最新值"""
        for obs in self.observations:
            if obs.value is not None:
                return obs.value
        return None
    
    @property
    def latest_date(self) -> Optional[str]:
        """获取最新日期"""
        for obs in self.observations:
            if obs.value is not None:
                return obs.date
        return None


def fetch_fred_series(
    series_id: str,
    days: int = 30,
    limit: int = 7
) -> Optional[FredSeries]:
    """
    获取单个 FRED 指标数据
    
    Args:
        series_id: FRED 指标代码
        days: 获取最近多少天的数据
        limit: 最多返回多少条记录
        
    Returns:
        FredSeries 对象，失败返回 None
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date.strftime("%Y-%m-%d"),
        "observation_end": end_date.strftime("%Y-%m-%d"),
        "sort_order": "desc",
        "limit": limit,
    }
    
    try:
        resp = requests.get(FRED_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "observations" not in data:
            logger.warning(f"[{series_id}] 无观测数据")
            return None
        
        observations = []
        for obs in data["observations"]:
            value = None
            if obs["value"] != ".":
                try:
                    value = float(obs["value"])
                except ValueError:
                    pass
            observations.append(FredObservation(
                date=obs["date"],
                value=value
            ))
        
        name = FED_INDICATORS.get(series_id, series_id)
        return FredSeries(
            series_id=series_id,
            name=name,
            observations=observations
        )
        
    except requests.RequestException as e:
        logger.error(f"[{series_id}] 请求失败: {e}")
        return None
    except Exception as e:
        logger.error(f"[{series_id}] 解析失败: {e}")
        return None


def fetch_fed_indicators(days: int = 30) -> Dict[str, FredSeries]:
    """
    获取所有美联储相关指标
    
    Args:
        days: 获取最近多少天的数据
        
    Returns:
        {series_id: FredSeries} 字典
    """
    results = {}
    
    for series_id in FED_INDICATORS:
        series = fetch_fred_series(series_id, days=days)
        if series:
            results[series_id] = series
    
    return results

