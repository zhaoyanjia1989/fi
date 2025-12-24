"""
GARP 计算模块
"""
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("GARP_Analyzer.calculator")


@dataclass
class GrowthResult:
    """增长率计算结果"""
    value: Optional[float]
    source: str


def calculate_garp(
    forward_pe: Optional[float],
    peg_ratio: Optional[float],
    calc_growth: Optional[float]
) -> Optional[float]:
    """
    计算 GARP 值
    
    GARP = Forward PE / Growth Rate
    
    Args:
        forward_pe: 远期市盈率
        peg_ratio: PEG 比率（优先使用）
        calc_growth: 计算得到的增长率
        
    Returns:
        GARP 值，计算失败返回 None
    """
    # 优先使用原始 PEG
    if peg_ratio is not None and peg_ratio > 0:
        return peg_ratio
    
    # 否则手动计算
    if forward_pe and calc_growth and calc_growth > 0:
        return forward_pe / calc_growth
    
    return None


def determine_growth_rate(
    next_2y_growth: Optional[float],
    next_year_growth: Optional[float],
    forward_pe: Optional[float],
    trailing_pe: Optional[float],
    peg_ratio: Optional[float],
    revenue_growth: Optional[float],
    earnings_growth: Optional[float],
) -> GrowthResult:
    """
    根据优先级策略确定核心增长率
    
    优先级 (分析师预测优先):
    P1: 后两年预测 (分析师长期预测)
    P2: 明年预测 (分析师短期预测)
    P3: PEG 反推
    P4: 市场隐含增长（PE差，作为备选）
    P5: 营收增长
    P6: 季度盈利增长
    
    Args:
        next_2y_growth: 后两年增长预测
        next_year_growth: 明年增长预测
        forward_pe: 远期市盈率
        trailing_pe: 滚动市盈率
        peg_ratio: PEG 比率
        revenue_growth: 营收增长率（已转为百分比）
        earnings_growth: 盈利增长率（已转为百分比）
        
    Returns:
        GrowthResult 包含增长率值和来源说明
    """
    # P1: 长期预测（后两年）
    if next_2y_growth and next_2y_growth > 0:
        return GrowthResult(value=next_2y_growth, source="P1: 预测(+2y)")
    
    # P2: 明年预测 (分析师预测优先于市场隐含)
    if next_year_growth and next_year_growth > 0:
        return GrowthResult(value=next_year_growth, source="P2: 预测(+1y)")
    
    # P3: PEG 反推
    if forward_pe and peg_ratio and peg_ratio > 0:
        implied = forward_pe / peg_ratio
        if implied > 0:
            return GrowthResult(value=implied, source="P3: PEG反推")
    
    # P4: 市场隐含增长（基于 PE 差异，作为备选）
    if forward_pe and trailing_pe and trailing_pe > forward_pe:
        implied_market_growth = (trailing_pe / forward_pe - 1) * 100
        if implied_market_growth > 0:
            return GrowthResult(value=implied_market_growth, source="P4: PE隐含")
    
    # P5: 营收增长
    if revenue_growth and revenue_growth > 0:
        return GrowthResult(value=revenue_growth, source="P5: 营收")
    
    # P6: 季度盈利
    if earnings_growth and earnings_growth > 0:
        return GrowthResult(value=earnings_growth, source="P6: 季度盈利")
    
    return GrowthResult(value=None, source="-")

