"""
数据解析工具模块
"""
from typing import Optional, Union
import logging

from config import YEAR_FILTER_MIN, YEAR_FILTER_MAX

logger = logging.getLogger("GARP_Analyzer.parser")


def _is_likely_year(val: float) -> bool:
    """
    判断数值是否疑似年份
    
    Args:
        val: 待检测的数值
        
    Returns:
        如果数值在年份范围内且为整数，返回 True
    """
    return YEAR_FILTER_MIN < val < YEAR_FILTER_MAX and val % 1 == 0


def parse_percentage(val: Union[str, int, float, None]) -> Optional[float]:
    """
    解析百分比数值
    
    支持多种输入格式：
    1. 数值类型 (int, float, numpy types)
    2. 字符串格式 (带 %, 逗号, unicode 空格)
    3. 自动过滤疑似年份的整数
    
    Args:
        val: 待解析的值
        
    Returns:
        解析后的浮点数，解析失败返回 None
    """
    # 尝试直接转换数值类型
    try:
        f_val = float(val)
        if _is_likely_year(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        pass

    # 处理字符串格式
    if isinstance(val, str):
        # 清理字符串：移除 %, 逗号, unicode 空格
        clean_val = val.replace('%', '').replace(',', '').replace('\xa0', '').strip()
        try:
            f_val = float(clean_val)
            if _is_likely_year(f_val):
                return None
            return f_val
        except ValueError:
            return None
    
    return None


def parse_growth_value(row_values: list, threshold: float = 10.0) -> Optional[float]:
    """
    从数据行中提取增长率数值
    
    Args:
        row_values: 行数据列表
        threshold: 小于此值时认为是小数形式，需要乘以100
        
    Returns:
        解析后的增长率百分比，解析失败返回 None
    """
    for col_val in row_values:
        v = parse_percentage(col_val)
        if v is not None:
            # 如果值小于阈值，认为是小数形式（如 0.15 表示 15%）
            if abs(v) < threshold:
                return v * 100
            return v
    return None


def match_growth_row(row_text: str, next_year: int, year_after_next: int) -> tuple[bool, bool]:
    """
    匹配增长率数据行
    
    Args:
        row_text: 行文本（小写）
        next_year: 明年年份
        year_after_next: 后年年份
        
    Returns:
        (是否匹配明年, 是否匹配后两年)
    """
    row_lower = row_text.lower()
    
    # 匹配明年: "+1y" 或 "next year" 或具体年份
    is_next_year = (
        '+1y' in row_lower or 
        'next year' in row_lower or 
        str(next_year) in row_lower
    )
    
    # 匹配后两年: "+2y" 或 "next 2 years" 或具体年份
    is_next_2y = (
        '+2y' in row_lower or 
        'next 2 years' in row_lower or 
        str(year_after_next) in row_lower
    )
    
    return is_next_year, is_next_2y

