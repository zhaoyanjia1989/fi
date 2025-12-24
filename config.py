"""
GARP 分析器配置文件
"""
from dataclasses import dataclass
from typing import Dict, List

# 港股转美股映射表
HK_TO_US_MAP: Dict[str, str] = {
    '9988.HK': 'BABA',   # 阿里巴巴
    '0700.HK': 'TCEHY',  # 腾讯控股
    '9868.HK': 'XPEV',   # 小鹏汽车
    '2015.HK': 'LI',     # 理想汽车
    '9866.HK': 'NIO',    # 蔚来
    '9618.HK': 'JD',     # 京东
    '9888.HK': 'BIDU',   # 百度
    '9626.HK': 'BILI',   # B站
    '9999.HK': 'NTES',   # 网易
    '0992.HK': 'LNVGY',  # 联想集团
}

# 默认股票列表
DEFAULT_TICKERS: List[str] = [
    "BABA", "TCEHY", "3690.HK", "1810.HK", "1024.HK", "XPEV",
    "SE", "NVDA", "GOOGL"
]


@dataclass
class GARPThresholds:
    """GARP 评估阈值配置"""
    undervalued: float = 0.75      # 低估阈值
    fair_upper: float = 1.25       # 合理估值上限
    min_growth: float = 0.0        # 最低增长率
    
    def evaluate(self, growth: float | None, garp: float | None) -> str:
        """评估股票状态"""
        if growth is None or growth <= self.min_growth:
            return "动力不足"
        if garp is None:
            return "数据缺失"
        if garp < self.undervalued:
            return "低估"
        if garp <= self.fair_upper:
            return "合理"
        return "偏高"


# 评价颜色映射
STATUS_COLORS: Dict[str, str] = {
    "低估": "#2E7D32",
    "合理": "#1565C0",
    "偏高": "#C62828",
    "动力不足": "#757575",
    "数据缺失": "#9E9E9E",
}

# 缓存配置
CACHE_TTL_SECONDS: int = 300

# 年份过滤范围（用于过滤疑似年份的数值）
YEAR_FILTER_MIN: int = 1900
YEAR_FILTER_MAX: int = 2100

