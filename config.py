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
    "BABA", "TCEHY", "3690.HK", "1810.HK", "3750.HK", "XPEV",
    "SE", "NVDA", "GOOGL", "NFLX"
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

# 汇率配置
# 人民币兑美元: 1 USD = 7 CNY (1 CNY = 1/7 USD)
USD_CNY_RATE: float = 7.0
CNY_USD_RATE: float = 1.0 / 7.0  # 1 CNY = 0.1429 USD

# 人民币兑港币: 1 CNY = 1.11 HKD (1 HKD = 1/1.11 CNY)
CNY_HKD_RATE: float = 1.11
HKD_CNY_RATE: float = 1.0 / 1.11  # 1 HKD = 0.9009 CNY

# 美元兑港币: 通过人民币计算 1 USD = 7 CNY = 7 * 1.11 HKD = 7.77 HKD
USD_HKD_RATE: float = USD_CNY_RATE * CNY_HKD_RATE  # 1 USD = 7.77 HKD
HKD_USD_RATE: float = 1.0 / USD_HKD_RATE  # 1 HKD = 0.1287 USD

