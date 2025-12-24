"""
CME 金属交割通知数据获取模块
"""
import requests
import pdfplumber
import re
import io
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("GARP_Analyzer.metals")

CME_METALS_PDF_URL = "https://www.cmegroup.com/delivery_reports/MetalsIssuesAndStopsMTDReport.pdf"


@dataclass
class DeliveryRecord:
    """单日交割记录"""
    date: str
    daily: int
    cumulative: int


@dataclass
class ContractDelivery:
    """合约交割数据"""
    contract_name: str
    records: List[DeliveryRecord]
    
    @property
    def latest_daily(self) -> int:
        """最新日交割量"""
        return self.records[-1].daily if self.records else 0
    
    @property
    def total_cumulative(self) -> int:
        """累计交割量"""
        return self.records[-1].cumulative if self.records else 0
    
    @property
    def recent_7d(self) -> List[int]:
        """最近7天日交割量"""
        return [r.daily for r in self.records[-7:]]
    
    @property
    def latest_date(self) -> str:
        """最新日期 (MM/DD格式)"""
        if self.records:
            # 转换 12/23/2025 -> 12/23
            full_date = self.records[-1].date
            parts = full_date.split('/')
            return f"{parts[0]}/{parts[1]}"
        return "-"
    
    @property
    def recent_7d_with_dates(self) -> List[tuple[str, int]]:
        """最近7天日交割量带日期"""
        result = []
        for r in self.records[-7:]:
            # 转换 12/23/2025 -> 12/23
            parts = r.date.split('/')
            short_date = f"{parts[0]}/{parts[1]}"
            result.append((short_date, r.daily))
        return result


def fetch_metals_delivery() -> Dict[str, ContractDelivery]:
    """
    获取 CME 金属交割通知数据
    
    Returns:
        {合约名: ContractDelivery} 字典
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }
    
    try:
        resp = requests.get(CME_METALS_PDF_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        
        results = {}
        current_contract = None
        current_records = []
        
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                for line in lines:
                    # 匹配合约名称
                    if 'CONTRACT:' in line:
                        # 保存上一个合约
                        if current_contract and current_records:
                            results[current_contract] = ContractDelivery(
                                contract_name=current_contract,
                                records=current_records
                            )
                        
                        # 提取新合约名
                        match = re.search(r'CONTRACT:\s*(.+)', line)
                        if match:
                            current_contract = match.group(1).strip()
                            current_records = []
                    
                    # 匹配交割数据行 (日期 日交割量 累计)
                    # 格式: 12/23/2025 417 36,498
                    data_match = re.match(r'(\d{2}/\d{2}/\d{4})\s+([\d,]+)\s+([\d,]+)', line)
                    if data_match and current_contract:
                        date_str = data_match.group(1)
                        daily = int(data_match.group(2).replace(',', ''))
                        cumulative = int(data_match.group(3).replace(',', ''))
                        current_records.append(DeliveryRecord(
                            date=date_str,
                            daily=daily,
                            cumulative=cumulative
                        ))
                
                # 保存最后一个合约
                if current_contract and current_records:
                    results[current_contract] = ContractDelivery(
                        contract_name=current_contract,
                        records=current_records
                    )
        
        return results
        
    except Exception as e:
        logger.error(f"获取 CME 金属交割数据失败: {e}")
        return {}


def get_gold_silver_delivery() -> tuple[Optional[ContractDelivery], Optional[ContractDelivery]]:
    """
    获取黄金和白银交割数据
    
    Returns:
        (黄金数据, 白银数据)
    """
    data = fetch_metals_delivery()
    
    gold = None
    silver = None
    
    for name, delivery in data.items():
        if 'GOLD' in name.upper() and '100' in name:
            gold = delivery
        elif 'SILVER' in name.upper() and '5000' in name:
            silver = delivery
    
    return gold, silver

