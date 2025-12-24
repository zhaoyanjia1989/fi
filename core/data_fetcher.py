"""
股票数据获取模块
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import logging

import yfinance as yf
import pandas as pd

from config import HK_TO_US_MAP
from .parser import parse_growth_value, match_growth_row

logger = logging.getLogger("GARP_Analyzer.fetcher")


@dataclass
class StockData:
    """股票数据结构"""
    ticker: str
    display_symbol: str
    name: str
    current_price: Optional[float] = None
    forward_pe: Optional[float] = None
    trailing_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    earnings_growth: Optional[float] = None  # 已转为百分比
    revenue_growth: Optional[float] = None   # 已转为百分比
    next_year_growth: Optional[float] = None
    next_2y_growth: Optional[float] = None
    # 财年对齐数据
    fy_next_pe: Optional[float] = None       # 下一财年PE (与增长率时间对齐)
    fy_next_growth: Optional[float] = None   # 下一财年增长率 (来自earnings_estimate +1y)
    fy_current_growth: Optional[float] = None # 当前财年增长率 (来自earnings_estimate 0y)
    analyst_count: Optional[int] = None      # 分析师数量
    # 财年信息 (0y = 当前财年, +1y = 下一财年)
    fy_current_name: Optional[str] = None    # 当前财年名称，如 "FY2026"
    fy_next_name: Optional[str] = None       # 下一财年名称，如 "FY2027"
    fy_current_range: Optional[str] = None   # 当前财年时间范围
    fy_next_range: Optional[str] = None      # 下一财年时间范围
    # PE SD Band (当前PE相对历史均值的标准差倍数)
    pe_sd_90d: Optional[float] = None        # 90天 PE SD Band
    pe_sd_180d: Optional[float] = None       # 180天 PE SD Band
    # EPS 范围 (用于计算股价预估范围, low/avg/high 的比例)
    eps_0y_low_ratio: Optional[float] = None   # 0y EPS_low / EPS_avg
    eps_0y_high_ratio: Optional[float] = None  # 0y EPS_high / EPS_avg
    eps_1y_low_ratio: Optional[float] = None   # +1y EPS_low / EPS_avg
    eps_1y_high_ratio: Optional[float] = None  # +1y EPS_high / EPS_avg
    # 估值相关数据
    beta: Optional[float] = None               # Beta 系数
    fcf_per_share: Optional[float] = None      # 每股自由现金流
    target_mean_price: Optional[float] = None  # 分析师目标均价
    target_low_price: Optional[float] = None   # 分析师目标低价
    target_high_price: Optional[float] = None  # 分析师目标高价
    price_to_sales: Optional[float] = None     # P/S 市销率
    price_to_book: Optional[float] = None      # P/B 市净率
    ev_to_ebitda: Optional[float] = None       # EV/EBITDA
    growth_estimates_raw: Any = None  # 用于调试
    error: Optional[str] = None


@dataclass
class FetchResult:
    """数据获取结果"""
    data: List[StockData] = field(default_factory=list)
    failed_tickers: List[str] = field(default_factory=list)
    fetch_time: datetime.datetime = field(default_factory=datetime.datetime.now)


def _get_display_symbol(ticker: str) -> str:
    """获取显示用的股票代码"""
    original = next((k for k, v in HK_TO_US_MAP.items() if v == ticker), ticker)
    if original != ticker:
        return f"{ticker} (原{original})"
    return ticker


def _calculate_pe_sd_band(
    stock: yf.Ticker,
    trailing_pe: Optional[float],
    trailing_eps: Optional[float],
    days: int
) -> Optional[float]:
    """
    计算 PE SD Band (当前PE相对历史均值的标准差倍数)
    
    Args:
        stock: yfinance Ticker 对象
        trailing_pe: 当前 PE TTM
        trailing_eps: Trailing EPS
        days: 历史天数 (90 或 180)
        
    Returns:
        SD Band 值 (如 +0.5 表示高于均值0.5个标准差)
    """
    if not trailing_pe or not trailing_eps or trailing_eps <= 0:
        return None
    
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty or len(hist) < 20:  # 至少需要20个数据点
            return None
        
        # 用历史收盘价和当前 EPS 计算历史 PE
        hist_pe = hist['Close'] / trailing_eps
        
        pe_mean = hist_pe.mean()
        pe_std = hist_pe.std()
        
        if pe_std <= 0:
            return None
        
        # 计算当前 PE 相对于均值的标准差倍数
        sd_band = (trailing_pe - pe_mean) / pe_std
        return sd_band
        
    except Exception as e:
        logger.debug(f"计算 PE SD Band ({days}天) 失败: {e}")
        return None


def _parse_growth_estimates(
    stock: yf.Ticker,
    ticker: str,
    next_year_int: int,
    year_after_next_int: int
) -> tuple[Optional[float], Optional[float], Any]:
    """
    解析增长预测数据
    
    Returns:
        (明年增长率, 后两年增长率, 原始数据表)
    """
    next_year_growth = None
    next_2y_growth = None
    growth_est_raw = None
    
    try:
        growth_estimates = stock.growth_estimates
        growth_est_raw = growth_estimates
        
        if growth_estimates is None or growth_estimates.empty:
            logger.warning(f"[{ticker}] Growth Estimates 表为空")
            return None, None, None
        
        logger.info(f"[{ticker}] 抓取到表 (Rows: {len(growth_estimates)})")
        df_check = growth_estimates.reset_index()
        
        for idx, row in df_check.iterrows():
            row_vals_str = [str(x) for x in row.values]
            row_text = " ".join(row_vals_str)
            
            # 提取数值
            found_val = parse_growth_value(list(row.values))
            
            # 匹配行类型
            is_next_year, is_next_2y = match_growth_row(
                row_text, next_year_int, year_after_next_int
            )
            
            # 日志记录
            if is_next_year or is_next_2y:
                match_info = f"匹配 (+1y={is_next_year}, +2y={is_next_2y})"
                if found_val is not None:
                    logger.info(f"[{ticker}] Row {idx}: {match_info} -> ✅ {found_val:.2f}%")
                else:
                    logger.info(f"[{ticker}] Row {idx}: {match_info} -> ⚠️ 无数字")
            
            # 赋值
            if found_val is not None:
                if is_next_year and next_year_growth is None:
                    next_year_growth = found_val
                if is_next_2y and next_2y_growth is None:
                    next_2y_growth = found_val
                    
    except AttributeError as e:
        logger.warning(f"[{ticker}] growth_estimates 属性不可用: {e}")
    except Exception as e:
        logger.error(f"[{ticker}] 解析预测表错误: {e}", exc_info=True)
    
    return next_year_growth, next_2y_growth, growth_est_raw


def fetch_single_stock(
    ticker: str,
    next_year_int: int,
    year_after_next_int: int
) -> StockData:
    """
    获取单只股票数据
    
    Args:
        ticker: 股票代码
        next_year_int: 明年年份
        year_after_next_int: 后年年份
        
    Returns:
        StockData 对象
    """
    display_symbol = _get_display_symbol(ticker)
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 基础信息
        name = info.get('shortName', ticker)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # 估值数据
        forward_pe = info.get('forwardPE')
        trailing_pe = info.get('trailingPE')
        peg_ratio = info.get('pegRatio')
        
        # 增长数据（转为百分比）
        earnings_growth_raw = info.get('earningsGrowth')
        revenue_growth_raw = info.get('revenueGrowth')
        
        earnings_growth = earnings_growth_raw * 100 if earnings_growth_raw is not None else None
        revenue_growth = revenue_growth_raw * 100 if revenue_growth_raw is not None else None
        
        # 获取分析师预测
        next_year_growth, next_2y_growth, growth_est_raw = _parse_growth_estimates(
            stock, ticker, next_year_int, year_after_next_int
        )
        
        # 获取财年信息
        # nextFiscalYearEnd = 当前财年(0y)的结束日期
        fy_current_name = None
        fy_next_name = None
        fy_current_range = None
        fy_next_range = None
        next_fy_ts = info.get('nextFiscalYearEnd')
        if next_fy_ts:
            next_fy = datetime.datetime.fromtimestamp(next_fy_ts)
            fy_end_month = next_fy.month
            fy_end_day = next_fy.day
            fy_end_year = next_fy.year
            
            # 当前财年(0y): 结束于 nextFiscalYearEnd
            if fy_end_month == 12:
                # 12月财年: FY2025 = 2025-01-01 ~ 2025-12-31
                fy_current_name = f"FY{fy_end_year}"
                fy_current_range = f"{fy_end_year}-01-01 ~ {fy_end_year}-12-31"
                fy_next_name = f"FY{fy_end_year + 1}"
                fy_next_range = f"{fy_end_year + 1}-01-01 ~ {fy_end_year + 1}-12-31"
            else:
                # 非12月财年: 如BABA 3月财年, nextFiscalYearEnd=2026-03-31
                # 0y = FY2026 = 2025-04-01 ~ 2026-03-31
                fy_current_name = f"FY{fy_end_year}"
                fy_start_year = fy_end_year - 1
                fy_current_range = f"{fy_start_year}-{fy_end_month + 1:02d}-01 ~ {fy_end_year}-{fy_end_month:02d}-{fy_end_day:02d}"
                # +1y = FY2027 = 2026-04-01 ~ 2027-03-31
                fy_next_name = f"FY{fy_end_year + 1}"
                fy_next_range = f"{fy_end_year}-{fy_end_month + 1:02d}-01 ~ {fy_end_year + 1}-{fy_end_month:02d}-{fy_end_day:02d}"
        
        # 获取财年对齐的 PE 和增长率 (来自 earnings_estimate)
        fy_next_pe = None
        fy_next_growth = None
        fy_current_growth = None
        analyst_count = None
        eps_0y_low_ratio = None
        eps_0y_high_ratio = None
        eps_1y_low_ratio = None
        eps_1y_high_ratio = None
        try:
            ee = stock.earnings_estimate
            if ee is not None and not ee.empty:
                # 获取当前财年增长率和分析师数量
                if '0y' in ee.index:
                    growth_0y = ee.loc['0y', 'growth']
                    if growth_0y is not None:
                        fy_current_growth = growth_0y * 100
                    if 'numberOfAnalysts' in ee.columns:
                        analyst_count = int(ee.loc['0y', 'numberOfAnalysts'])
                    # 获取 0y EPS 范围比例
                    eps_0y_avg = ee.loc['0y', 'avg']
                    eps_0y_low = ee.loc['0y', 'low']
                    eps_0y_high = ee.loc['0y', 'high']
                    if eps_0y_avg and eps_0y_avg > 0:
                        if eps_0y_low:
                            eps_0y_low_ratio = eps_0y_low / eps_0y_avg
                        if eps_0y_high:
                            eps_0y_high_ratio = eps_0y_high / eps_0y_avg
                
                if '+1y' in ee.index:
                    # 获取下一财年的 EPS 预测和增长率
                    eps_next = ee.loc['+1y', 'avg']
                    eps_0y = ee.loc['0y', 'avg'] if '0y' in ee.index else None
                    growth_next = ee.loc['+1y', 'growth']
                    forward_eps_info = info.get('forwardEps')
                    
                    # 获取 +1y EPS 范围比例
                    eps_1y_low = ee.loc['+1y', 'low']
                    eps_1y_high = ee.loc['+1y', 'high']
                    if eps_next and eps_next > 0:
                        if eps_1y_low:
                            eps_1y_low_ratio = eps_1y_low / eps_next
                        if eps_1y_high:
                            eps_1y_high_ratio = eps_1y_high / eps_next
                    
                    # 更新分析师数量为 +1y 的
                    if 'numberOfAnalysts' in ee.columns:
                        analyst_count = int(ee.loc['+1y', 'numberOfAnalysts'])
                    
                    if eps_next and eps_next > 0 and current_price:
                        if forward_eps_info and forward_eps_info > 0:
                            # 判断 Forward EPS 更接近 0y 还是 +1y
                            # 用于决定是否需要货币换算
                            ratio_to_0y = abs(eps_0y / forward_eps_info - 1) if eps_0y and eps_0y > 0 else float('inf')
                            ratio_to_1y = abs(eps_next / forward_eps_info - 1)
                            
                            if ratio_to_1y < 0.15:
                                # Forward EPS ≈ +1y EPS，同货币，直接用
                                fy_next_pe = current_price / eps_next
                                logger.debug(f"[{ticker}] Forward EPS ≈ +1y EPS, 直接计算 +1y PE = {fy_next_pe:.2f}")
                            elif ratio_to_0y < 0.15 and eps_0y and eps_0y > 0:
                                # Forward EPS ≈ 0y EPS，同货币，用增长率推算
                                fy_next_pe = current_price / eps_next
                                logger.debug(f"[{ticker}] Forward EPS ≈ 0y EPS, 直接计算 +1y PE = {fy_next_pe:.2f}")
                            elif eps_0y and eps_0y > 0:
                                # 货币不同（如 BABA），用比例换算
                                currency_ratio = eps_0y / forward_eps_info
                                eps_next_converted = eps_next / currency_ratio
                                fy_next_pe = current_price / eps_next_converted
                                logger.debug(f"[{ticker}] 货币换算 ratio={currency_ratio:.2f}, +1y PE = {fy_next_pe:.2f}")
                            else:
                                # 0y EPS 不可用，用增长率反推
                                if forward_pe and growth_next and growth_next > 0:
                                    fy_next_pe = forward_pe / (1 + growth_next)
                        elif forward_pe and growth_next and growth_next > 0:
                            # Forward EPS 不可用，用增长率反推
                            fy_next_pe = forward_pe / (1 + growth_next)
                        
                    if growth_next is not None:
                        fy_next_growth = growth_next * 100  # 转为百分比
        except Exception as e:
            logger.warning(f"[{ticker}] 获取 earnings_estimate 失败: {e}")
        
        # 计算 PE SD Band
        trailing_eps = info.get('trailingEps')
        pe_sd_90d = _calculate_pe_sd_band(stock, trailing_pe, trailing_eps, 90)
        pe_sd_180d = _calculate_pe_sd_band(stock, trailing_pe, trailing_eps, 180)
        
        # 获取估值相关数据
        beta = info.get('beta')
        fcf = info.get('freeCashflow')
        shares = info.get('sharesOutstanding')
        fcf_per_share = fcf / shares if fcf and shares and shares > 0 else None
        
        target_mean_price = info.get('targetMeanPrice')
        target_low_price = info.get('targetLowPrice')
        target_high_price = info.get('targetHighPrice')
        
        price_to_sales = info.get('priceToSalesTrailing12Months')
        price_to_book = info.get('priceToBook')
        ev_to_ebitda = info.get('enterpriseToEbitda')
        
        return StockData(
            ticker=ticker,
            display_symbol=display_symbol,
            name=name,
            current_price=current_price,
            forward_pe=forward_pe,
            trailing_pe=trailing_pe,
            peg_ratio=peg_ratio,
            earnings_growth=earnings_growth,
            revenue_growth=revenue_growth,
            next_year_growth=next_year_growth,
            next_2y_growth=next_2y_growth,
            fy_next_pe=fy_next_pe,
            fy_next_growth=fy_next_growth,
            fy_current_growth=fy_current_growth,
            analyst_count=analyst_count,
            fy_current_name=fy_current_name,
            fy_next_name=fy_next_name,
            fy_current_range=fy_current_range,
            fy_next_range=fy_next_range,
            pe_sd_90d=pe_sd_90d,
            pe_sd_180d=pe_sd_180d,
            eps_0y_low_ratio=eps_0y_low_ratio,
            eps_0y_high_ratio=eps_0y_high_ratio,
            eps_1y_low_ratio=eps_1y_low_ratio,
            eps_1y_high_ratio=eps_1y_high_ratio,
            beta=beta,
            fcf_per_share=fcf_per_share,
            target_mean_price=target_mean_price,
            target_low_price=target_low_price,
            target_high_price=target_high_price,
            price_to_sales=price_to_sales,
            price_to_book=price_to_book,
            ev_to_ebitda=ev_to_ebitda,
            growth_estimates_raw=growth_est_raw,
        )
        
    except Exception as e:
        logger.error(f"[{ticker}] 获取数据失败: {e}", exc_info=True)
        return StockData(
            ticker=ticker,
            display_symbol=display_symbol,
            name=ticker,
            error=str(e)
        )


def fetch_stock_data(
    tickers: List[str],
    max_workers: int = 5,
    progress_callback: Optional[callable] = None
) -> FetchResult:
    """
    批量获取股票数据（支持并行）
    
    Args:
        tickers: 股票代码列表
        max_workers: 并行线程数
        progress_callback: 进度回调函数 (current, total, ticker)
        
    Returns:
        FetchResult 对象
    """
    current_year = datetime.datetime.now().year
    next_year_int = current_year + 1
    year_after_next_int = current_year + 2
    
    logger.info(f"=== 开始批量数据抓取 ({len(tickers)} 只股票, {max_workers} 线程) ===")
    
    result = FetchResult()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_ticker = {
            executor.submit(
                fetch_single_stock, 
                ticker, 
                next_year_int, 
                year_after_next_int
            ): ticker 
            for ticker in tickers
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            
            if progress_callback:
                progress_callback(completed, len(tickers), ticker)
            
            try:
                stock_data = future.result()
                if stock_data.error:
                    result.failed_tickers.append(ticker)
                else:
                    result.data.append(stock_data)
            except Exception as e:
                logger.error(f"[{ticker}] 任务执行异常: {e}")
                result.failed_tickers.append(ticker)
    
    logger.info(f"=== 抓取完成: 成功 {len(result.data)}, 失败 {len(result.failed_tickers)} ===")
    return result

