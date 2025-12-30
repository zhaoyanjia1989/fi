"""
GARP 分析器 - 命令行版本
直接输出数据表格到终端
"""
import logging
import sys
import datetime

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GARP")

# 抑制子模块日志
logging.getLogger("GARP_Analyzer.fetcher").setLevel(logging.WARNING)

from config import DEFAULT_TICKERS, HK_TO_US_MAP, GARPThresholds
from core import fetch_stock_data, calculate_garp, determine_growth_rate
from core.fred_fetcher import fetch_fed_indicators
from core.metals_fetcher import get_gold_silver_delivery
from rich.console import Console
from rich.table import Table



def main():
    # 处理股票代码
    ticker_list = [HK_TO_US_MAP.get(t, t) for t in DEFAULT_TICKERS]
    thresholds = GARPThresholds()
    
    logger.info(f"开始获取 {len(ticker_list)} 只股票数据...")
    
    # 获取数据
    result = fetch_stock_data(tickers=ticker_list, max_workers=5)
    
    if result.failed_tickers:
        logger.warning(f"获取失败: {', '.join(result.failed_tickers)}")
    
    if not result.data:
        logger.error("未获取到任何数据")
        return
    
    # 处理数据
    records = []
    for stock in result.data:
        if stock.forward_pe is None:
            continue
        
        growth_result = determine_growth_rate(
            next_2y_growth=stock.next_2y_growth,
            next_year_growth=stock.next_year_growth,
            forward_pe=stock.forward_pe,
            trailing_pe=stock.trailing_pe,
            peg_ratio=stock.peg_ratio,
            revenue_growth=stock.revenue_growth,
            earnings_growth=stock.earnings_growth,
        )
        
        garp_value = calculate_garp(
            forward_pe=stock.forward_pe,
            peg_ratio=stock.peg_ratio,
            calc_growth=growth_result.value
        )
        
        # 财年对齐的 GARP (使用 earnings_estimate 数据)
        fy_garp = None
        if stock.fy_next_pe and stock.fy_next_growth and stock.fy_next_growth > 0:
            fy_garp = stock.fy_next_pe / stock.fy_next_growth
        
        # 使用财年对齐的数据评估
        eval_garp = fy_garp if fy_garp else garp_value
        eval_growth = stock.fy_next_growth if stock.fy_next_growth else growth_result.value
        status = thresholds.evaluate(eval_growth, eval_garp)
        
        final_garp = fy_garp if fy_garp else garp_value
        
        pe_ttm = stock.trailing_pe
        records.append({
            "代码": stock.display_symbol,
            "名称": stock.name,
            "价格": round(stock.current_price, 2) if stock.current_price else None,
            "PE_TTM": round(pe_ttm, 2) if pe_ttm else None,
            "FY_PE": round(stock.fy_next_pe, 2) if stock.fy_next_pe else None,
            "FY0增长%": round(stock.fy_current_growth, 2) if stock.fy_current_growth else None,
            "FY1增长%": round(stock.fy_next_growth, 2) if stock.fy_next_growth else None,
            "分析师": stock.analyst_count,
            "GARP": round(final_garp, 2) if final_garp else None,
            "评价": status,
            # PE SD Band
            "SD90": round(stock.pe_sd_90d, 2) if stock.pe_sd_90d is not None else None,
            "SD180": round(stock.pe_sd_180d, 2) if stock.pe_sd_180d is not None else None,
            # 财年信息
            "FY0名称": stock.fy_current_name,
            "FY1名称": stock.fy_next_name,
            "FY0范围": stock.fy_current_range,
            "FY1范围": stock.fy_next_range,
            # 财年结束月份 (从范围中提取，如 "2025-01-01 ~ 2025-12-31" -> 12)
            "财年结束月": int(stock.fy_next_range.split(" ~ ")[1].split("-")[1]) if stock.fy_next_range else None,
            # EPS 数据（使用转换后的EPS）
            "+0y EPS": round(stock.eps_0y_converted, 2) if stock.eps_0y_converted is not None else (round(stock.eps_0y, 2) if stock.eps_0y else None),
            "+1y EPS": round(stock.eps_1y_converted, 2) if stock.eps_1y_converted is not None else (round(stock.eps_1y, 2) if stock.eps_1y else None),
            # EPS货币显示：转换路径 + 汇率信息
            "EPS货币": (
                f"{stock.eps_currency}->{stock.eps_converted_currency} (汇率:{stock.eps_exchange_rate:.4f})" 
                if stock.eps_converted_currency and stock.eps_exchange_rate and stock.eps_currency 
                   and stock.eps_exchange_rate != 1.0 and stock.eps_currency != stock.eps_converted_currency
                else (stock.eps_converted_currency or stock.eps_currency or 'USD')
            ),
        })
    
    # 排序
    records.sort(key=lambda x: x["GARP"] if x["GARP"] else 999)
    
    # 输出表格
    console = Console()
    # 使用通用标签: 0y=当前财年, +1y=下一财年
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    table = Table(title=f"GARP 卖方分析结果 (0y=当前财年, +1y=下一财年) - {current_time}", show_header=True)
    
    table.add_column("代码", no_wrap=True)
    table.add_column("名称", no_wrap=True, max_width=8)
    table.add_column("FY", justify="right")
    table.add_column("现价", justify="right")
    table.add_column("+1yPE", justify="right")
    table.add_column("+1y%", justify="right")
    table.add_column("+0y EPS", justify="right")
    table.add_column("+1y EPS", justify="right")
    table.add_column("EPS货币", justify="right")
    table.add_column("GARP", justify="right")
    
    for r in records:
        table.add_row(
            r["代码"].split(" ")[0],
            r["名称"][:8] if r["名称"] else "-",
            f"{r['财年结束月']}月" if r["财年结束月"] else "-",
            str(r["价格"]) if r["价格"] else "-",
            str(r["FY_PE"]) if r["FY_PE"] else "-",
            str(r["FY1增长%"]) if r["FY1增长%"] else "-",
            str(r["+0y EPS"]) if r["+0y EPS"] else "-",
            str(r["+1y EPS"]) if r["+1y EPS"] else "-",
            r["EPS货币"] if r["EPS货币"] else "-",
            str(r["GARP"]) if r["GARP"] else "-",
        )
    
    console.print()
    console.print(table)
    console.print()
    
    # === 美联储指标数据 ===
    logger.info("获取美联储指标数据...")
    fed_data = fetch_fed_indicators(days=30)
    
    def format_trend(series, fmt="rate", limit=5):
        """格式化走势数据, fmt: rate(利率%), amount(金额), 顺序: 旧→新"""
        if not series or not series.observations:
            return "-"
        vals = []
        for obs in series.observations[:limit]:
            if obs.value is not None:
                if fmt == "rate":
                    vals.append(f"{obs.value:.2f}")
                elif fmt == "tga":
                    vals.append(f"{obs.value/100:.0f}")
                elif fmt == "reserves":
                    vals.append(f"{obs.value/1000000:.2f}")
                else:
                    vals.append(f"{obs.value:.2f}")
        vals.reverse()  # 反转为旧→新
        return " → ".join(vals) if vals else "-"
    
    if fed_data:
        fed_table = Table(title="美联储流动性指标 (FRED)", show_header=True)
        fed_table.add_column("指标", no_wrap=True)
        fed_table.add_column("最新值", justify="right")
        fed_table.add_column("近一周走势 (旧→新)", justify="left")
        fed_table.add_column("说明", justify="left")
        
        # 获取各指标值
        sofr = fed_data.get("SOFR")
        iorb = fed_data.get("IORB")
        effr = fed_data.get("EFFR")
        rate_upper = fed_data.get("DFEDTARU")
        rate_lower = fed_data.get("DFEDTARL")
        tga = fed_data.get("WTREGEN")
        reserves = fed_data.get("WRESBAL")
        
        # 利率范围
        if rate_upper and rate_lower and rate_upper.latest_value and rate_lower.latest_value:
            rate_range = f"{rate_lower.latest_value:.2f}% ~ {rate_upper.latest_value:.2f}%"
            rate_date = rate_lower.latest_date or rate_upper.latest_date
            rate_display = f"{rate_range}\n({rate_date})" if rate_date else rate_range
            fed_table.add_row("利率目标区间", rate_display, "-", "联邦基金目标利率")
        
        # SOFR
        if sofr and sofr.latest_value:
            sofr_date = sofr.latest_date or ""
            sofr_display = f"{sofr.latest_value:.2f}%\n({sofr_date})" if sofr_date else f"{sofr.latest_value:.2f}%"
            fed_table.add_row("SOFR", sofr_display, format_trend(sofr), "担保隔夜融资利率")
        
        # IORB
        if iorb and iorb.latest_value:
            iorb_date = iorb.latest_date or ""
            iorb_display = f"{iorb.latest_value:.2f}%\n({iorb_date})" if iorb_date else f"{iorb.latest_value:.2f}%"
            fed_table.add_row("IORB", iorb_display, format_trend(iorb), "准备金利率")
        
        
        # EFFR
        if effr and effr.latest_value:
            effr_date = effr.latest_date or ""
            effr_display = f"{effr.latest_value:.2f}%\n({effr_date})" if effr_date else f"{effr.latest_value:.2f}%"
            fed_table.add_row("EFFR", effr_display, format_trend(effr), "有效联邦基金利率")
        
        # SOFR-IORB 差值
        if sofr and iorb and sofr.latest_value and iorb.latest_value:
            spread = (sofr.latest_value - iorb.latest_value) * 100  # 转为bp
            spread_note = "正常" if spread < 5 else "流动性趋紧" if spread > 10 else "关注"
            # 计算历史差值
            spread_trend = []
            for i in range(min(5, len(sofr.observations), len(iorb.observations))):
                s_val = sofr.observations[i].value
                i_val = iorb.observations[i].value if i < len(iorb.observations) else None
                if s_val and i_val:
                    spread_trend.append(f"{(s_val - i_val)*100:+.0f}")
            spread_trend.reverse()  # 反转为旧→新
            spread_trend_str = " → ".join(spread_trend) if spread_trend else "-"
            # 使用SOFR或IORB的日期（优先使用SOFR）
            spread_date = sofr.latest_date or iorb.latest_date or ""
            spread_display = f"{spread:+.0f}bp\n({spread_date})" if spread_date else f"{spread:+.0f}bp"
            fed_table.add_row("SOFR-IORB", spread_display, spread_trend_str + "bp", spread_note)
        
        # TGA 余额 (单位: 百万美元 -> 亿美元)
        if tga and tga.latest_value:
            tga_b = tga.latest_value / 100  # 百万 -> 亿
            tga_date = tga.latest_date or ""
            tga_display = f"{tga_b:.0f}亿$\n({tga_date})" if tga_date else f"{tga_b:.0f}亿$"
            fed_table.add_row("TGA余额", tga_display, format_trend(tga, "tga") + "亿", "财政部一般账户")
        
        # 银行准备金 (单位: 百万美元 -> 万亿美元)
        if reserves and reserves.latest_value:
            reserves_t = reserves.latest_value / 1000000  # 百万 -> 万亿
            reserves_date = reserves.latest_date or ""
            reserves_display = f"{reserves_t:.2f}万亿$\n({reserves_date})" if reserves_date else f"{reserves_t:.2f}万亿$"
            fed_table.add_row("银行准备金", reserves_display, format_trend(reserves, "reserves") + "万亿", "充裕>3万亿")
        
        console.print(fed_table)
        console.print()
    
    # === CME 金属交割通知数据 ===
    logger.info("获取 CME 金属交割数据...")
    gold_delivery, silver_delivery = get_gold_silver_delivery()
    
    if gold_delivery or silver_delivery:
        metals_table = Table(title="COMEX 金属交割通知 (MTD)", show_header=True)
        metals_table.add_column("品种", no_wrap=True)
        metals_table.add_column("最新日期", justify="right")
        metals_table.add_column("今日", justify="right")
        metals_table.add_column("累计", justify="right")
        metals_table.add_column("近7天交割量 (日期:数量)", justify="left")
        
        if gold_delivery:
            recent_7d_data = gold_delivery.recent_7d_with_dates
            recent_7d_str = " → ".join([f"{d}:{x:,}" for d, x in recent_7d_data]) if recent_7d_data else "-"
            metals_table.add_row(
                "黄金 (GC)",
                gold_delivery.latest_date,
                f"{gold_delivery.latest_daily:,}",
                f"{gold_delivery.total_cumulative:,}",
                recent_7d_str,
            )
        
        if silver_delivery:
            recent_7d_data = silver_delivery.recent_7d_with_dates
            recent_7d_str = " → ".join([f"{d}:{x:,}" for d, x in recent_7d_data]) if recent_7d_data else "-"
            metals_table.add_row(
                "白银 (SI)",
                silver_delivery.latest_date,
                f"{silver_delivery.latest_daily:,}",
                f"{silver_delivery.total_cumulative:,}",
                recent_7d_str,
            )
        
        console.print(metals_table)
        console.print()


if __name__ == "__main__":
    main()

