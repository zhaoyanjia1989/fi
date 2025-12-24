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
        
        # 检测低基数高增长陷阱
        remark = ""
        pe_ttm = stock.trailing_pe
        fwd_pe = stock.forward_pe
        growth = stock.fy_next_growth if stock.fy_next_growth else growth_result.value
        
        if growth and growth > 100:
            remark = "⚠低基数"
        elif pe_ttm and fwd_pe and growth:
            if fwd_pe > pe_ttm and growth > 50:
                remark = "⚠低基数"
        elif not pe_ttm and growth and growth > 50:
            remark = "⚠缺TTM"
        
        final_garp = fy_garp if fy_garp else garp_value
        
        # 计算股价预估范围
        price = stock.current_price
        # 0y 股价范围：基于当前价格和 0y EPS 分散度
        price_0y_range = None
        if price and stock.eps_0y_low_ratio and stock.eps_0y_high_ratio:
            p_0y_low = price * stock.eps_0y_low_ratio
            p_0y_high = price * stock.eps_0y_high_ratio
            price_0y_range = f"{p_0y_low:.1f}~{p_0y_high:.1f}"
        
        # +1y 股价范围：基于预期增长后的价格和 +1y EPS 分散度
        price_1y_range = None
        if price and stock.fy_next_growth and stock.eps_1y_low_ratio and stock.eps_1y_high_ratio:
            # 预期均价 = 当前价格 × (1 + 增速)
            expected_price = price * (1 + stock.fy_next_growth / 100)
            p_1y_low = expected_price * stock.eps_1y_low_ratio
            p_1y_high = expected_price * stock.eps_1y_high_ratio
            price_1y_range = f"{p_1y_low:.1f}~{p_1y_high:.1f}"
        
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
            "备注": remark,
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
            # 股价预估范围
            "0y股价": price_0y_range,
            "+1y股价": price_1y_range,
        })
    
    # 排序
    records.sort(key=lambda x: x["GARP"] if x["GARP"] else 999)
    
    # 输出表格
    console = Console()
    # 使用通用标签: 0y=当前财年, +1y=下一财年
    table = Table(title="GARP 分析结果 (0y=当前财年, +1y=下一财年)", show_header=True)
    
    table.add_column("代码", no_wrap=True)
    table.add_column("名称", no_wrap=True, max_width=8)
    table.add_column("FY", justify="right")
    table.add_column("现价", justify="right")
    table.add_column("0y股价预估", justify="right")
    table.add_column("+1y股价预估", justify="right")
    table.add_column("+1yPE", justify="right")
    table.add_column("+1y%", justify="right")
    table.add_column("GARP", justify="right")
    table.add_column("备注", no_wrap=True)
    
    for r in records:
        table.add_row(
            r["代码"].split(" ")[0],
            r["名称"][:8] if r["名称"] else "-",
            f"{r['财年结束月']}月" if r["财年结束月"] else "-",
            str(r["价格"]) if r["价格"] else "-",
            r["0y股价"] if r["0y股价"] else "-",
            r["+1y股价"] if r["+1y股价"] else "-",
            str(r["FY_PE"]) if r["FY_PE"] else "-",
            str(r["FY1增长%"]) if r["FY1增长%"] else "-",
            str(r["GARP"]) if r["GARP"] else "-",
            r["备注"],
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
            fed_table.add_row("利率目标区间", rate_range, "-", "联邦基金目标利率")
        
        # SOFR
        if sofr and sofr.latest_value:
            fed_table.add_row("SOFR", f"{sofr.latest_value:.2f}%", format_trend(sofr), "担保隔夜融资利率")
        
        # IORB
        if iorb and iorb.latest_value:
            fed_table.add_row("IORB", f"{iorb.latest_value:.2f}%", format_trend(iorb), "准备金利率")
        
        # EFFR
        if effr and effr.latest_value:
            fed_table.add_row("EFFR", f"{effr.latest_value:.2f}%", format_trend(effr), "有效联邦基金利率")
        
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
            fed_table.add_row("SOFR-IORB", f"{spread:+.0f}bp", spread_trend_str + "bp", spread_note)
        
        # TGA 余额 (单位: 百万美元 -> 亿美元)
        if tga and tga.latest_value:
            tga_b = tga.latest_value / 100  # 百万 -> 亿
            fed_table.add_row("TGA余额", f"{tga_b:.0f}亿$", format_trend(tga, "tga") + "亿", "财政部一般账户")
        
        # 银行准备金 (单位: 百万美元 -> 万亿美元)
        if reserves and reserves.latest_value:
            reserves_t = reserves.latest_value / 1000000  # 百万 -> 万亿
            fed_table.add_row("银行准备金", f"{reserves_t:.2f}万亿$", format_trend(reserves, "reserves") + "万亿", "充裕>2.5万亿")
        
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

