"""
GARP (Growth at Reasonable Price) 分析器
市场数据分析应用 - 基于 Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import sys
import datetime
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GARP_Analyzer")

# === 导入核心模块 ===
from config import (
    HK_TO_US_MAP, 
    DEFAULT_TICKERS, 
    GARPThresholds, 
    STATUS_COLORS,
    CACHE_TTL_SECONDS,
)
from core import (
    fetch_stock_data, 
    StockData,
    calculate_garp, 
    determine_growth_rate,
)

# === 页面配置 ===
st.set_page_config(
    page_title="GARP 卖方数据分析",
    layout="wide"
)


def setup_sidebar() -> tuple[list[str], GARPThresholds]:
    """
    设置侧边栏并返回配置
    
    Returns:
        (股票代码列表, GARP阈值配置)
    """
    st.sidebar.markdown("**参数配置**")
    
    # 股票代码输入
    st.sidebar.markdown("**证券代码**")
    custom_tickers = st.sidebar.text_area(
        "每行一个代码",
        value="\n".join(DEFAULT_TICKERS),
        height=150,
        label_visibility="collapsed"
    )
    
    # 处理代码列表（港股转美股）
    raw_list = [t.strip() for t in custom_tickers.split('\n') if t.strip()]
    ticker_list = [HK_TO_US_MAP.get(t, t) for t in raw_list]
    
    st.sidebar.markdown("---")
    
    # GARP 阈值配置
    st.sidebar.markdown("**估值阈值**")
    undervalued = st.sidebar.slider(
        "低估阈值 (GARP <)", 
        min_value=0.0, 
        max_value=1.5, 
        value=0.75, 
        step=0.05,
        help="GARP 值低于此值判定为低估"
    )
    fair_upper = st.sidebar.slider(
        "合理上限 (GARP ≤)", 
        min_value=0.5, 
        max_value=2.5, 
        value=1.25, 
        step=0.05,
        help="GARP 值高于此值判定为偏高"
    )
    
    thresholds = GARPThresholds(undervalued=undervalued, fair_upper=fair_upper)
    
    return ticker_list, thresholds


def process_stock_data(
    stocks: list[StockData], 
    thresholds: GARPThresholds
) -> pd.DataFrame:
    """
    处理股票数据并生成 DataFrame
    
    Args:
        stocks: StockData 列表
        thresholds: GARP 阈值配置
        
    Returns:
        处理后的 DataFrame
    """
    import yfinance as yf
    
    records = []
    
    for stock in stocks:
        # 跳过无远期PE的股票
        if stock.forward_pe is None:
            continue
        
        # 计算增长率
        growth_result = determine_growth_rate(
            next_2y_growth=stock.next_2y_growth,
            next_year_growth=stock.next_year_growth,
            forward_pe=stock.forward_pe,
            trailing_pe=stock.trailing_pe,
            peg_ratio=stock.peg_ratio,
            revenue_growth=stock.revenue_growth,
            earnings_growth=stock.earnings_growth,
        )
        
        # 计算 GARP
        garp_value = calculate_garp(
            forward_pe=stock.forward_pe,
            peg_ratio=stock.peg_ratio,
            calc_growth=growth_result.value
        )
        
        # 评估状态
        status = thresholds.evaluate(growth_result.value, garp_value)
        
        # 使用转换后的EPS数据（按交易市场货币）
        eps_0y = round(stock.eps_0y_converted, 2) if stock.eps_0y_converted is not None else (round(stock.eps_0y, 2) if stock.eps_0y else None)
        eps_1y = round(stock.eps_1y_converted, 2) if stock.eps_1y_converted is not None else (round(stock.eps_1y, 2) if stock.eps_1y else None)
        
        # EPS货币显示：转换路径 + 汇率信息
        if stock.eps_converted_currency and stock.eps_exchange_rate and stock.eps_currency:
            if stock.eps_exchange_rate != 1.0 and stock.eps_currency != stock.eps_converted_currency:
                # 有汇率转换，显示转换路径
                eps_currency = f"{stock.eps_currency}->{stock.eps_converted_currency} (汇率:{stock.eps_exchange_rate:.4f})"
            else:
                # 无转换
                eps_currency = stock.eps_converted_currency
        else:
            eps_currency = stock.eps_currency or 'USD'
        
        records.append({
            "代码": stock.display_symbol,
            "名称": stock.name,
            "当前价格": round(stock.current_price, 2) if stock.current_price else None,
            "PE TTM": round(stock.trailing_pe, 2) if stock.trailing_pe else None,
            "远期PE": round(stock.forward_pe, 2),
            "核心增长率(%)": round(growth_result.value, 2) if growth_result.value else None,
            "参考指标": growth_result.source,
            "GARP值": round(garp_value, 2) if garp_value else None,
            "预测(后两年)": round(stock.next_2y_growth, 2) if stock.next_2y_growth else None,
            "预测(明年)": round(stock.next_year_growth, 2) if stock.next_year_growth else None,
            "+0y EPS": eps_0y,
            "+1y EPS": eps_1y,
            "EPS货币": eps_currency,
            "评价": status,
        })
    
    return pd.DataFrame(records)


def render_scatter_chart(df: pd.DataFrame):
    """渲染散点图"""
    plot_df = df.dropna(subset=['远期PE', '核心增长率(%)'])
    
    if plot_df.empty:
        st.info("暂无足够数据生成图表")
        return
    
    fig = px.scatter(
        plot_df,
        x="核心增长率(%)",
        y="远期PE",
        color="评价",
        size="远期PE",
        hover_name="名称",
        hover_data=["代码", "PE TTM", "参考指标", "预测(后两年)"],
        text="名称",
        color_discrete_map=STATUS_COLORS,
        height=500,
    )
    
    fig.update_traces(
        textposition='top center',
        textfont_size=10,
    )
    
    fig.update_layout(
        xaxis_title="核心增长率 (%)",
        yaxis_title="远期市盈率 (Forward PE)",
        legend_title="估值状态",
        hovermode='closest',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame):
    """渲染数据表格"""
    display_df = df.sort_values(by="GARP值", na_position='last')
    
    st.dataframe(
        display_df,
        column_config={
            "参考指标": st.column_config.TextColumn(
                "增长来源", 
                help="P1: 后两年预测 (最优)\nP2: PEG推算\nP2.5: 市场隐含\nP3: 明年预测\nP4: 营收增长\nP5: 季度盈利"
            ),
            "PE TTM": st.column_config.NumberColumn(format="%.1f", help="滚动市盈率"),
            "远期PE": st.column_config.NumberColumn(format="%.1f", help="远期市盈率"),
            "核心增长率(%)": st.column_config.NumberColumn(format="%.1f"),
            "GARP值": st.column_config.NumberColumn(format="%.2f", help="< 0.75 低估, 0.75-1.25 合理, > 1.25 偏高"),
            "预测(后两年)": st.column_config.NumberColumn(format="%.1f%%"),
            "预测(明年)": st.column_config.NumberColumn(format="%.1f%%"),
            "当前价格": st.column_config.NumberColumn(format="%.2f"),
            "+0y EPS": st.column_config.NumberColumn(format="%.2f", help="当前财年 EPS (已转换为交易市场货币)"),
            "+1y EPS": st.column_config.NumberColumn(format="%.2f", help="下一财年 EPS (已转换为交易市场货币)"),
            "EPS货币": st.column_config.TextColumn(width="medium", help="EPS货币 (已转换为交易市场货币，显示汇率)"),
            "评价": st.column_config.TextColumn(width="small"),
        },
        hide_index=True,
        use_container_width=True,
    )


def get_eps_analysis_data(stocks: list[StockData]) -> pd.DataFrame:
    """
    获取 EPS 分析数据
    
    Args:
        stocks: StockData 列表
        
    Returns:
        EPS 分析 DataFrame
    """
    import yfinance as yf
    
    records = []
    
    for stock in stocks:
        # 初始化基础数据（即使获取失败也显示）
        eps_data = {
            "代码": stock.display_symbol,
            "名称": stock.name,
            "货币": None,
            "0q EPS": None,
            "+1q EPS": None,
            "+0y EPS": None,
            "+0y 最低": None,
            "+0y 最高": None,
            "+1y EPS": None,
            "+1y 最低": None,
            "+1y 最高": None,
            "分析师数": None,
        }
        
        try:
            ticker_obj = yf.Ticker(stock.ticker)
            info = ticker_obj.info
            
            # 货币信息
            currency = info.get('currency', 'USD')
            financial_currency = info.get('financialCurrency', 'USD')
            eps_data["货币"] = financial_currency
            
            # 获取 earnings_estimate
            ee = ticker_obj.earnings_estimate
            
            if ee is None:
                logger.debug(f"[{stock.ticker}] earnings_estimate 为 None")
                records.append(eps_data)
                continue
            
            if ee.empty:
                logger.debug(f"[{stock.ticker}] earnings_estimate 为空")
                records.append(eps_data)
                continue
            
            logger.debug(f"[{stock.ticker}] earnings_estimate 形状: {ee.shape}, 索引: {list(ee.index)}")
            
            # 季度 EPS
            if '0q' in ee.index:
                eps_0q = ee.loc['0q', 'avg']
                eps_data["0q EPS"] = round(eps_0q, 2) if pd.notna(eps_0q) else None
            else:
                logger.debug(f"[{stock.ticker}] 无 0q 数据")
            
            if '+1q' in ee.index:
                eps_1q = ee.loc['+1q', 'avg']
                eps_data["+1q EPS"] = round(eps_1q, 2) if pd.notna(eps_1q) else None
            else:
                logger.debug(f"[{stock.ticker}] 无 +1q 数据")
            
            # 财年 EPS
            if '0y' in ee.index:
                eps_0y = ee.loc['0y', 'avg']
                eps_data["+0y EPS"] = round(eps_0y, 2) if pd.notna(eps_0y) else None
                
                eps_0y_low = ee.loc['0y', 'low']
                eps_data["+0y 最低"] = round(eps_0y_low, 2) if pd.notna(eps_0y_low) else None
                
                eps_0y_high = ee.loc['0y', 'high']
                eps_data["+0y 最高"] = round(eps_0y_high, 2) if pd.notna(eps_0y_high) else None
            else:
                logger.debug(f"[{stock.ticker}] 无 0y 数据")
            
            if '+1y' in ee.index:
                eps_1y = ee.loc['+1y', 'avg']
                eps_data["+1y EPS"] = round(eps_1y, 2) if pd.notna(eps_1y) else None
                
                eps_1y_low = ee.loc['+1y', 'low']
                eps_data["+1y 最低"] = round(eps_1y_low, 2) if pd.notna(eps_1y_low) else None
                
                eps_1y_high = ee.loc['+1y', 'high']
                eps_data["+1y 最高"] = round(eps_1y_high, 2) if pd.notna(eps_1y_high) else None
            else:
                logger.debug(f"[{stock.ticker}] 无 +1y 数据")
            
            # 分析师数量
            if 'numberOfAnalysts' in ee.columns:
                if '+1y' in ee.index and pd.notna(ee.loc['+1y', 'numberOfAnalysts']):
                    eps_data["分析师数"] = int(ee.loc['+1y', 'numberOfAnalysts'])
                elif '0y' in ee.index and pd.notna(ee.loc['0y', 'numberOfAnalysts']):
                    eps_data["分析师数"] = int(ee.loc['0y', 'numberOfAnalysts'])
            
            records.append(eps_data)
            
        except Exception as e:
            logger.warning(f"[{stock.ticker}] 获取 EPS 数据失败: {e}", exc_info=True)
            # 即使失败也添加记录，显示基本信息
            records.append(eps_data)
            continue
    
    return pd.DataFrame(records)


def render_eps_analysis_table(stocks: list[StockData]):
    """
    渲染 EPS 分析表格
    
    Args:
        stocks: StockData 列表
    """
    df = get_eps_analysis_data(stocks)
    
    if df.empty:
        st.warning("⚠️ 暂无 EPS 分析数据。可能的原因：\n- Yahoo Finance 未提供该股票的 earnings_estimate 数据\n- 网络连接问题\n- 股票代码不正确")
        return
    
    # 显示数据统计
    total_stocks = len(df)
    stocks_with_eps = len(df[df['+0y EPS'].notna() | df['+1y EPS'].notna()])
    st.caption(f"共 {total_stocks} 只股票，其中 {stocks_with_eps} 只有 EPS 数据")
    
    # 重新排列列的顺序
    column_order = [
        "代码", "名称", "货币",
        "0q EPS", "+1q EPS",
        "+0y EPS", "+0y 最低", "+0y 最高",
        "+1y EPS", "+1y 最低", "+1y 最高",
        "分析师数"
    ]
    
    # 只保留存在的列
    available_columns = [col for col in column_order if col in df.columns]
    df_display = df[available_columns]
    
    st.dataframe(
        df_display,
        column_config={
            "代码": st.column_config.TextColumn(width="small"),
            "名称": st.column_config.TextColumn(width="medium"),
            "货币": st.column_config.TextColumn(width="small", help="财报货币"),
            "0q EPS": st.column_config.NumberColumn(format="%.2f", help="当前季度 EPS"),
            "+1q EPS": st.column_config.NumberColumn(format="%.2f", help="下一季度 EPS"),
            "+0y EPS": st.column_config.NumberColumn(format="%.2f", help="当前财年 EPS 平均"),
            "+0y 最低": st.column_config.NumberColumn(format="%.2f", help="当前财年 EPS 最低"),
            "+0y 最高": st.column_config.NumberColumn(format="%.2f", help="当前财年 EPS 最高"),
            "+1y EPS": st.column_config.NumberColumn(format="%.2f", help="下一财年 EPS 平均"),
            "+1y 最低": st.column_config.NumberColumn(format="%.2f", help="下一财年 EPS 最低"),
            "+1y 最高": st.column_config.NumberColumn(format="%.2f", help="下一财年 EPS 最高"),
            "分析师数": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
        use_container_width=True,
    )


def render_realtime_quotes(ticker_list: list[str]):
    """
    渲染实时行情表格 (长桥 API)
    
    注意: 长桥 API 功能已暂时移除
    """
    st.info("实时行情功能暂时不可用")


def main():
    """主函数"""
    # 标题
    st.markdown("### 市场数据分析概览")
    
    # 设置侧边栏
    ticker_list, thresholds = setup_sidebar()
    
    # 主区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        refresh_btn = st.button("刷新数据")
    
    with col2:
        st.caption(f"共 {len(ticker_list)} 只股票")
    
    if refresh_btn:
        # 进度条
        progress_bar = st.progress(0, text="准备获取数据...")
        status_text = st.empty()
        
        def update_progress(current: int, total: int, ticker: str):
            progress = current / total
            progress_bar.progress(progress, text=f"正在处理: {ticker}")
            status_text.text(f"进度: {current}/{total}")
        
        # 获取数据
        with st.spinner('正在连接 Yahoo Finance...'):
            result = fetch_stock_data(
                tickers=ticker_list,
                max_workers=5,
                progress_callback=update_progress
            )
        
        # 清理进度显示
        progress_bar.empty()
        status_text.empty()
        
        # 显示获取状态
        if result.failed_tickers:
            st.warning(f"以下股票获取失败: {', '.join(result.failed_tickers)}")
        
        # 处理数据
        if result.data:
            # 先显示 EPS 分析表格
            st.markdown("**📊 Yahoo Finance EPS 分析**")
            render_eps_analysis_table(result.data)
            
            st.markdown("---")
            
            df = process_stock_data(result.data, thresholds)
            
            if not df.empty:
                # 输出到日志
                logger.info("=== 数据表格 ===")
                logger.info("\n" + df.to_string(index=False))
                
                # 显示时间戳
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"数据更新: {result.fetch_time.strftime('%Y-%m-%d %H:%M:%S')} | 当前时间: {current_time}")
                
                # 图表
                st.markdown("**指标散点分布**")
                render_scatter_chart(df)
                
                st.markdown("---")
                
                # 数据表
                st.markdown("**详细数据表**")
                render_data_table(df)
                
                # 导出功能
                st.markdown("---")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="导出 CSV",
                    data=csv,
                    file_name=f"garp_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                # 实时行情表格 (长桥 API)
                st.markdown("---")
                st.markdown("**📈 实时行情 (长桥API)**")
                render_realtime_quotes(ticker_list)
                
            else:
                st.warning("无有效数据（所有股票缺少远期PE）")
        else:
            st.error("未能获取任何股票数据")
    else:
        st.text("请点击刷新按钮获取数据")
    
    # 页脚
    st.markdown("---")
    st.caption("注：数据来源 Yahoo Finance。分析师预测数据缺失时，会自动降级使用市场隐含增长率或营收增长率。")


if __name__ == "__main__":
    main()
