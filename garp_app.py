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
    page_title="GARP 数据分析",
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
        
        records.append({
            "代码": stock.display_symbol,
            "名称": stock.name,
            "当前价格": stock.current_price,
            "PE TTM": round(stock.trailing_pe, 2) if stock.trailing_pe else None,
            "远期PE": round(stock.forward_pe, 2),
            "核心增长率(%)": round(growth_result.value, 2) if growth_result.value else None,
            "参考指标": growth_result.source,
            "GARP值": round(garp_value, 2) if garp_value else None,
            "预测(后两年)": round(stock.next_2y_growth, 2) if stock.next_2y_growth else None,
            "预测(明年)": round(stock.next_year_growth, 2) if stock.next_year_growth else None,
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
            "评价": st.column_config.TextColumn(width="small"),
        },
        hide_index=True,
        use_container_width=True,
    )


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
            df = process_stock_data(result.data, thresholds)
            
            if not df.empty:
                # 输出到日志
                logger.info("=== 数据表格 ===")
                logger.info("\n" + df.to_string(index=False))
                
                # 显示时间戳
                st.caption(f"数据更新: {result.fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
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
