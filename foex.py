# -*- coding: utf-8 -*-
"""
中文版网页全球宏观分析报告生成器
版本: 9.0 (旗舰版：全交叉盘 + 完整智能分析逻辑回归)
更新:
1. [修复] 完整恢复了 v8.6 丢失的 "战术机会"、"应回避资产"、"纵向对比" 等深度分析模块。
2. 保持 7 大主流货币全交叉盘覆盖 (30+ 资产)。
3. 保持 "商品货币内战" 等新增的趋势扫描逻辑。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略 pandas 的某些未来版本警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 配置参数
# =============================================================================

BENCHMARK_TICKER = 'UUP'
TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

# 直盘
G10_CURRENCIES = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X"]
EM_CURRENCIES = ["USDZAR=X", "USDMXN=X", "USDBRL=X"]

# 全量交叉盘列表
CROSS_CURRENCIES = [
    # 日元交叉盘
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "NZDJPY=X", "CADJPY=X", "CHFJPY=X",
    # 欧元交叉盘
    "EURGBP=X", "EURCHF=X", "EURAUD=X", "EURNZD=X", "EURCAD=X",
    # 英镑交叉盘
    "GBPAUD=X", "GBPNZD=X", "GBPCAD=X", "GBPCHF=X",
    # 澳元交叉盘
    "AUDNZD=X", "AUDCAD=X", "AUDCHF=X",
    # 纽元/加元/瑞郎 其他交叉
    "NZDCAD=X", "NZDCHF=X", "CADCHF=X"
]

GLOBAL_MACRO_ASSETS = {
    "标普500指数": "ES=F",
    "MSCI全球指数": "URTH",
    "美国十年期国债收益率": "^TNX",
    "原油": "CL=F",
    "黄金": "GC=F",
    "铜": "HG=F",
    "VIX恐慌指数": "^VIX"
}

ALL_ANALYSIS_ASSETS = list(set(G10_CURRENCIES + EM_CURRENCIES + CROSS_CURRENCIES + list(GLOBAL_MACRO_ASSETS.values())))

COLUMN_TRANSLATIONS = {
    'master_score': '综合大师分',
    'weighted_z_score_rs': '加权相对Z值',
    'acceleration': '动能加速度',
    f'z_score_rs_{TIME_PERIODS["long_term"]}d': f'{TIME_PERIODS["long_term"]}日相对Z值',
    f'z_score_rs_{TIME_PERIODS["mid_term"]}d': f'{TIME_PERIODS["mid_term"]}日相对Z值',
    f'z_score_rs_{TIME_PERIODS["short_term"]}d': f'{TIME_PERIODS["short_term"]}日相对Z值'
}

# =============================================================================
# 2. 数据获取与计算逻辑
# =============================================================================
def fetch_data_robust(tickers, period="2y"):
    print(f"正在下载 {len(tickers)} 个资产在过去 {period} 的价格数据...")
    all_data, valid_tickers = [], []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex): data = data['Close']
            else: data = data['Close'] if 'Close' in data.columns else data

            if not data.empty:
                if isinstance(data, pd.DataFrame): data = data.iloc[:, 0]
                all_data.append(data)
                valid_tickers.append(ticker)
        except Exception as e: print(f"  - 错误: 下载 '{ticker}' 失败: {e}")
    
    if not all_data: return pd.DataFrame()
    combined_df = pd.concat(all_data, axis=1)
    combined_df.columns = valid_tickers
    combined_df.ffill(inplace=True); combined_df.bfill(inplace=True)
    combined_df.dropna(how='all', axis=0, inplace=True); combined_df.dropna(how='all', axis=1, inplace=True)
    print(f"\n数据准备完成。成功合并 {len(combined_df.columns)} 个资产的数据。")
    return combined_df

def calculate_professional_momentum_score(price_data, benchmark_price):
    results = []
    reversed_macro_map = {v: k for k, v in GLOBAL_MACRO_ASSETS.items()}
    for ticker in price_data.columns:
        if ticker == benchmark_price.name: continue
        etf_price = price_data[ticker]
        aligned_benchmark_price = benchmark_price.reindex(etf_price.index).ffill()
        relative_price = (etf_price / aligned_benchmark_price).dropna()
        if len(relative_price) < max(TIME_PERIODS.values()): continue
        metrics = {'Ticker': ticker}
        weighted_z_score_sum = 0
        for term, period_days in TIME_PERIODS.items():
            if len(relative_price) >= period_days:
                rs_returns = (relative_price / relative_price.shift(period_days)) - 1
                mean, std = rs_returns.mean(), rs_returns.std()
                if std > 0:
                    z_score = (rs_returns.iloc[-1] - mean) / std
                    metrics[f'z_score_rs_{period_days}d'] = z_score
                    weighted_z_score_sum += z_score * PERIOD_WEIGHTS[term]
                else: weighted_z_score_sum = np.nan
        if np.isnan(weighted_z_score_sum): continue
        metrics['weighted_z_score_rs'] = weighted_z_score_sum
        lookback_vol = TIME_PERIODS['long_term']
        if len(etf_price) >= lookback_vol:
            annualized_volatility = etf_price.pct_change().dropna().tail(lookback_vol).std() * np.sqrt(252)
            metrics['master_score'] = weighted_z_score_sum / annualized_volatility if annualized_volatility > 0 else 0
        else: continue
        results.append(metrics)
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results).dropna().set_index('Ticker')
    df.rename(index=reversed_macro_map, inplace=True)
    return df

# =============================================================================
# 3. HTML 报告生成模块 (完全体)
# =============================================================================

def generate_market_sentiment_module(all_scores_df):
    html = "<h2>市场情绪指标 (Market Sentiment Indicator)</h2>"
    
    def get_z(asset_name):
        if asset_name in all_scores_df.index:
            return all_scores_df.loc[asset_name, 'weighted_z_score_rs']
        return 0

    vix_z = get_z('VIX恐慌指数')
    spx_z = get_z('标普500指数')
    copper_z = get_z('铜')
    strong_usd_map = {"USDJPY=X": 1, "USDCHF=X": 1, "USDCAD=X": 1, "EURUSD=X": -1, "GBPUSD=X": -1, "AUDUSD=X": -1}
    usd_strength_z = sum(get_z(asset) * direction for asset, direction in strong_usd_map.items() if asset in all_scores_df.index) / len(strong_usd_map)
    
    total_z = spx_z + copper_z - vix_z - usd_strength_z
    sentiment_score = np.clip(total_z * 1.25, -10, 10)
    
    if sentiment_score > 7: status, color = "极度贪婪", "#dc3545"
    elif sentiment_score > 3: status, color = "贪婪", "#ffc107"
    elif sentiment_score > -3: status, color = "中性", "#6c757d"
    elif sentiment_score > -7: status, color = "恐惧", "#28a745"
    else: status, color = "极度恐惧", "#17a2b8"
    
    html += f"""
    <div style='text-align:center; margin: 20px 0;'>
        <div style='font-size: 1.5em;'>当前市场情绪: <strong style='color:{color};'>{status}</strong></div>
        <div style='font-size: 3em; font-weight: bold; margin: 10px 0; color:{color}'>{sentiment_score:.2f}</div>
        <div style='width: 100%; background-color: #e9ecef; border-radius: 5px; height: 20px; position: relative;'>
            <div style='height: 100%; width: 2px; background-color: #343a40; position: absolute; left: 50%;'></div>
            <div style='height: 20px; width: 20px; background-color: {color}; border: 2px solid #fff; border-radius: 50%; position: absolute; top: 0; left: calc({(sentiment_score + 10) * 5}% - 10px);'></div>
        </div>
    </div>
    """
    return html

def generate_deep_dive_analysis_html(all_scores_df, correlation_matrix):
    title = "综合评估 (智能分析 - 旗舰版)"
    html = f"<h2>{title}</h2>"
    
    def get_scores(asset_name, df):
        try:
            if asset_name in df.index: return df.loc[asset_name]
            if asset_name in GLOBAL_MACRO_ASSETS.values():
                rev_map = {v: k for k, v in GLOBAL_MACRO_ASSETS.items()}
                return df.loc[rev_map[asset_name]]
            return None
        except KeyError: return None

    # --- 1. 动量加速度 (含详细解读) ---
    html += "<h3>1. 动量加速度分析：谁在加速？谁在急刹车？</h3>"
    accelerating = all_scores_df[all_scores_df['acceleration'] > 0.5].sort_values('acceleration', ascending=False)
    decelerating = all_scores_df[all_scores_df['acceleration'] < -0.5].sort_values('acceleration', ascending=True)

    html += "<h4>🚀 加速上涨区 (动能爆发)</h4>"
    if not accelerating.empty:
        html += "<ul style='list-style-type: none; padding-left: 0;'>"
        for asset, row in accelerating.head(2).iterrows(): 
            html += f"<li style='margin-bottom: 15px;'><b>{asset}</b>：<b>加速度第一 ({row['acceleration']:.2f})</b>。从60日({row[f'z_score_rs_{TIME_PERIODS['long_term']}d']:.2f})暴力拉升至5日({row[f'z_score_rs_{TIME_PERIODS['short_term']}d']:.2f})。</li>"
        html += "</ul>"
    else: html += "<p>无显著加速资产。</p>"

    html += "<h4>🛑 急剧减速区 (动能衰竭)</h4>"
    if not decelerating.empty:
        html += "<ul style='list-style-type: none; padding-left: 0;'>"
        for asset, row in decelerating.head(2).iterrows():
            html += f"<li style='margin-bottom: 15px;'><b>{asset}</b>：<b>减速第一 ({row['acceleration']:.2f})</b>。5日Z值已跌至({row[f'z_score_rs_{TIME_PERIODS['short_term']}d']:.2f})，拥挤交易正在瓦解。</li>"
        html += "</ul>"
    else: html += "<p>无显著减速资产。</p>"

    # --- 2. 趋势反转扫描 (Pivot List) - 含所有分组 ---
    html += "<h3 style='margin-top: 20px;'>2. 趋势反转扫描 (The \"Pivot\" List)</h3>"
    pivot_groups = [
        {"name": "非美货币直盘", "assets": ["AUDUSD=X", "NZDUSD=X", "GBPUSD=X", "EURUSD=X"], "interpretation": "美元霸权松动。"}, 
        {"name": "标普500 / MSCI全球", "assets": ["标普500指数", "MSCI全球指数"], "interpretation": "股市试图反攻。"}, 
        {"name": "日元套息交叉盘", "assets": ["EURJPY=X", "GBPJPY=X", "AUDJPY=X"], "interpretation": "Risk On/Off 风向标转向。"},
        {"name": "欧系货币组", "assets": ["EURGBP=X", "EURCHF=X"], "interpretation": "欧洲内部资金流向逆转。"},
        {"name": "商品货币内战组", "assets": ["AUDNZD=X", "AUDCAD=X"], "interpretation": "商品货币强弱易手。"}
    ]
    pivot_results = []
    for group in pivot_groups:
        group_assets_df = all_scores_df[all_scores_df.index.isin(group['assets'])]
        if group_assets_df.empty: continue
        long_term_col, short_term_col = f'z_score_rs_{TIME_PERIODS["long_term"]}d', f'z_score_rs_{TIME_PERIODS["short_term"]}d'
        if (group_assets_df[long_term_col] < -0.1).all() and (group_assets_df[short_term_col] > 0.1).all():
            pivot_results.append({"asset": group['name'], "old_world": "📉 弱势", "new_world": "📈 转强", "signal": group['interpretation']})
        if (group_assets_df[long_term_col] > 0.1).all() and (group_assets_df[short_term_col] < -0.1).all():
             pivot_results.append({"asset": group['name'], "old_world": "📈 强势", "new_world": "📉 转弱", "signal": group['interpretation']})
    
    if pivot_results:
        html += "<table class='pivot-table'><thead><tr><th>资产组</th><th>60日趋势</th><th>5日趋势</th><th>信号解读</th></tr></thead><tbody>"
        for item in pivot_results: html += f"<tr><td>{item['asset']}</td><td>{item['old_world']}</td><td>{item['new_world']}</td><td>{item['signal']}</td></tr>"
        html += "</tbody></table>"
    else: html += "<p>当前未发现明确的、成组的趋势反转信号。</p>"

    # --- 3. 宏观因子合成 ---
    html += "<h3 style='margin-top: 20px;'>3. 宏观因子合成</h3>"
    risk_assets = ["标普500指数", "铜", "MSCI全球指数", "AUDUSD=X", "AUDJPY=X", "CADJPY=X"]
    risk_score = 0; count = 0
    for asset in risk_assets:
        scores = get_scores(asset, all_scores_df)
        if scores is not None: risk_score += scores['weighted_z_score_rs']; count += 1
    risk_score = risk_score / count if count > 0 else 0
    risk_status = "Risk On" if risk_score > 0.5 else ("Risk Off" if risk_score < 0 else "温和复苏")
    html += f"<h4>🐂 风险偏好合成指数：{risk_score:.2f} ({risk_status})</h4>"
    
    spx_scores = get_scores("标普500指数", all_scores_df)
    vix_scores = get_scores("VIX恐慌指数", all_scores_df)
    if spx_scores is not None and vix_scores is not None:
        if spx_scores['weighted_z_score_rs'] < 0 and vix_scores['weighted_z_score_rs'] < -0.1:
            html += "<h4>⚠️ 市场异常警示：SPX vs VIX 背离</h4><p>股市跌但VIX未涨，市场处于Complacency状态，警惕补跌。</p>"

    # --- 4. 交易策略启示 (完整逻辑回归) ---
    html += "<h3 style='margin-top: 20px;'>4. 交易策略启示 (Actionable Insights)</h3>"
    z_cols = [f'z_score_rs_{p}d' for p in TIME_PERIODS.values()]
    
    # Core Longs
    core_longs = all_scores_df[(all_scores_df['master_score'] > 5) & (all_scores_df[z_cols] > 0).all(axis=1)].sort_values('master_score', ascending=False)
    html += "<h4>- 核心多头建议 (Core Longs)</h4>"
    if not core_longs.empty:
        html += "<ul>"
        for asset, row in core_longs.head(3).iterrows(): html += f"<li><b>做多 {asset}</b> ({row['master_score']:.2f}): 全周期Z值均为正，趋势一致性极高。</li>"
        html += "</ul>"
    else: html += "<p>暂无符合标准的核心多头。</p>"

    # Core Shorts
    core_shorts = all_scores_df[(all_scores_df['master_score'] < -5) & (all_scores_df[z_cols] < 0).all(axis=1)].sort_values('master_score', ascending=True)
    html += "<h4>- 核心空头建议 (Core Shorts)</h4>"
    if not core_shorts.empty:
        html += "<ul>"
        for asset, row in core_shorts.head(3).iterrows(): html += f"<li><b>做空 {asset}</b> ({row['master_score']:.2f}): 全周期Z值均为负，典型的弱势品种。</li>"
        html += "</ul>"
    else: html += "<p>暂无符合标准的核心空头。</p>"

    # Tactical Plays (恢复!)
    html += "<h4>- 战术机会 (Tactical Plays)</h4>"
    html += "<ul>"
    tactical_insights = 0
    if not decelerating.empty:
        asset, row = decelerating.iloc[0], decelerating.iloc[0]
        html += f"<li><b>(逆势) 押注 {decelerating.index[0]} 趋势衰竭</b>: 它是当前<b>动能减速最快</b>({row['acceleration']:.2f})的品种。适合博取回调。</li>"
        tactical_insights += 1
    bullish_reversal_assets = [p['asset'] for p in pivot_results if "转强" in p['new_world']]
    if bullish_reversal_assets:
        html += f"<li><b>(顺势) 跟随 {bullish_reversal_assets[0]} 的看涨反转</b>: 该组别已出现明确的反转信号，适合左侧布局。</li>"
        tactical_insights += 1
    if tactical_insights == 0: html += "<li>暂无显著的战术性(逆势或反转)机会。</li>"
    html += "</ul>"

    # Avoid List (恢复!)
    html += "<h4>- 应回避的资产 (Avoid List)</h4>"
    html += "<ul>"
    avoid_insights = 0
    # Bull Traps: Long term strong, short term crashing
    bearish_reversal_assets = all_scores_df[(all_scores_df[f'z_score_rs_{TIME_PERIODS["long_term"]}d'] > 0.5) & (all_scores_df[f'z_score_rs_{TIME_PERIODS["short_term"]}d'] < -1.0)]
    if not bearish_reversal_assets.empty:
        for asset, row in bearish_reversal_assets.iterrows():
            html += f"<li><b>{asset}</b>: <b>多头陷阱</b>。长期趋势向上但短期抛压巨大，极易双向亏损。</li>"
            avoid_insights += 1
    # Choppy: No trend, no momentum
    choppy_assets = all_scores_df[(all_scores_df['master_score'].abs() < 1) & (all_scores_df['acceleration'].abs() < 0.3)]
    if not choppy_assets.empty:
        # 只显示前3个垃圾时间的资产
        for asset, row in choppy_assets.head(3).iterrows():
            html += f"<li><b>{asset}</b>: <b>垃圾时间</b>。无明确方向且无动能，建议回避。</li>"
            avoid_insights += 1
    if avoid_insights == 0: html += "<li>当前所有资产均有较明确的趋势信号。</li>"
    html += "</ul>"

    # --- 5. 综合大师分横向与纵向 (恢复纵向!) ---
    html += "<h3 style='margin-top: 20px;'>5. 综合大师分的纵向与横向解读</h3>"
    top3 = all_scores_df.sort_values('master_score', ascending=False).head(3)
    bottom3 = all_scores_df.sort_values('master_score', ascending=True).head(3)
    html += f"<h4>横向对比 (此刻谁最强/最弱)</h4>"
    html += f"<p><b>👑 冠军: {top3.index[0]} ({top3.iloc[0]['master_score']:.2f})</b><br>🥈 亚军: {top3.index[1]}<br>🥉 季军: {top3.index[2]}</p>"
    html += f"<p><b>🥀 倒数第一: {bottom3.index[0]} ({bottom3.iloc[0]['master_score']:.2f})</b></p>"
    
    html += "<h4>纵向对比 (谁在变好/变坏)</h4>"
    biggest_improver = accelerating.head(1)
    biggest_worsener = decelerating.head(1)
    if not biggest_improver.empty: html += f"<p><b>📈 趋势改善最快: {biggest_improver.index[0]}</b>。加速度为正，基本面或情绪正在好转。</p>"
    if not biggest_worsener.empty: html += f"<p><b>📉 趋势恶化最快: {biggest_worsener.index[0]}</b>。加速度为负，宏观压力急剧增大。</p>"

    # --- 6. 相关性矩阵解读 (恢复股债解读!) ---
    html += "<h3 style='margin-top: 20px;'>6. 最近 60 日资产回报相关性矩阵解读</h3>"
    if correlation_matrix.empty: html += "<p>数据不足。</p>"
    else:
        try:
            # 股债关系
            if "标普500指数" in correlation_matrix.index and "美国十年期国债收益率" in correlation_matrix.index:
                stock_bond_corr = correlation_matrix.loc["标普500指数", "美国十年期国债收益率"]
                if stock_bond_corr > 0.2: html += f"<p><b>股债关系 (SPX vs TNX): {stock_bond_corr:.2f} (正相关)</b>。避险功能失效，通胀/利率是主导因子，股债双杀风险存续。</p>"
                else: html += f"<p><b>股债关系 (SPX vs TNX): {stock_bond_corr:.2f} (负相关/不相关)</b>。股债跷跷板效应正常，债券具备避险属性。</p>"
            
            corr_unstacked = correlation_matrix.unstack()
            corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
            max_corr = corr_unstacked.idxmax()
            html += f"<p><b>矩阵中最强正相关: {max_corr[0]} vs {max_corr[1]} ({corr_unstacked.max():.2f})</b>。</p>"
        except Exception: pass

    return html


# --- 通用函数 ---
def colorize(val):
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else ('#dc3545' if val < 0 else '#6c757d')
        if abs(val) > 0.7:
             return f'<span style="background-color: #ffc107; color: #343a40; font-weight: bold;">{val:.2f}</span>'
        return f'<span style="color: {color}; font-weight: bold;">{val:.2f}</span>'
    return val

def generate_html_table(df, title):
    if df is None or df.empty: return f"<h2>{title}</h2><p>数据不足。</p>"
    df_display = df.copy()
    df_display.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    formatters = {col: colorize for col in df_display.columns if pd.api.types.is_numeric_dtype(df_display[col])}
    html = df_display.to_html(classes='styled-table', escape=False, border=0, justify='center', formatters=formatters)
    return f"<h2>{title}</h2>\n{html}"

def create_html_report(all_html_sections, filename="foex.html"):
    css_style = """<style>
        body{font-family:"Microsoft YaHei","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;margin:0;padding:2rem;background-color:#f8f9fa;color:#212529}
        h1,h2{color:#343a40;border-bottom:2px solid #dee2e6;padding-bottom:.5rem;margin-top:2rem}
        h1{text-align:center;font-weight:600} h3{color:#0056b3;border-left:4px solid #0056b3;padding-left:10px;} 
        h4{color:#495057;margin-top:1.5rem; border-bottom: 1px dotted #ccc; padding-bottom: 5px;} 
        .container{max-width:1200px;margin:auto;background-color:#fff;padding:2rem;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.1)}
        .timestamp{text-align:center;color:#6c757d;margin-bottom:2rem}
        .styled-table, .pivot-table{border-collapse:collapse;margin:25px 0;font-size:.9em;width:100%;box-shadow:0 0 20px rgba(0,0,0,.1)}
        .styled-table thead tr, .pivot-table thead tr{background-color:#007bff;color:#fff;text-align:center;font-weight:700}
        .styled-table th,.styled-table td, .pivot-table th, .pivot-table td{padding:12px 15px;text-align:center;border:1px solid #ddd}
        .styled-table tbody tr:nth-of-type(even), .pivot-table tbody tr:nth-of-type(even){background-color:#f3f3f3}
        .styled-table td:first-child{text-align:left;font-weight:700}
        .footer{text-align:center;margin-top:2rem;font-size:.8em;color:#6c757d}
        li{line-height:1.8}
    </style>"""
    html_template = f"""
    <!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>全球宏观分析报告 v9.0</title>{css_style}</head>
    <body><div class="container">
        <h1>全球宏观交易决策看板 (v9.0 旗舰版)</h1>
        <p class="timestamp">报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        {''.join(all_html_sections)}
        <div class="footer"><p>由专业级量化分析框架生成</p></div>
    </div></body></html>"""
    try:
        with open(filename, 'w', encoding='utf-8') as f: f.write(html_template)
        print(f"\n报告生成成功！文件已保存为: {filename}")
    except Exception as e:
        print(f"\n错误：写入HTML文件失败。原因: {e}")

# =============================================================================
# 4. 主程序
# =============================================================================
if __name__ == '__main__':
    print("启动全球宏观交易决策看板生成器 (v9.0 旗舰版)...")
    all_tickers = list(set(ALL_ANALYSIS_ASSETS + [BENCHMARK_TICKER]))
    price_data = fetch_data_robust(all_tickers, period="2y")
    html_sections = []
    
    if not price_data.empty and BENCHMARK_TICKER in price_data.columns:
        benchmark_data = price_data[BENCHMARK_TICKER]
        analysis_data = price_data.drop(columns=[BENCHMARK_TICKER], errors='ignore')
        tickers_to_process = [t for t in analysis_data.columns]

        print("\n正在计算所有资产的动量得分...")
        full_analysis_df = calculate_professional_momentum_score(analysis_data[tickers_to_process], benchmark_data)

        if full_analysis_df is not None and not full_analysis_df.empty:
            st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
            mt_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
            if st_col in full_analysis_df.columns and mt_col in full_analysis_df.columns:
                full_analysis_df['acceleration'] = full_analysis_df[st_col] - full_analysis_df[mt_col]
            else: full_analysis_df['acceleration'] = 0

            print("\n正在计算资产相关性矩阵...")
            correlation_assets = G10_CURRENCIES + ["EURJPY=X", "AUDJPY=X", "EURGBP=X", "ES=F", "^TNX", "CL=F", "GC=F", "^VIX"]
            correlation_tickers = [t for t in correlation_assets if t in price_data.columns]
            correlation_matrix = pd.DataFrame()
            if correlation_tickers:
                returns = price_data[correlation_tickers].pct_change().dropna()
                if len(returns) >= 60:
                    correlation_matrix = returns.tail(60).corr()
                    reversed_macro_map = {v: k for k, v in GLOBAL_MACRO_ASSETS.items()}
                    correlation_matrix.rename(index=reversed_macro_map, columns=reversed_macro_map, inplace=True)

            print("\n正在生成市场情绪指标 & 深度交易洞察...")
            html_sections.append(generate_market_sentiment_module(full_analysis_df))
            html_sections.append(generate_deep_dive_analysis_html(full_analysis_df, correlation_matrix))

            print("\n正在生成各资产组的动量排名表...")
            # 分组展示
            group_configs = [
                ("G10直盘动量排名 (相对美元)", G10_CURRENCIES),
                ("新兴市场货币动量排名 (相对美元)", EM_CURRENCIES),
                ("日元交叉盘 (JPY Crosses) 动量排名", [t for t in CROSS_CURRENCIES if "JPY" in t]),
                ("欧系交叉盘 (EUR/GBP Crosses) 动量排名", [t for t in CROSS_CURRENCIES if ("EUR" in t or "GBP" in t) and "JPY" not in t]),
                ("商品货币交叉盘 (AUD/NZD/CAD Crosses) 动量排名", [t for t in CROSS_CURRENCIES if ("AUD" in t or "NZD" in t or "CAD" in t) and "JPY" not in t and "EUR" not in t and "GBP" not in t]),
                ("全球宏观资产动量排名", list(GLOBAL_MACRO_ASSETS.keys()))
            ]
            
            for group_name, group_tickers in group_configs:
                target_asset_names = group_tickers
                group_results = full_analysis_df.loc[full_analysis_df.index.isin(target_asset_names)]
                if not group_results.empty:
                    display_df = group_results.copy() 
                    sorted_results = display_df.sort_values('master_score', ascending=False)
                    z_score_cols = [f'z_score_rs_{p}d' for p in sorted(TIME_PERIODS.values(), reverse=True)]
                    display_cols = ['master_score', 'weighted_z_score_rs'] + z_score_cols + ['acceleration']
                    display_cols_exist = [col for col in display_cols if col in sorted_results.columns]
                    html_sections.append(generate_html_table(sorted_results[display_cols_exist], group_name))

            if not correlation_matrix.empty:
                 html_sections.append(generate_html_table(correlation_matrix, f"深度分析: 重点资产回报相关性矩阵"))

            create_html_report(html_sections)
    else:
        print("\n未能下载分析所需的核心数据，无法生成报告。")
    print("\n分析完成。")
