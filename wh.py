# -*- coding: utf-8 -*-
"""
中文版网页全球宏观分析报告生成器
版本: 8.6 (加入动能加速度版)
更新:
1. 在最终的排名表中新增了 "动能加速度" 列。
2. 保持了 >0.7 黄色高亮的视觉风格。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略 pandas 的某些未来版本警告，让输出更整洁
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 配置参数
# =============================================================================

BENCHMARK_TICKER = 'UUP'
TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

G10_CURRENCIES = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X"]
EM_CURRENCIES = ["USDZAR=X", "USDMXN=X", "USDBRL=X"]

GLOBAL_MACRO_ASSETS = {
    "标普500指数": "ES=F",
    "MSCI全球指数": "URTH",
    "美国十年期国债收益率": "^TNX",
    "原油": "CL=F",
    "黄金": "GC=F",
    "铜": "HG=F",
    "VIX恐慌指数": "^VIX"
}
ALL_ANALYSIS_ASSETS = list(set(G10_CURRENCIES + EM_CURRENCIES + list(GLOBAL_MACRO_ASSETS.values())))

# [修改 1] 添加加速度的中文翻译
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
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)['Close']
            if not data.empty:
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
# 3. HTML 报告生成模块
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
        <div style='width: 100%; background-color: #e9ecef; border-radius: 5px; text-align: left;'>
            <div style='width: 50%; border-right: 1px solid #fff; display: inline-block; box-sizing: border-box; text-align:center; font-weight:bold;'>恐惧</div>
            <div style='width: 50%; display: inline-block; box-sizing: border-box; text-align:center; font-weight:bold;'>贪婪</div>
        </div>
        <div style='width: 100%; background-color: #e9ecef; border-radius: 5px; height: 20px; position: relative;'>
            <div style='height: 100%; width: 2px; background-color: #343a40; position: absolute; left: 50%;'></div>
            <div style='height: 20px; width: 20px; background-color: {color}; border: 2px solid #fff; border-radius: 50%; position: absolute; top: 0; left: calc({(sentiment_score + 10) * 5}% - 10px);'></div>
        </div>
        <div style='text-align: left; margin-top: 20px; font-size: 0.9em;'>
            <p><b>解读:</b> {status}情绪意味着市场参与者普遍{ '乐观，风险偏好高，但需警惕潜在的回调' if sentiment_score > 3 else ('悲观，避险情绪浓厚，但可能隐藏着逆势机会' if sentiment_score < -3 else '情绪摇摆，在等待更明确的宏观信号') }。</p>
        </div>
    </div>
    """
    return html

def generate_deep_dive_analysis_html(all_scores_df, correlation_matrix):
    title = "综合评估 (智能分析)"
    html = f"<h2>{title}</h2>"
    
    def get_scores(asset_name, df):
        try:
            if asset_name in df.index: return df.loc[asset_name]
            if asset_name in GLOBAL_MACRO_ASSETS.values():
                rev_map = {v: k for k, v in GLOBAL_MACRO_ASSETS.items()}
                return df.loc[rev_map[asset_name]]
            return None
        except KeyError: return None
            
    html += "<h3>1. 动量加速度分析：谁在加速？谁在急刹车？</h3>"
    html += "<p>通过计算“短期动能 (5日)”与“中期动能 (20日)”的差值，我们发现了市场上最拥挤交易的松动。</p>"
    # [注意] 加速度在主函数已经计算，这里直接用
    accelerating = all_scores_df[all_scores_df['acceleration'] > 0.5].sort_values('acceleration', ascending=False)
    decelerating = all_scores_df[all_scores_df['acceleration'] < -0.5].sort_values('acceleration', ascending=True)

    html += "<h4>🚀 加速上涨区 (动能爆发)</h4>"
    if not accelerating.empty:
        html += "<ul style='list-style-type: none; padding-left: 0;'>"
        for asset, row in accelerating.head(2).iterrows(): html += f"<li style='margin-bottom: 15px;'><b>{asset}</b>：<b>加速度第一 ({row['acceleration']:.2f})</b>。这是一个强力信号。该资产动能从60日的({row[f'z_score_rs_{TIME_PERIODS['long_term']}d']:.2f})区间暴力拉升至5日的({row[f'z_score_rs_{TIME_PERIODS['short_term']}d']:.2f})，表明市场正在形成新的共识。</li>"
        html += "</ul>"
    else: html += "<p>当前未发现显著的动能爆发资产。</p>"

    html += "<h4>🛑 急剧减速区 (动能衰竭)</h4>"
    if not decelerating.empty:
        html += "<ul style='list-style-type: none; padding-left: 0;'>"
        for asset, row in decelerating.head(2).iterrows(): html += f"<li style='margin-bottom: 15px;'><b>{asset}</b>：<b>减速第一 ({row['acceleration']:.2f})</b>。这是最显著的逆转信号。虽然其综合大师分仍高({row['master_score']:.2f})，但5日Z值已跌至({row[f'z_score_rs_{TIME_PERIODS['short_term']}d']:.2f})。这意味着围绕该资产的拥挤交易正在快速瓦解。</li>"
        html += "</ul>"
    else: html += "<p>当前未发现显著的动能衰竭资产。</p>"

    html += "<h3 style='margin-top: 20px;'>2. 趋势反转扫描 (The \"Pivot\" List)</h3>"
    html += "<p>通过对比60日趋势与5日趋势的符号差异，我们识别出正在发生<b>根本性方向逆转</b>的资产：</p>"
    pivot_groups = [{"name": "非美货币 (AUD, NZD, GBP, EUR)", "assets": ["AUDUSD=X", "NZDUSD=X", "GBPUSD=X", "EURUSD=X"], "interpretation": "美元霸权松动。资金正在从美元流出，回流至欧系和商品货币。"}, {"name": "标普500 / MSCI全球", "assets": ["标普500指数", "MSCI全球指数"], "interpretation": "尽管中期趋势偏弱，但短期试图反攻。需警惕美债收益率上涨是否会扼杀此反弹。"}, {"name": "原油", "assets": ["原油"], "interpretation": "能源板块可能成为短期阿尔法收益的来源。"}]
    pivot_results = []
    for group in pivot_groups:
        group_assets_df = all_scores_df[all_scores_df.index.isin(group['assets'])]
        if group_assets_df.empty: continue
        long_term_col, short_term_col = f'z_score_rs_{TIME_PERIODS["long_term"]}d', f'z_score_rs_{TIME_PERIODS["short_term"]}d'
        if (group_assets_df[long_term_col] < -0.1).all() and (group_assets_df[short_term_col] > 0.1).all():
            pivot_results.append({"asset": group['name'], "old_world": "📉 弱势 (负值)", "new_world": "📈 转强 (正值)", "signal": group['interpretation']})
        if (group_assets_df[long_term_col] > 0.1).all() and (group_assets_df[short_term_col] < -0.1).all():
             pivot_results.append({"asset": group['name'], "old_world": "📈 强势 (正值)", "new_world": "📉 转弱 (负值)", "signal": "市场风向转变，前期强势资产开始面临抛压。"})
    if not pivot_results: html += "<p>当前未发现明确的趋势反转信号组。</p>"
    else:
        html += "<table class='pivot-table'><thead><tr><th>资产</th><th>60日趋势 (旧世界)</th><th>5日趋势 (新世界)</th><th>信号解读</th></tr></thead><tbody>"
        for item in pivot_results: html += f"<tr><td>{item['asset']}</td><td>{item['old_world']}</td><td>{item['new_world']}</td><td>{item['signal']}</td></tr>"
        html += "</tbody></table>"

    html += "<h3 style='margin-top: 20px;'>3. 宏观因子合成与相关性警示</h3>"
    strong_usd_map = {"USDJPY=X": 1, "USDCHF=X": 1, "USDCAD=X": 1, "USDZAR=X": 1, "USDMXN=X": 1, "USDBRL=X": 1, "EURUSD=X": -1, "GBPUSD=X": -1, "AUDUSD=X": -1, "NZDUSD=X": -1}
    usd_strength_score = 0; count = 0
    for asset, direction in strong_usd_map.items():
        if asset in all_scores_df.index:
            usd_strength_score += all_scores_df.loc[asset, 'weighted_z_score_rs'] * direction
            count += 1
    usd_strength_score = usd_strength_score / count if count > 0 else 0
    usd_status = "转强" if usd_strength_score > 0.3 else ("转弱" if usd_strength_score < -0.3 else "震荡")
    html += f"<h4>🇺🇸 美元强度合成指数：{usd_strength_score:.2f} ({usd_status})</h4>"
    risk_assets = ["标普500指数", "铜", "MSCI全球指数", "AUDUSD=X"]
    risk_score = 0; count = 0
    for asset in risk_assets:
        scores = get_scores(asset, all_scores_df)
        if scores is not None:
             risk_score += scores['weighted_z_score_rs']
             count += 1
    risk_score = risk_score / count if count > 0 else 0
    risk_status = "强劲扩张" if risk_score > 0.5 else ("温和复苏" if risk_score > 0 else "收缩")
    html += f"<h4>🐂 风险偏好合成指数：{risk_score:.2f} ({risk_status})</h4>"
    spx_scores = get_scores("标普500指数", all_scores_df)
    vix_scores = get_scores("VIX恐慌指数", all_scores_df)
    if spx_scores is not None and vix_scores is not None:
        if spx_scores['weighted_z_score_rs'] < 0 and vix_scores['weighted_z_score_rs'] < -0.1:
            html += "<h4>⚠️ 市场异常警示：SPX vs VIX 背离</h4>"
            html += f"<p><b>异常点</b>：标普500的加权Z值是负的({spx_scores['weighted_z_score_rs']:.2f})，表现平平；但VIX的Z值也是负的({vix_scores['weighted_z_score_rs']:.2f})，非常低。</p>"
            html += "<p><b>深度含义</b>：通常股市跌VIX应该涨。现在的状况是<b>“市场下跌但并不恐慌”</b> (Complacency)。这种低波动率的下跌往往掩盖了风险，一旦有外部冲击，VIX可能会出现报复性反弹。</p>"

    html += "<h3 style='margin-top: 20px;'>4. 交易策略启示 (Actionable Insights)</h3>"
    html += "<h4>- 核心多头建议 (Core Longs)</h4>"
    z_cols = [f'z_score_rs_{p}d' for p in TIME_PERIODS.values()]
    core_longs = all_scores_df[(all_scores_df['master_score'] > 5) & (all_scores_df[z_cols] > 0).all(axis=1)].sort_values('master_score', ascending=False)
    if not core_longs.empty:
        html += "<ul>"
        for asset, row in core_longs.head(2).iterrows(): html += f"<li><b>做多 {asset}</b>: <b>逻辑 &rarr;</b> 趋势健康，确定性高。该资产不仅大师分极高({row['master_score']:.2f})，且全周期(5/20/60日)Z值均为正，表明其上涨趋势获得了长、中、短期的一致确认。</li>"
        html += "</ul>"
    else: html += "<p>暂无符合“核心多头”标准的资产(要求大师分>5且全周期Z值为正)。</p>"
    html += "<h4>- 核心空头建议 (Core Shorts)</h4>"
    core_shorts = all_scores_df[(all_scores_df['master_score'] < -5) & (all_scores_df[z_cols] < 0).all(axis=1)].sort_values('master_score', ascending=True)
    if not core_shorts.empty:
        html += "<ul>"
        for asset, row in core_shorts.head(2).iterrows(): html += f"<li><b>做空 {asset}</b>: <b>逻辑 &rarr;</b> 高质量的下跌趋势。该资产大师分极低({row['master_score']:.2f})，且全周期Z值均为负，是典型的“价值陷阱”或“宏观弃子”，短期内难有起色。</li>"
        html += "</ul>"
    else: html += "<p>暂无符合“核心空头”标准的资产(要求大师分<-5且全周期Z值为负)。</p>"
    html += "<h4>- 战术机会 (Tactical Plays)</h4>"
    html += "<ul>"
    tactical_insights = 0
    if not decelerating.empty:
        asset, row = decelerating.iloc[0], decelerating.iloc[0]
        html += f"<li><b>(逆势) 押注 {decelerating.index[0]} 趋势衰竭</b>: <b>逻辑 &rarr;</b> 捕捉拥挤交易的瓦解。该资产是当前**动能减速最快**({row['acceleration']:.2f})的品种。虽然主趋势仍在，但这是趋势末期的典型信号，适合风险偏好较高的投资者进行逆势操作。</li>"
        tactical_insights += 1
    bullish_reversal_assets = [p['asset'] for p in pivot_results if "转强" in p['new_world']]
    if bullish_reversal_assets:
        html += f"<li><b>(顺势) 跟随 {bullish_reversal_assets[0]} 的看涨反转</b>: <b>逻辑 &rarr;</b> 抓住新趋势的起点。该资产组已出现明确的“旧世界(弱) vs 新世界(强)”反转信号，适合希望尽早布局新趋势的交易者。</li>"
        tactical_insights += 1
    if tactical_insights == 0: html += "<li>当前市场处于趋势的稳定期，暂无显著的战术性(逆势或反转)机会。</li>"
    html += "</ul>"
    
    html += "<h4>- 应回避的资产 (Avoid List)</h4>"
    html += "<ul>"
    avoid_insights = 0
    bearish_reversal_assets = all_scores_df[(all_scores_df[f'z_score_rs_{TIME_PERIODS["long_term"]}d'] > 0.5) & (all_scores_df[f'z_score_rs_{TIME_PERIODS["short_term"]}d'] < -1.0)]
    if not bearish_reversal_assets.empty:
        for asset, row in bearish_reversal_assets.iterrows():
            html += f"<li><b>{asset}</b>: <b>逻辑 &rarr;</b> 多空陷阱。长期趋势(60d)向上，但短期(5d)抛压巨大，方向矛盾，是典型的“多头不死，空头不止”拉锯战，极易双向亏损。</li>"
            avoid_insights += 1
    choppy_assets = all_scores_df[(all_scores_df['master_score'].abs() < 1) & (all_scores_df['acceleration'].abs() < 0.3)]
    if not choppy_assets.empty:
        for asset, row in choppy_assets.iterrows():
            html += f"<li><b>{asset}</b>: <b>逻辑 &rarr;</b> 无明确方向。该资产大师分和加速度都接近于零，市场对其没有明确看法，处于“垃圾时间”，交易价值很低。</li>"
            avoid_insights += 1
    if avoid_insights == 0: html += "<li>当前所有受监控资产均有较明确的趋势或风险信号。</li>"
    html += "</ul>"

    html += "<h3 style='margin-top: 20px;'>5. 综合大师分的纵向与横向解读</h3>"
    html += "<h4>横向对比 (此刻谁最强/最弱)</h4>"
    top3 = all_scores_df.sort_values('master_score', ascending=False).head(3)
    bottom3 = all_scores_df.sort_values('master_score', ascending=True).head(3)
    html += f"<p>“大师分”衡量了资产经波动率调整后的相对动量。此刻，<b>全市场表现最强的资产是 {top3.index[0]} ({top3.iloc[0]['master_score']:.2f})</b>，其次是 {top3.index[1]} 和 {top3.index[2]}。<b>表现最弱的是 {bottom3.index[0]} ({bottom3.iloc[0]['master_score']:.2f})</b>。</p>"
    html += "<h4>纵向对比 (谁在变好/变坏)</h4>"
    biggest_improver = accelerating.head(1)
    biggest_worsener = decelerating.head(1)
    if not biggest_improver.empty: html += f"<p><b>趋势改善最快: {biggest_improver.index[0]}</b>。其动量加速度为正，表明其基本面或市场情绪在近期得到了显著的、超越其他资产的改善。</p>"
    if not biggest_worsener.empty: html += f"<p><b>趋势恶化最快: {biggest_worsener.index[0]}</b>。其动量加速度为负，表明其面临的宏观压力在近期急剧增大。</p>"
    html += "<h3 style='margin-top: 20px;'>6. 最近 60 日资产回报相关性矩阵解读</h3>"
    if correlation_matrix.empty: html += "<p>数据不足，无法进行相关性解读。</p>"
    else:
        try:
            stock_bond_corr = correlation_matrix.loc["标普500指数", "美国十年期国债收益率"]
            if stock_bond_corr > 0.2: html += f"<p><b>核心关系：股债“避险”功能失效。</b>标普500与十年期国债收益率呈正相关({stock_bond_corr:.2f})，意味着通胀是市场主线，股市下跌时，债券因加息预期也在跌，传统60/40组合失效。</p>"
            else: html += f"<p><b>核心关系：股债“跷跷板”效应良好。</b>标普500与十年期国债收益率呈负相关({stock_bond_corr:.2f})，市场处于典型的“风险开/关”模式，债券的避险属性良好。</p>"
            corr_unstacked = correlation_matrix.unstack()
            corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
            max_corr, min_corr = corr_unstacked.idxmax(), corr_unstacked.idxmin()
            html += f"<p><b>最强正相关: {max_corr[0]} vs {max_corr[1]} ({corr_unstacked.max():.2f})</b>。这两个资产高度同涨同跌，可能受同一宏观因子驱动。</p>"
            html += f"<p><b>最强负相关: {min_corr[0]} vs {min_corr[1]} ({corr_unstacked.min():.2f})</b>。这两个资产是绝佳的对冲组合。</p>"
        except KeyError: html += "<p>关键资产（如标普500或美债收益率）数据不足，无法进行核心关系解读。</p>"

    return html


# --- 通用函数 ---
def colorize(val):
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else ('#dc3545' if val < 0 else '#6c757d')
        # [保留] 显著值高亮
        if abs(val) > 0.7:
             return f'<span style="background-color: #ffc107; color: #343a40; font-weight: bold;">{val:.2f}</span>'
        return f'<span style="color: {color}; font-weight: bold;">{val:.2f}</span>'
    return val

def generate_html_table(df, title):
    if df is None or df.empty: return f"<h2>{title}</h2><p>数据不足，无法生成此部分报告。</p>"
    df_display = df.copy()
    df_display.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    formatters = {col: colorize for col in df_display.columns if pd.api.types.is_numeric_dtype(df_display[col])}
    html = df_display.to_html(classes='styled-table', escape=False, border=0, justify='center', formatters=formatters)
    return f"<h2>{title}</h2>\n{html}"

def create_html_report(all_html_sections, filename="wh.html"):

    css_style = """<style>
        body{font-family:"Microsoft YaHei","Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;margin:0;padding:2rem;background-color:#f8f9fa;color:#212529}
        h1,h2{color:#343a40;border-bottom:2px solid #dee2e6;padding-bottom:.5rem;margin-top:2rem}
        h1{text-align:center;font-weight:600} h3{color:#0056b3;border-left:4px solid #0056b3;padding-left:10px;} 
        h4{color:#495057;margin-top:1.5rem; border-bottom: 1px dotted #ccc; padding-bottom: 5px;} 
        h5{color:#6c757d;font-style:italic;}
        .container{max-width:1200px;margin:auto;background-color:#fff;padding:2rem;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.1)}
        .timestamp{text-align:center;color:#6c757d;margin-bottom:2rem}
        .styled-table, .pivot-table{border-collapse:collapse;margin:25px 0;font-size:.9em;width:100%;box-shadow:0 0 20px rgba(0,0,0,.1)}
        .styled-table thead tr, .pivot-table thead tr{background-color:#007bff;color:#fff;text-align:center;font-weight:700}
        .styled-table th,.styled-table td, .pivot-table th, .pivot-table td{padding:12px 15px;text-align:center;border:1px solid #ddd}
        .pivot-table td:last-child{text-align:left}
        .styled-table tbody tr, .pivot-table tbody tr{border-bottom:1px solid #ddd}
        .styled-table tbody tr:nth-of-type(even), .pivot-table tbody tr:nth-of-type(even){background-color:#f3f3f3}
        .styled-table tbody tr:last-of-type, .pivot-table tbody tr:last-of-type{border-bottom:2px solid #007bff}
        .styled-table th:first-child,.styled-table td:first-child{text-align:left;font-weight:700}
        .footer{text-align:center;margin-top:2rem;font-size:.8em;color:#6c757d}
        li{line-height:1.8}
    </style>"""
    html_template = f"""
    <!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>全球宏观分析报告 v8.6</title>{css_style}</head>
    <body><div class="container">
        <h1>全球宏观交易决策看板 (v8.6 加速度增强版)</h1>
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
    print("启动全球宏观交易决策看板生成器 (v8.6)...")
    all_tickers = list(set(ALL_ANALYSIS_ASSETS + [BENCHMARK_TICKER]))
    price_data = fetch_data_robust(all_tickers, period="2y")
    html_sections = []
    
    if not price_data.empty and BENCHMARK_TICKER in price_data.columns:
        benchmark_data = price_data[BENCHMARK_TICKER]
        analysis_data = price_data.drop(columns=[BENCHMARK_TICKER], errors='ignore')
        tickers_to_process = [t for t in analysis_data.columns]

        if not tickers_to_process:
            print("\n错误：过滤掉基准后，没有可供分析的资产。")
        else:
            print("\n正在计算所有资产的动量得分...")
            full_analysis_df = calculate_professional_momentum_score(analysis_data[tickers_to_process], benchmark_data)

            # [修改 2] 在全局计算动能加速度 (5d - 20d)，确保后续所有表格都能用到
            if full_analysis_df is not None and not full_analysis_df.empty:
                st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
                mt_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
                if st_col in full_analysis_df.columns and mt_col in full_analysis_df.columns:
                    full_analysis_df['acceleration'] = full_analysis_df[st_col] - full_analysis_df[mt_col]
                else:
                    full_analysis_df['acceleration'] = 0

            print("\n正在计算资产相关性矩阵...")
            correlation_assets = G10_CURRENCIES + ["USDMXN=X", "ES=F", "^TNX", "CL=F", "GC=F", "^VIX"]
            correlation_tickers = [t for t in correlation_assets if t in price_data.columns]
            correlation_matrix = pd.DataFrame()
            if correlation_tickers:
                returns = price_data[correlation_tickers].pct_change().dropna()
                if len(returns) >= 60:
                    correlation_matrix = returns.tail(60).corr()
                    reversed_macro_map = {v: k for k, v in GLOBAL_MACRO_ASSETS.items()}
                    correlation_matrix.rename(index=reversed_macro_map, columns=reversed_macro_map, inplace=True)

            if full_analysis_df is not None and not full_analysis_df.empty:
                print("\n正在生成市场情绪指标...")
                html_sections.append(generate_market_sentiment_module(full_analysis_df))
                
                print("\n正在生成深度交易洞察分析...")
                html_sections.append(generate_deep_dive_analysis_html(full_analysis_df, correlation_matrix))

            
            print("\n正在生成各资产组的动量排名表...")
            for group_name, group_tickers in [("G10货币动量排名 (相对美元指数)", G10_CURRENCIES), 
                                               ("新兴市场货币动量排名 (相对美元指数)", EM_CURRENCIES), 
                                               ("全球宏观资产动量排名 (相对美元指数)", list(GLOBAL_MACRO_ASSETS.keys()))]:
                
                target_asset_names = group_tickers
                group_results = full_analysis_df.loc[full_analysis_df.index.isin(target_asset_names)]

                if not group_results.empty:
                    # [修改 3] 不再 drop 加速度列
                    display_df = group_results.copy() 
                    sorted_results = display_df.sort_values('master_score', ascending=False)
                    
                    # 定义展示列 (把加速度放进去)
                    z_score_cols = [f'z_score_rs_{p}d' for p in sorted(TIME_PERIODS.values(), reverse=True)]
                    display_cols = ['master_score', 'weighted_z_score_rs'] + z_score_cols + ['acceleration']
                    
                    display_cols_exist = [col for col in display_cols if col in sorted_results.columns]
                    html_sections.append(generate_html_table(sorted_results[display_cols_exist], group_name))

            if not correlation_matrix.empty:
                 html_sections.append(generate_html_table(correlation_matrix, f"深度分析: 最近 {60} 日资产回报相关性矩阵"))

            create_html_report(html_sections)
    else:
        print("\n未能下载分析所需的核心数据，无法生成报告。")
    print("\n分析完成。")

