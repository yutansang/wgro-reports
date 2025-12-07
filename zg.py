# -*- coding: utf-8 -*-
"""
中概股全景交易决策看板 (China ADR Deep Dive Edition)
版本: 2.0 (深度探索增强版)
新增功能: 
1. [深度探索] 模块：包含核心定调、风格分化、宏观背离、策略建议四个维度。
2. 动态逻辑引擎：能区分"超跌反弹"、"强者恒强"、"阴跌不止"等不同市场状态。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 配置参数
# =============================================================================

BENCHMARK_TICKER = 'SPY' 
TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

# --- 资产清单 ---
MACRO_INDICATORS = {
    "离岸人民币汇率 (USD/CNH)": "CNH=F",
    "富时中国A50指数": "CN", 
    "中概互联ETF (KWEB)": "KWEB",
    "富时中国50ETF (FXI)": "FXI",
    "3倍做多中国 (YINN)": "YINN",
    "纳斯达克金龙中国指数": "PGJ"
}

SECTOR_MAPPING = {
    # 互联网巨头
    "阿里巴巴 (BABA)": "BABA", "拼多多 (PDD)": "PDD", "京东 (JD)": "JD", 
    "百度 (BIDU)": "BIDU", "网易 (NTES)": "NTES", "腾讯控股(ADR)": "TCEHY",
    # 造车新势力
    "蔚来 (NIO)": "NIO", "小鹏 (XPEV)": "XPEV", "理想 (LI)": "LI", "极氪 (ZK)": "ZK",
    # 消费 & 平台
    "贝壳 (BEKE)": "BEKE", "携程 (TCOM)": "TCOM", "百胜中国 (YUMC)": "YUMC", 
    "新东方 (EDU)": "EDU", "唯品会 (VIPS)": "VIPS",
    # 金融 & 高弹性
    "富途控股 (FUTU)": "FUTU", "老虎证券 (TIGR)": "TIGR", 
    "哔哩哔哩 (BILI)": "BILI", "满帮 (YMM)": "YMM"
}

ALL_ANALYSIS_ASSETS = list(set(list(MACRO_INDICATORS.values()) + list(SECTOR_MAPPING.values())))

COLUMN_TRANSLATIONS = {
    'master_score': '综合大师分 (Alpha)',
    'weighted_z_score_rs': '加权相对Z值',
    'acceleration': '动能加速度',
    f'z_score_rs_{TIME_PERIODS["long_term"]}d': f'{TIME_PERIODS["long_term"]}日相对趋势',
    f'z_score_rs_{TIME_PERIODS["mid_term"]}d': f'{TIME_PERIODS["mid_term"]}日相对趋势',
    f'z_score_rs_{TIME_PERIODS["short_term"]}d': f'{TIME_PERIODS["short_term"]}日相对趋势'
}

COLUMN_ORDER = ['master_score', 'weighted_z_score_rs', f'z_score_rs_{TIME_PERIODS["long_term"]}d', f'z_score_rs_{TIME_PERIODS["mid_term"]}d', f'z_score_rs_{TIME_PERIODS["short_term"]}d', 'acceleration']

# =============================================================================
# 2. 数据获取与计算逻辑
# =============================================================================
def fetch_data_robust(tickers, period="2y"):
    print(f"正在下载 {len(tickers)} 个中概股资产数据...")
    all_data = []
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by='ticker')
        if len(tickers) == 1:
             df = data['Close'].to_frame(); df.columns = tickers; return df
        extracted_data = {}
        for ticker in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex): series = data[ticker]['Close']
                else: series = data['Close']
                if not series.empty: extracted_data[ticker] = series
            except KeyError: pass
        if not extracted_data: return pd.DataFrame()
        combined_df = pd.DataFrame(extracted_data)
        combined_df.ffill(inplace=True); combined_df.bfill(inplace=True)
        combined_df.dropna(how='all', axis=0, inplace=True)
        return combined_df
    except Exception as e:
        print(f"批量下载出错: {e}"); return pd.DataFrame()

def calculate_professional_momentum_score(price_data, benchmark_price):
    results = []
    ticker_to_name = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_MAPPING}.items()}
    
    for ticker in price_data.columns:
        if ticker == benchmark_price.name: continue
        asset_price = price_data[ticker]
        aligned_benchmark = benchmark_price.reindex(asset_price.index).ffill()
        
        is_macro = ticker in MACRO_INDICATORS.values()
        relative_price = asset_price if is_macro else (asset_price / aligned_benchmark).dropna()

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
        if len(asset_price) >= lookback_vol:
            annualized_vol = asset_price.pct_change().dropna().tail(lookback_vol).std() * np.sqrt(252)
            metrics['master_score'] = weighted_z_score_sum / annualized_vol if annualized_vol > 0 else 0
        else: continue
        results.append(metrics)

    if not results: return pd.DataFrame()
    df = pd.DataFrame(results).dropna().set_index('Ticker')
    new_index = []
    for t in df.index: new_index.append(ticker_to_name.get(t, t))
    df.index = new_index
    return df

# =============================================================================
# 3. 报告生成模块 (含新增的深度探索)
# =============================================================================

### 市场情绪 ###
def generate_market_sentiment_module(all_scores_df):
    html = "<h2>🐉 中概市场情绪指标 (China Sentiment Gauge)</h2>"
    def get_z(name):
        for n in [name, MACRO_INDICATORS.get(name), SECTOR_MAPPING.get(name)]:
            if n in all_scores_df.index: return all_scores_df.loc[n, 'weighted_z_score_rs']
        return 0

    cnh_z = get_z("离岸人民币汇率 (USD/CNH)")
    currency_pressure = cnh_z * -1.0 
    market_heat = get_z("中概互联ETF (KWEB)")
    leverage_sentiment = get_z("3倍做多中国 (YINN)")

    sentiment_score = (market_heat * 0.5) + (leverage_sentiment * 0.3) + (currency_pressure * 0.8)
    sentiment_score = np.clip(sentiment_score * 1.5, -10, 10)
    
    if sentiment_score > 7: s, c = "极度狂热 (FOMO)", "#dc3545"
    elif sentiment_score > 3: s, c = "乐观 (Bullish)", "#ffc107"
    elif sentiment_score > -3: s, c = "中性 (Neutral)", "#6c757d"
    elif sentiment_score > -7: s, c = "悲观 (Bearish)", "#28a745"
    else: s, c = "极度恐慌 (Panic)", "#17a2b8"
    
    html += f"""
    <div style='text-align:center; margin:20px 0; padding:20px; background:#fff; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
        <div style='font-size:1.5em;'>当前中概情绪: <strong style='color:{c};'>{s}</strong></div>
        <div style='font-size:3.5em; font-weight:bold; margin:15px 0; color:{c}'>{sentiment_score:.2f}</div>
        <div style='width:80%; margin:auto; background-color:#e9ecef; border-radius:10px; height:25px; position:relative;'>
            <div style='height:100%; width:2px; background-color:#343a40; position:absolute; left:50%;'></div>
            <div style='height:25px; width:25px; background-color:{c}; border:3px solid #fff; border-radius:50%; position:absolute; top:0; left:calc({(sentiment_score+10)*5}% - 12.5px);'></div>
        </div>
        <p style='margin-top:15px; font-size:0.9em; color:#666;'>因子解构: 汇率压力({cnh_z:.2f}) | 市场热度({market_heat:.2f})</p>
    </div>"""
    return html

### 原有的 AI Insight ###
def generate_deep_interpretation_module(all_scores_df):
    html = "<h2>🧐 深度解读 (AI Insight)</h2>"
    ticker_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_MAPPING}.items()}
    def get_val(name, col):
        rn = ticker_map.get(name, name)
        return all_scores_df.loc[rn, col] if rn in all_scores_df.index else None
    
    kweb_acc = get_val("中概互联ETF (KWEB)", 'acceleration')
    
    headline = ""
    if kweb_acc is not None and kweb_acc > 0.5:
        headline = f"<b>🚀 核心头条: 暴力反弹中。</b>中概互联(KWEB)动能正在加速(Acc={kweb_acc:.2f})，空头回补正在发生。"
    elif kweb_acc is not None and kweb_acc < -0.5:
        headline = f"<b>📉 核心头条: 阴跌不止。</b>中概资产仍处于失血状态(Acc={kweb_acc:.2f})。"
    else:
        headline = "<b>😴 核心头条: 窄幅震荡。</b>市场缺乏明确方向。"
    html += f"<p>{headline}</p>"
    return html

### [NEW] 深度探索模块 (逻辑增强版) ###
def generate_deep_exploration_module(all_scores_df):
    html = "<h2>🔍 深度探索 (Deep Exploration)</h2>"
    ticker_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_MAPPING}.items()}
    
    # 辅助函数
    def get_val(name, col):
        rn = ticker_map.get(name, name)
        return all_scores_df.loc[rn, col] if rn in all_scores_df.index else None

    # --- 1. 核心定调 (Market Definition) ---
    html += "<h3>1. 核心定调</h3>"
    kweb_score = get_val("中概互联ETF (KWEB)", 'master_score')
    kweb_acc = get_val("中概互联ETF (KWEB)", 'acceleration')
    
    if kweb_score is not None and kweb_acc is not None:
        if kweb_score < -1.0 and kweb_acc > 0.5:
            html += f"<p><b>📈 弱者回血 (Oversold Bounce)</b>。证据：KWEB的长期大师分极低({kweb_score:.2f})，说明处于深熊区间；但加速度为正且强劲(+{kweb_acc:.2f})。<b>结论：</b>这不是牛市归来，而是极度超跌后的<b>修正性反弹/空头回补</b>。</p>"
        elif kweb_score > 1.0 and kweb_acc > 0.3:
            html += f"<p><b>🐂 强者恒强 (Bull Trend)</b>。证据：KWEB大师分为正({kweb_score:.2f})且动能持续加速。<b>结论：</b>中概股处于健康的主升浪中，右侧交易胜率较高。</p>"
        elif kweb_score < -1.0 and kweb_acc < -0.5:
            html += f"<p><b>📉 阴跌中继 (Bear Continuation)</b>。证据：大师分和加速度双负。<b>结论：</b>任何反弹都是死猫跳，市场还在寻底。</p>"
        else:
            html += f"<p><b>⚖️ 混沌震荡</b>。市场信号矛盾，缺乏主线逻辑。</p>"

    # --- 2. 风格剧烈分化 (Style Divergence) ---
    html += "<h3 style='margin-top:20px;'>2. 风格剧烈分化</h3>"
    
    # 计算板块平均加速度
    groups = {
        "互联网 (Tech)": ["BABA", "PDD", "JD", "BIDU"],
        "造车 (EV)": ["NIO", "XPEV", "LI"],
        "消费 (Consumption)": ["YUMC", "TCOM", "EDU"]
    }
    
    group_stats = {}
    for g_name, tickers in groups.items():
        vals = [get_val(t, 'acceleration') for t in tickers if get_val(t, 'acceleration') is not None]
        if vals: group_stats[g_name] = np.mean(vals)
    
    if group_stats:
        best_g = max(group_stats, key=group_stats.get)
        worst_g = min(group_stats, key=group_stats.get)
        gap = group_stats[best_g] - group_stats[worst_g]
        
        if gap > 0.5:
            html += f"<p><b>⚡ 板块撕裂：{best_g} 进攻，{worst_g} 崩塌。</b></p>"
            html += f"<ul><li><b>{best_g}</b>: 平均加速度 <b style='color:#28a745'>+{group_stats[best_g]:.2f}</b>。资金正在抱团该板块进行攻击。</li>"
            html += f"<li><b>{worst_g}</b>: 平均加速度 <b style='color:#dc3545'>{group_stats[worst_g]:.2f}</b>。惨遭资金抛弃，是市场的最大雷点。</li></ul>"
            html += f"<p><b>深度解读：</b>这种极致的分化说明这依然是<b>存量博弈</b>，资金在拆东墙补西墙，并未出现全面普涨。</p>"
        else:
            html += "<p>各板块走势趋同，未出现显著的风格撕裂。</p>"

    # --- 3. 极度危险的宏观背离 (Macro Divergence) ---
    html += "<h3 style='margin-top:20px;'>3. 极度危险的宏观背离（关键警示！）</h3>"
    cnh_z = get_val("离岸人民币汇率 (USD/CNH)", 'z_score_rs_5d') # 5日汇率趋势
    stock_acc = kweb_acc if kweb_acc is not None else 0
    
    if cnh_z is not None:
        # 场景A: 汇率贬值(CNH涨, Z>0.5) + 股市涨(Acc>0.3) = 危险背离
        if cnh_z > 0.5 and stock_acc > 0.3:
            html += f"<p>⚠️ <b>不可持续的背离！</b></p><ul>"
            html += f"<li><b>离岸人民币</b>: 5日趋势 <b style='color:#dc3545'>+{cnh_z:.2f} (加速贬值)</b>。</li>"
            html += f"<li><b>中概互联</b>: 动能加速度 <b style='color:#28a745'>+{stock_acc:.2f} (反弹)</b>。</li></ul>"
            html += f"<p><b>推演：</b>人民币贬值通常严重利空中概。当前的股市反弹是在逆风而行，可能是<b>'逃命波'</b>。一旦汇率压力传导，股市反弹随时可能夭折。</p>"
        
        # 场景B: 汇率升值(CNH跌, Z<-0.5) + 股市涨(Acc>0.3) = 完美共振
        elif cnh_z < -0.5 and stock_acc > 0.3:
            html += f"<p>✅ <b>完美的宏观共振！</b>汇率升值（利好）伴随股市反弹，这是最健康的上涨模式，行情持续性强。</p>"
        
        # 场景C: 汇率贬值 + 股市跌 = 流动性枯竭
        elif cnh_z > 0.5 and stock_acc < -0.3:
            html += f"<p>❄️ <b>戴维斯双杀。</b>汇率贬值叠加股市下跌，外资正在加速流出，深不见底。</p>"
        
        else:
            html += f"<p>宏观因子与股市走势处于正常相关范围，未见极端异常。</p>"

    # --- 4. 交易策略建议 (Actionable) ---
    html += "<h3 style='margin-top:20px;'>4. 交易策略建议 (Tactical)</h3>"
    
    # 寻找多头
    longs = all_scores_df.sort_values('acceleration', ascending=False)
    # 寻找空头
    shorts = all_scores_df.sort_values('acceleration', ascending=True)
    
    html += "<ul>"
    
    # 策略 A: 真多头 (大师分高 + 加速)
    true_bulls = all_scores_df[(all_scores_df['master_score'] > 1) & (all_scores_df['acceleration'] > 0)]
    if not true_bulls.empty:
        s = true_bulls.sort_values('acceleration', ascending=False).iloc[0]
        html += f"<li><b>🟢 稳健做多 ({s.name})</b>: 全场唯一的'真·多头'。大师分({s['master_score']:.2f})为正，属于上升通道中的加速，安全边际最高。</li>"
    
    # 策略 B: 博反弹 (大师分低 + 极速)
    rebounds = all_scores_df[(all_scores_df['master_score'] < -1) & (all_scores_df['acceleration'] > 1.0)]
    if not rebounds.empty:
        s = rebounds.sort_values('acceleration', ascending=False).iloc[0]
        html += f"<li><b>⚡ 短线博反弹 ({s.name})</b>: 弹性之王。虽然长期趋势差，但爆发力(Acc={s['acceleration']:.2f})最强，适合作为Beta工具快进快出。</li>"
        
    # 策略 C: 坚决回避 (大师分低 + 减速)
    avoids = shorts[shorts['master_score'] < -1].head(1)
    if not avoids.empty:
        s = avoids.iloc[0]
        html += f"<li><b>🔴 坚决回避 ({s.name})</b>: 深不见底。大师分低且加速下跌(Acc={s['acceleration']:.2f})，千万别接飞刀。</li>"
        
    html += "</ul>"
    
    return html

### 综合HTML生成 ###
def generate_deep_dive_analysis_html(all_scores_df):
    html = "<h2>📊 板块轮动雷达 (Sector Rotation)</h2>"
    pivot_groups = [
        {"name": "互联网巨头 (Big Tech)", "assets": ["BABA", "PDD", "JD", "BIDU"], "desc": "业绩稳健/估值修复"},
        {"name": "造车新势力 (EV)", "assets": ["NIO", "XPEV", "LI"], "desc": "高波动/高弹性"},
        {"name": "消费复苏 (Consumption)", "assets": ["TCOM", "YUMC", "EDU", "BEKE"], "desc": "内需/政策敏感"}
    ]
    ticker_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_MAPPING}.items()}
    pivot_html = "<table class='pivot-table'><thead><tr><th>板块风格</th><th>长期趋势 (60d)</th><th>短期趋势 (5d)</th><th>状态判定</th></tr></thead><tbody>"
    for group in pivot_groups:
        target_indices = []
        for ticker in group['assets']:
            real_name = ticker_map.get(ticker, ticker)
            if real_name in all_scores_df.index: target_indices.append(real_name)
        if not target_indices: continue
        rows = all_scores_df.loc[target_indices]
        lt, st = rows[f'z_score_rs_{TIME_PERIODS["long_term"]}d'].mean(), rows[f'z_score_rs_{TIME_PERIODS["short_term"]}d'].mean()
        lt_s = "<span style='color:#28a745'>强势</span>" if lt>0 else "<span style='color:#dc3545'>弱势</span>"
        st_s = "<span style='color:#28a745'>走强</span>" if st>0 else "<span style='color:#dc3545'>走弱</span>"
        status = "趋势延续"
        if lt<-0.1 and st>0.1: status="📈 底部反转 (关注)"
        elif lt>0.1 and st<-0.1: status="📉 顶部回撤 (警惕)"
        pivot_html += f"<tr><td><b>{group['name']}</b><br><span style='font-size:0.8em;color:#888'>{group['desc']}</span></td><td>{lt_s} ({lt:.2f})</td><td>{st_s} ({st:.2f})</td><td><b>{status}</b></td></tr>"
    html += pivot_html + "</tbody></table>"
    return html

# --- 样式与辅助 ---
def colorize(val):
    if isinstance(val, (int, float)):
        color = '#28a745' if val > 0 else ('#dc3545' if val < 0 else '#6c757d')
        if abs(val) > 0.7: return f'<span style="background-color: #ffc107; color: #343a40; font-weight: bold;">{val:.2f}</span>'
        return f'<span style="color: {color}; font-weight: bold;">{val:.2f}</span>'
    return val

def generate_html_table(df, title):
    if df is None or df.empty: return ""
    df_display = df.copy()
    ordered_cols = [c for c in COLUMN_ORDER if c in df_display.columns]
    df_display = df_display[ordered_cols]
    df_display.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    formatters = {col: colorize for col in df_display.columns if pd.api.types.is_numeric_dtype(df_display[col])}
    html = df_display.to_html(classes='styled-table', escape=False, border=0, justify='center', formatters=formatters)
    return f"<h2>{title}</h2>\n{html}"

def create_html_report(all_html_sections, filename="中概股深度分析报告.html"):
    css = """<style>
        body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;padding:2rem;background:#f4f4f4;color:#333}
        h1{text-align:center;color:#d93025;border-bottom:3px solid #d93025;padding-bottom:10px} 
        h2{color:#333;border-left:5px solid #d93025;padding-left:10px;margin-top:30px;background:#fff;padding:10px}
        h3{color:#d93025;margin-top:20px} 
        .container{max-width:1200px;margin:auto;background:#fff;padding:30px;border-radius:12px;box-shadow:0 6px 15px rgba(0,0,0,.05)}
        .styled-table, .pivot-table{width:100%;border-collapse:collapse;margin:20px 0;box-shadow:0 0 10px rgba(0,0,0,0.05)}
        .styled-table th, .pivot-table th{background:#d93025;color:#fff;padding:12px;text-align:center}
        .styled-table td, .pivot-table td{padding:10px;border-bottom:1px solid #ddd;text-align:center}
        .styled-table tr:nth-child(even){background:#fff5f5}
        li{margin-bottom:8px} b{font-weight:700;color:#000}
    </style>"""
    html_t = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>中概股报告</title>{css}</head><body><div class='container'><h1>🇨🇳 中概股(ADR)全景交易决策看板</h1><p style='text-align:center;color:#888'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>{''.join(all_html_sections)}</div></body></html>"
    with open(filename, 'w', encoding='utf-8') as f: f.write(html_t)
    print(f"报告已生成: {filename}")

# =============================================================================
# 4. 主程序
# =============================================================================
if __name__ == '__main__':
    print("启动中概股深度分析引擎...")
    all_tickers = list(set(ALL_ANALYSIS_ASSETS + [BENCHMARK_TICKER]))
    price_data = fetch_data_robust(all_tickers, period="2y")
    
    if not price_data.empty and BENCHMARK_TICKER in price_data.columns:
        benchmark_data = price_data[BENCHMARK_TICKER]
        
        print("正在计算Alpha动量...")
        full_analysis_df = calculate_professional_momentum_score(price_data, benchmark_data)
        
        # 全局计算加速度
        st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
        mt_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
        if st_col in full_analysis_df.columns and mt_col in full_analysis_df.columns:
            full_analysis_df['acceleration'] = full_analysis_df[st_col] - full_analysis_df[mt_col]
        else: full_analysis_df['acceleration'] = 0
        
        html_sections = []
        if not full_analysis_df.empty:
            html_sections.append(generate_market_sentiment_module(full_analysis_df))
            html_sections.append(generate_deep_dive_analysis_html(full_analysis_df)) # 原有的板块雷达
            html_sections.append(generate_deep_interpretation_module(full_analysis_df)) # 原有的简报
            html_sections.append(generate_deep_exploration_module(full_analysis_df)) # [NEW] 深度探索
            
            groups = [
                ("🔥 热门中概股动量排名 (vs SPY)", SECTOR_MAPPING.values()),
                ("🌍 宏观与ETF指标", MACRO_INDICATORS.values())
            ]
            reverse_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_MAPPING}.items()}
            for title, tickers in groups:
                target_names = []
                for t in tickers:
                    if t in full_analysis_df.index: target_names.append(t)
                    elif reverse_map.get(t) in full_analysis_df.index: target_names.append(reverse_map.get(t))
                subset = full_analysis_df.loc[target_names].sort_values('master_score', ascending=False)
                html_sections.append(generate_html_table(subset, title))

        create_html_report(html_sections)
    else:
        print("数据不足。")
