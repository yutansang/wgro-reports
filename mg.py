# -*- coding: utf-8 -*-
"""
美股深度全景分析报告生成器 (US Stock Deep Dive Edition)
版本: 3.0 (深度叙事逻辑增强版)
新增: 
1. "深度解读"模块：包含核心头条、个股显微镜、宏观背离、风格验证、操作建议。
2. 动态推理引擎：能根据不同行情（普涨、普跌、轮动、背离）自动生成对应的分析文案。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略 pandas 的未来版本警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 配置参数
# =============================================================================

BENCHMARK_TICKER = 'SPY' 
TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

# --- 资产清单 ---
MACRO_INDICATORS = {
    "VIX恐慌指数": "^VIX",
    "十年期美债收益率": "^TNX",
    "美元指数": "UUP",
    "WTI原油": "CL=F"
}

SECTOR_ETFS = {
    "科技 (XLK)": "XLK",
    "通信 (XLC)": "XLC",
    "可选消费 (XLY)": "XLY",
    "金融 (XLF)": "XLF",
    "医疗 (XLV)": "XLV",
    "工业 (XLI)": "XLI",
    "能源 (XLE)": "XLE",
    "必选消费 (XLP)": "XLP",
    "公用事业 (XLU)": "XLU",
    "半导体 (SMH)": "SMH"
}

WATCHLIST_STOCKS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
    "AMD", "AVGO", "TSM", 
    "JPM", "BAC", 
    "LLY", "UNH", 
    "XOM", "CVX", 
    "COST", "WMT", 
    "NFLX", "DIS"
]

ALL_ANALYSIS_ASSETS = list(set(list(MACRO_INDICATORS.values()) + list(SECTOR_ETFS.values()) + WATCHLIST_STOCKS))

COLUMN_TRANSLATIONS = {
    'master_score': '综合大师分 (Alpha)',
    'weighted_z_score_rs': '加权相对Z值',
    'acceleration': '动能加速度',
    f'z_score_rs_{TIME_PERIODS["long_term"]}d': f'{TIME_PERIODS["long_term"]}日相对趋势',
    f'z_score_rs_{TIME_PERIODS["mid_term"]}d': f'{TIME_PERIODS["mid_term"]}日相对趋势',
    f'z_score_rs_{TIME_PERIODS["short_term"]}d': f'{TIME_PERIODS["short_term"]}日相对趋势'
}

COLUMN_ORDER = [
    'master_score', 
    'weighted_z_score_rs', 
    f'z_score_rs_{TIME_PERIODS["long_term"]}d',
    f'z_score_rs_{TIME_PERIODS["mid_term"]}d',
    f'z_score_rs_{TIME_PERIODS["short_term"]}d',
    'acceleration'
]

# =============================================================================
# 2. 数据获取与计算逻辑
# =============================================================================
def fetch_data_robust(tickers, period="2y"):
    print(f"正在下载 {len(tickers)} 个美股资产数据...")
    all_data = []
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by='ticker')
        if len(tickers) == 1:
             df = data['Close'].to_frame()
             df.columns = tickers
             return df
        
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
        print(f"批量下载出错: {e}，尝试逐个下载...")
        for ticker in tickers:
            try:
                d = yf.download(ticker, period=period, auto_adjust=True, progress=False)['Close']
                if not d.empty: all_data.append(d.rename(ticker))
            except: pass
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

def calculate_professional_momentum_score(price_data, benchmark_price):
    results = []
    ticker_to_name = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_ETFS}.items()}
    
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
# 3. HTML 报告生成模块
# =============================================================================

### 市场情绪 ###
def generate_market_sentiment_module(all_scores_df):
    html = "<h2>🏛️ 美股市场情绪指标 (Equity Sentiment Gauge)</h2>"
    def get_z(name):
        for n in [name, MACRO_INDICATORS.get(name), SECTOR_ETFS.get(name)]:
            if n in all_scores_df.index: return all_scores_df.loc[n, 'weighted_z_score_rs']
        return 0

    fear = get_z("VIX恐慌指数") + get_z("十年期美债收益率")
    risk_on = get_z("可选消费 (XLY)") - get_z("必选消费 (XLP)")
    tech = get_z("半导体 (SMH)")
    
    score = risk_on + (tech * 0.5) - (fear * 0.8)
    score = np.clip(score * 2.0, -10, 10)
    
    if score > 7: s, c = "极度贪婪", "#dc3545"
    elif score > 3: s, c = "贪婪", "#ffc107"
    elif score > -3: s, c = "中性", "#6c757d"
    elif score > -7: s, c = "恐惧", "#28a745"
    else: s, c = "极度恐惧", "#17a2b8"
    
    html += f"""
    <div style='text-align:center; margin:20px 0; padding:20px; background:#fff; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
        <div style='font-size:1.5em;'>当前美股情绪: <strong style='color:{c};'>{s}</strong></div>
        <div style='font-size:3.5em; font-weight:bold; margin:15px 0; color:{c}'>{score:.2f}</div>
        <div style='width:80%; margin:auto; background-color:#e9ecef; border-radius:10px; height:25px; position:relative;'>
            <div style='height:100%; width:2px; background-color:#343a40; position:absolute; left:50%;'></div>
            <div style='height:25px; width:25px; background-color:{c}; border:3px solid #fff; border-radius:50%; position:absolute; top:0; left:calc({(score+10)*5}% - 12.5px);'></div>
        </div>
    </div>"""
    return html

### 综合分析 ###
def generate_deep_dive_analysis_html(all_scores_df, correlation_matrix):
    html = "<h2>🧠 智能深度洞察 (AI Deep Dive)</h2>"
    
    # 1. 加速度
    accelerating = all_scores_df.sort_values('acceleration', ascending=False)
    html += "<h3>1. 动能加速榜</h3><ul>"
    for asset, row in accelerating.head(3).iterrows():
        if row['acceleration'] > 0.3:
            html += f"<li><b>🚀 {asset}</b> (加速度: +{row['acceleration']:.2f}): 短期爆发力强，5日趋势显著优于20日趋势。</li>"
    html += "</ul>"

    # 2. 风格扫描
    html += "<h3 style='margin-top:20px;'>2. 风格切换雷达</h3>"
    pivot_groups = [
        {"name": "科技成长 (Growth)", "assets": ["XLK", "SMH", "NVDA", "QQQ"], "desc": "科技/半导体/纳指"},
        {"name": "传统价值 (Value)", "assets": ["XLE", "XLF", "XLI"], "desc": "能源/金融/工业"},
        {"name": "防御避险 (Defensive)", "assets": ["XLP", "XLU", "XLV"], "desc": "公用事业/必选消费"}
    ]
    ticker_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_ETFS}.items()}
    
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

    # 3. 策略 (简单版)
    html += "<h3 style='margin-top:20px;'>3. 交易策略建议</h3>"
    longs = all_scores_df[(all_scores_df['master_score']>3) & (all_scores_df['acceleration']>-0.5)].sort_values('master_score', ascending=False).head(3)
    if not longs.empty:
        html += "<h4>🌟 核心多头</h4><ul>"
        for asset, row in longs.iterrows(): html += f"<li><b>{asset}</b>: 大师分 {row['master_score']:.2f}，趋势稳健。</li>"
        html += "</ul>"

    return html

### [NEW] 深度解读模块 (完全动态逻辑) ###
def generate_deep_interpretation_module(all_scores_df):
    html = "<h2>🧐 深度解读 (Data-Driven Narrative)</h2>"
    ticker_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_ETFS}.items()}

    # 辅助数据获取
    def get_val(name, col):
        real_name = ticker_map.get(name, name)
        if real_name in all_scores_df.index: return all_scores_df.loc[real_name, col]
        return None
    
    # --- 1. 核心头条 (Core Headline) ---
    html += "<h3>1. 核心头条：资金流向何方？</h3>"
    
    # 找出动能最强和最弱的板块
    sectors_df = all_scores_df[all_scores_df.index.isin(SECTOR_ETFS.keys())]
    if not sectors_df.empty:
        best_sector = sectors_df.sort_values('acceleration', ascending=False).iloc[0]
        worst_sector = sectors_df.sort_values('acceleration', ascending=True).iloc[0]
        
        # 判定市场剧本
        headline_text = ""
        if best_sector['acceleration'] > 1.0 and worst_sector['acceleration'] < -1.0:
            headline_text = f"<b>暴力风格切换 (Great Rotation)</b>。资金正在从<b>{worst_sector.name}</b>板块恐慌出逃（加速度 {worst_sector['acceleration']:.2f}），并暴力涌入<b>{best_sector.name}</b>（加速度 +{best_sector['acceleration']:.2f}）。这不是普涨，这是一场血腥的调仓换股。"
        elif best_sector['acceleration'] > 0.5 and worst_sector['acceleration'] > -0.5:
            headline_text = f"<b>多头共振 (Broad Rally)</b>。市场呈现普涨态势，领头羊是<b>{best_sector.name}</b>。并未出现明显的板块溃败，市场风险偏好较高。"
        elif best_sector['acceleration'] < 0.5 and worst_sector['acceleration'] < -1.0:
            headline_text = f"<b>避险模式 (Flight to Safety)</b>。市场缺乏明显的进攻热点，而<b>{worst_sector.name}</b>正在遭受重挫。建议保持谨慎。"
        else:
            headline_text = f"<b>震荡分化</b>。最强的板块是{best_sector.name}，最弱的是{worst_sector.name}，但强度均未达到极端水平，市场处于存量博弈阶段。"
            
        html += f"<p>{headline_text}</p>"

    # --- 2. 个股显微镜 (Stock Microscope) ---
    html += "<h3 style='margin-top:20px;'>2. 个股显微镜：巨头的悲喜</h3><ul>"
    
    # 扫描个股 (WATCHLIST)
    stock_rows = []
    for s in WATCHLIST_STOCKS:
        rn = ticker_map.get(s, s)
        if rn in all_scores_df.index: stock_rows.append(all_scores_df.loc[rn])
    
    if stock_rows:
        stocks_df = pd.DataFrame(stock_rows)
        
        # 场景A: 沉睡巨人 (长期差，短期爆发)
        waking = stocks_df[(stocks_df['master_score'] < -1) & (stocks_df['acceleration'] > 1.0)]
        if not waking.empty:
            s = waking.iloc[0]
            html += f"<li><b>🐂 沉睡巨人苏醒 ({s.name})</b>: 它的总分很低({s['master_score']:.2f})，说明调整了很久。但看它的加速度(+{s['acceleration']:.2f})！这是典型的<b>底部反转</b>信号，右侧交易机会可能已经出现。</li>"
        
        # 场景B: 稳如泰山 (长期好，短期稳)
        steady = stocks_df[(stocks_df['master_score'] > 3) & (stocks_df['acceleration'] > -0.5)].sort_values('master_score', ascending=False)
        if not steady.empty:
            s = steady.iloc[0]
            html += f"<li><b>👑 稳如泰山 ({s.name})</b>: 当之无愧的核心多头。大师分高达 {s['master_score']:.2f}，全周期趋势健康，是持仓的定海神针。</li>"
            
        # 场景C: 掉落的飞刀 (长期差，短期更差)
        falling = stocks_df[(stocks_df['master_score'] < -3) & (stocks_df['acceleration'] < -0.5)].sort_values('master_score', ascending=True)
        if not falling.empty:
            s = falling.iloc[0]
            html += f"<li><b>🔪 掉落的飞刀 ({s.name})</b>: 千万不要去接。大师分极低({s['master_score']:.2f})且还在加速下跌，说明基本面可能有硬伤，<b>坚决回避</b>。</li>"
            
    html += "</ul>"

    # --- 3. 宏观背离警示 (Macro Divergence) ---
    html += "<h3 style='margin-top:20px;'>3. 宏观背离警示</h3>"
    tnx_z = get_val("十年期美债收益率", 'z_score_rs_5d')
    xlk_z = get_val("科技 (XLK)", 'z_score_rs_5d')
    
    if tnx_z is not None and xlk_z is not None:
        if tnx_z > 0.5 and xlk_z > 0.5:
            html += f"<p>⚠️ <b>异常背离！</b>美债收益率飙升(5d Z={tnx_z:.2f})通常利空科技股，但科技股却在顶风作案(5d Z={xlk_z:.2f})。这要么说明科技股基本面强到无视利率，要么是一次<b>不可持续的逼空</b>，需警惕收益率继续上行带来的补跌风险。</p>"
        elif tnx_z < -0.5 and xlk_z > 0.5:
            html += f"<p>✅ <b>顺风顺水。</b>美债收益率下行(5d Z={tnx_z:.2f})，为科技股的上涨提供了完美的流动性环境，这种上涨通常比较健康。</p>"
        elif tnx_z > 0.5 and xlk_z < -0.5:
            html += f"<p>📉 <b>教科书式压制。</b>利率上行(5d Z={tnx_z:.2f})正在精准打击高估值的科技股，这是标准的宏观逻辑，建议等待利率企稳。</p>"
        else:
            html += "<p>当前宏观因子与股市的关系处于正常波动范围，未见显著异常。</p>"
    else: html += "<p>宏观数据不足，无法判定。</p>"

    # --- 4. 风格扫描验证 (Style Check) ---
    html += "<h3 style='margin-top:20px;'>4. 风格扫描验证</h3>"
    # 检查 Value (传统价值) 的状态
    val_assets = ["XLE", "XLF", "XLI"]
    val_acc_sum = 0
    count = 0
    for a in val_assets:
        acc = get_val(a, 'acceleration')
        if acc is not None: 
            val_acc_sum += acc
            count += 1
    
    avg_acc = val_acc_sum / count if count else 0
    if avg_acc > 0.3:
        html += f"<p><b>传统价值 (Value) 正在反攻</b> (平均加速度 +{avg_acc:.2f})。如果此时科技股也在涨，说明是复苏交易；如果科技股在跌，说明是防御性轮动。请结合上文判断。</p>"
    elif avg_acc < -0.3:
        html += f"<p><b>传统价值 (Value) 正在失血</b> (平均加速度 {avg_acc:.2f})。资金可能正在抛弃旧经济，流向成长股或现金。</p>"
    else:
        html += "<p>传统价值风格表现平稳，市场主要矛盾可能集中在成长板块内部。</p>"

    # --- 5. 操作建议 (Actionable) ---
    html += "<h3 style='margin-top:20px;'>5. 操作建议 (基于数据推理)</h3><ul>"
    
    # 动态生成
    top_buy = all_scores_df.sort_values('acceleration', ascending=False).head(1)
    top_sell = all_scores_df.sort_values('acceleration', ascending=True).head(1)
    
    if not top_buy.empty:
        a = top_buy.iloc[0]
        html += f"<li><b>🟢 做多方向</b>: 关注 <b>{a.name}</b>。动能刚刚翻正/加速，爆发力最强，适合追击右侧。</li>"
    if not top_sell.empty:
        a = top_sell.iloc[0]
        html += f"<li><b>🔴 止盈/做空方向</b>: 回避 <b>{a.name}</b>。它正在遭受最剧烈的资金抛售，短期下行惯性极大，不要接飞刀。</li>"
        
    html += "</ul>"
    
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
    if df is None or df.empty: return ""
    df_display = df.copy()
    ordered_cols = [c for c in COLUMN_ORDER if c in df_display.columns]
    df_display = df_display[ordered_cols]
    df_display.rename(columns=COLUMN_TRANSLATIONS, inplace=True)
    
    formatters = {col: colorize for col in df_display.columns if pd.api.types.is_numeric_dtype(df_display[col])}
    html = df_display.to_html(classes='styled-table', escape=False, border=0, justify='center', formatters=formatters)
    return f"<h2>{title}</h2>\n{html}"

def create_html_report(all_html_sections, filename="mg.html"):

    css = """<style>
        body{font-family:"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:2rem;background:#f0f2f5;color:#333}
        h1{text-align:center;color:#1a73e8;border-bottom:3px solid #1a73e8;padding-bottom:10px}
        h2{color:#444;border-left:5px solid #1a73e8;padding-left:10px;margin-top:30px;background:#fff;padding:10px}
        h3{color:#1a73e8;margin-top:20px} h4{color:#d93025;margin-top:15px}
        .container{max-width:1200px;margin:auto;background:#fff;padding:30px;border-radius:12px;box-shadow:0 6px 15px rgba(0,0,0,.05)}
        .styled-table, .pivot-table{width:100%;border-collapse:collapse;margin:20px 0;box-shadow:0 0 10px rgba(0,0,0,0.05)}
        .styled-table th, .pivot-table th{background:#1a73e8;color:#fff;padding:12px;text-align:center}
        .styled-table td, .pivot-table td{padding:10px;border-bottom:1px solid #ddd;text-align:center}
        .styled-table tr:nth-child(even){background:#f9f9f9}
        li{margin-bottom:8px} b{font-weight:700;color:#333}
    </style>"""
    html_t = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>美股报告</title>{css}</head><body><div class='container'><h1>🇺🇸 美股全景交易决策看板 (v3.0 深度解读版)</h1><p style='text-align:center;color:#888'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>{''.join(all_html_sections)}</div></body></html>"
    with open(filename, 'w', encoding='utf-8') as f: f.write(html_t)
    print(f"报告已生成: {filename}")

# =============================================================================
# 4. 主程序
# =============================================================================
if __name__ == '__main__':
    print("启动美股深度分析引擎 (v3.0)...")
    all_tickers = list(set(ALL_ANALYSIS_ASSETS + [BENCHMARK_TICKER]))
    price_data = fetch_data_robust(all_tickers, period="2y")
    
    if not price_data.empty and BENCHMARK_TICKER in price_data.columns:
        benchmark_data = price_data[BENCHMARK_TICKER]
        
        print("正在计算动量...")
        full_analysis_df = calculate_professional_momentum_score(price_data, benchmark_data)
        
        # 全局计算加速度
        st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
        mt_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
        if st_col in full_analysis_df.columns and mt_col in full_analysis_df.columns:
            full_analysis_df['acceleration'] = full_analysis_df[st_col] - full_analysis_df[mt_col]
        else:
            full_analysis_df['acceleration'] = 0
        
        print("正在计算相关性...")
        corr_tickers = [t for t in WATCHLIST_STOCKS[:10] + list(MACRO_INDICATORS.values()) if t in price_data.columns]
        corr_matrix = pd.DataFrame()
        if corr_tickers:
            mapper = {**MACRO_INDICATORS, **SECTOR_ETFS}
            corr_matrix = price_data[corr_tickers].pct_change().dropna().tail(60).corr()
            corr_matrix.rename(index=mapper, columns=mapper, inplace=True)

        html_sections = []
        if not full_analysis_df.empty:
            html_sections.append(generate_market_sentiment_module(full_analysis_df))
            html_sections.append(generate_deep_dive_analysis_html(full_analysis_df, corr_matrix))
            
            # [新增] 插入深度解读模块
            html_sections.append(generate_deep_interpretation_module(full_analysis_df))
            
            groups = [
                ("📊 十大板块动量排名 (vs SPY)", SECTOR_ETFS.values()),
                ("🔥 核心关注个股排名 (vs SPY)", WATCHLIST_STOCKS),
                ("🌍 宏观指标趋势", MACRO_INDICATORS.values())
            ]
            
            reverse_map = {v: k for k, v in {**MACRO_INDICATORS, **SECTOR_ETFS}.items()}
            for title, tickers in groups:
                target_names = []
                for t in tickers:
                    if t in full_analysis_df.index: target_names.append(t)
                    elif reverse_map.get(t) in full_analysis_df.index: target_names.append(reverse_map.get(t))
                
                subset = full_analysis_df.loc[target_names].sort_values('master_score', ascending=False)
                html_sections.append(generate_html_table(subset, title))

        create_html_report(html_sections)
    else:
        print("数据不足，无法生成。")

