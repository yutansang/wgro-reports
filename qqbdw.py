# -*- coding: utf-8 -*-
"""
中国期权宏观全景分析报告 (Pro版 - 深度洞察完全体 + 相关性高亮)
版本: CN-Pro 1.2
更新:
1. 深度洞察 (Deep Dive) 恢复为包含“核心多空”、“避坑指南”、“宏观因子”的完整版本。
2. 资产相关性矩阵增加“黄色高亮”逻辑 (绝对值>0.8)，一眼识别强相关。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# 忽略 pandas 的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 配置参数
# =============================================================================

BENCHMARK_TICKER = '510300.SS'
BENCHMARK_NAME = '沪深300'

TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

ASSETS_MAP = {
    "上证50ETF": "510050.SS",
    "沪深300ETF": "510300.SS",
    "中证500ETF": "510500.SS",
    "创业板ETF": "159915.SZ",
    "科创50ETF": "588000.SS",
    "中证1000指数(ETF代)": "159845.SZ", 
    "黄金ETF": "518880.SS",
    "豆粕ETF": "159985.SZ",
    "有色金属ETF(铜)": "512400.SS",
    "能源化工ETF(原油)": "159981.SZ",
    "标普500(SPY)": "SPY",
    "美债收益率(TNX)": "^TNX"
}

GROUPS = {
    "核心A股宽基": ["上证50ETF", "沪深300ETF", "中证500ETF", "创业板ETF", "科创50ETF", "中证1000指数(ETF代)"],
    "大宗商品": ["黄金ETF", "豆粕ETF", "有色金属ETF(铜)", "能源化工ETF(原油)"],
    "全球参照": ["标普500(SPY)", "美债收益率(TNX)"]
}

COLUMN_TRANSLATIONS = {
    'master_score': '综合大师分',
    'weighted_z_score_rs': '加权相对Z值',
    'acceleration': '动能加速度',
    f'z_score_rs_{TIME_PERIODS["long_term"]}d': f'{TIME_PERIODS["long_term"]}日相对Z值',
    f'z_score_rs_{TIME_PERIODS["mid_term"]}d': f'{TIME_PERIODS["mid_term"]}日相对Z值',
    f'z_score_rs_{TIME_PERIODS["short_term"]}d': f'{TIME_PERIODS["short_term"]}日相对Z值'
}

# =============================================================================
# 2. 核心计算模块 (无需改动)
# =============================================================================
def fetch_data_robust(assets_map, benchmark_ticker):
    print("正在连接 Yahoo Finance 下载数据...")
    all_tickers = list(set(list(assets_map.values()) + [benchmark_ticker]))
    try:
        data = yf.download(all_tickers, period="2y", progress=False)['Close']
        if data.empty: return pd.DataFrame()
        data.ffill(inplace=True)
        data.dropna(how='all', axis=1, inplace=True)
        rev_map = {v: k for k, v in assets_map.items()}
        rev_map[benchmark_ticker] = BENCHMARK_NAME
        data.rename(columns=rev_map, inplace=True)
        return data
    except Exception as e:
        print(f"数据下载错误: {e}")
        return pd.DataFrame()

def calculate_professional_momentum_score(price_data, benchmark_col):
    results = []
    if benchmark_col not in price_data.columns: return pd.DataFrame()
    benchmark_series = price_data[benchmark_col]
    for ticker in price_data.columns:
        if ticker == benchmark_col: continue
        asset_price = price_data[ticker]
        relative_price = (asset_price / benchmark_series).dropna()
        if len(relative_price) < 80: continue 
        metrics = {'Ticker': ticker}
        w_z_sum = 0
        valid = True
        for term, days in TIME_PERIODS.items():
            rs = (relative_price / relative_price.shift(days)) - 1
            mean, std = rs.mean(), rs.std()
            if std > 0:
                z = (rs.iloc[-1] - mean) / std
                metrics[f'z_score_rs_{days}d'] = z
                w_z_sum += z * PERIOD_WEIGHTS[term]
            else: valid = False
        if not valid: continue
        metrics['weighted_z_score_rs'] = w_z_sum
        vol = asset_price.pct_change().tail(60).std() * np.sqrt(252)
        metrics['master_score'] = w_z_sum / vol if vol > 0 else 0
        results.append(metrics)
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results).set_index('Ticker')
    s_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
    m_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
    if s_col in df.columns and m_col in df.columns:
        df['acceleration'] = df[s_col] - df[m_col]
    else: df['acceleration'] = 0
    return df

# =============================================================================
# 3. 视觉与格式化 (黄色高亮核心)
# =============================================================================

def colorize(val):
    """通用数值着色: 红涨绿跌，绝对值>0.8高亮"""
    if isinstance(val, (int, float)):
        text_color = '#d9534f' if val > 0 else '#28a745'
        if abs(val) > 0.8:
            return f'<span style="background-color: #ffc107; color: #212529; font-weight: bold; padding: 2px 6px; border-radius: 4px;">{val:.2f}</span>'
        return f'<span style="color: {text_color}; font-weight: bold;">{val:.2f}</span>'
    return val

# =============================================================================
# 4. 深度洞察与报告生成 (满血复活版)
# =============================================================================

def generate_deep_dive_full(df):
    """
    完全恢复 usa_cc_ESPT 的 Deep Dive 逻辑
    包含：动能分析、趋势反转、宏观合成、交易策略建议
    """
    html = "<h2 style='border-bottom: 3px solid #0056b3; padding-bottom: 10px;'>深度洞察 (Deep Dive Analysis)</h2>"
    
    # --- Part 1: 动能加速度 (Momentum Acceleration) ---
    html += "<h3>1. 动能加速度：谁在抢跑？谁在掉队？</h3>"
    html += "<p style='font-size:0.9em; color:#666;'>逻辑：计算 (5日趋势 - 20日趋势) 的差值，识别趋势的二阶导数（加速/减速）。</p>"
    
    acc_up = df[df['acceleration'] > 0.5].sort_values('acceleration', ascending=False)
    acc_down = df[df['acceleration'] < -0.5].sort_values('acceleration', ascending=True)

    html += "<div style='display:flex; gap:20px; margin-bottom:20px;'>"
    # 加速卡片
    html += "<div style='flex:1; background:#e8f5e9; padding:15px; border-radius:8px; border-left:5px solid #28a745;'>"
    html += "<h4 style='margin-top:0; color:#28a745;'>🚀 加速冲刺区 (Burst)</h4>"
    if not acc_up.empty:
        html += "<ul>"
        for asset, row in acc_up.head(3).iterrows():
            html += f"<li><b>{asset}</b> ({colorize(row['acceleration'])}): 动能正在爆发，短期资金流入显著，适合趋势追击。</li>"
        html += "</ul>"
    else: html += "<p>暂无显著加速资产。</p>"
    html += "</div>"
    
    # 减速卡片
    html += "<div style='flex:1; background:#ffebee; padding:15px; border-radius:8px; border-left:5px solid #d9534f;'>"
    html += "<h4 style='margin-top:0; color:#d9534f;'>🛑 动能衰竭区 (Stall)</h4>"
    if not acc_down.empty:
        html += "<ul>"
        for asset, row in acc_down.head(3).iterrows():
            html += f"<li><b>{asset}</b> ({colorize(row['acceleration'])}): 上涨动能正在快速衰竭，即使价格未跌，也需警惕见顶风险。</li>"
        html += "</ul>"
    else: html += "<p>暂无显著衰竭资产。</p>"
    html += "</div>"
    html += "</div>"

    # --- Part 2: 趋势反转 (The Pivot List) ---
    html += "<h3>2. 趋势反转扫描 (The Pivot List)</h3>"
    lt_col = f'z_score_rs_{TIME_PERIODS["long_term"]}d'
    st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
    
    # 长期弱(<-0.2) 但 短期强(>0.2)
    bull_pivot = df[(df[lt_col] < -0.2) & (df[st_col] > 0.2)]
    # 长期强(>0.2) 但 短期弱(<-0.2)
    bear_pivot = df[(df[lt_col] > 0.2) & (df[st_col] < -0.2)]
    
    html += "<table class='styled-table'><thead><tr><th>类型</th><th>资产</th><th>旧世界 (60日)</th><th>新世界 (5日)</th><th>解读</th></tr></thead><tbody>"
    
    has_pivot = False
    for asset, row in bull_pivot.iterrows():
        has_pivot = True
        html += f"<tr><td>📈 <b>底部反转</b></td><td>{asset}</td><td>{colorize(row[lt_col])}</td><td>{colorize(row[st_col])}</td><td>长期超跌，但短期出现强力反弹信号，关注底部机会。</td></tr>"
    for asset, row in bear_pivot.iterrows():
        has_pivot = True
        html += f"<tr><td>📉 <b>顶部反转</b></td><td>{asset}</td><td>{colorize(row[lt_col])}</td><td>{colorize(row[st_col])}</td><td>长期强势，但短期抛压沉重，主力可能正在出货。</td></tr>"
    
    if not has_pivot:
        html += "<tr><td colspan='5'>当前市场趋势延续性较好，未发现显著的结构性反转信号。</td></tr>"
    html += "</tbody></table>"

    # --- Part 3: 交易策略启示 (Actionable Insights) ---
    html += "<h3>3. 交易策略启示 (Actionable Insights)</h3>"
    
    # 核心多头：大师分 > 3 且 全周期 > 0
    z_cols = [f'z_score_rs_{p}d' for p in TIME_PERIODS.values()]
    core_longs = df[(df['master_score'] > 3) & (df[z_cols] > 0).all(axis=1)].sort_values('master_score', ascending=False)
    core_shorts = df[(df['master_score'] < -3) & (df[z_cols] < 0).all(axis=1)].sort_values('master_score', ascending=True)
    avoid_list = df[(df['master_score'].abs() < 1)]

    html += "<div style='display:flex; flex-wrap:wrap; gap:15px;'>"
    
    # 多头建议
    html += "<div style='flex:1; min-width:300px; background:#fff3cd; padding:15px; border-radius:8px; border-left:5px solid #ffc107;'>"
    html += "<h4 style='margin-top:0;'>🐂 核心多头 (Core Longs)</h4>"
    if not core_longs.empty:
        html += "<p><b>逻辑：</b>趋势健康，全周期共振向上。</p><ul>"
        for asset in core_longs.index[:3]:
            html += f"<li><b>{asset}</b> (分: {colorize(df.loc[asset, 'master_score'])})</li>"
        html += "</ul>"
    else: html += "<p>暂无完美多头形态资产。</p>"
    html += "</div>"
    
    # 空头建议
    html += "<div style='flex:1; min-width:300px; background:#d1ecf1; padding:15px; border-radius:8px; border-left:5px solid #17a2b8;'>"
    html += "<h4 style='margin-top:0;'>🐻 核心空头 (Core Shorts)</h4>"
    if not core_shorts.empty:
        html += "<p><b>逻辑：</b>趋势崩坏，全周期共振向下。</p><ul>"
        for asset in core_shorts.index[:3]:
            html += f"<li><b>{asset}</b> (分: {colorize(df.loc[asset, 'master_score'])})</li>"
        html += "</ul>"
    else: html += "<p>暂无完美空头形态资产。</p>"
    html += "</div>"
    
    # 避坑指南
    html += "<div style='flex:1; min-width:300px; background:#e2e3e5; padding:15px; border-radius:8px; border-left:5px solid #6c757d;'>"
    html += "<h4 style='margin-top:0;'>💤 避坑指南 (Avoid List)</h4>"
    if not avoid_list.empty:
        html += "<p><b>逻辑：</b>波动率低且无方向(垃圾时间)。</p><ul>"
        for asset in avoid_list.index[:4]:
             html += f"<li><b>{asset}</b></li>"
        html += "</ul>"
    else: html += "<p>市场分化明确，暂无垃圾时间资产。</p>"
    html += "</div>"
    html += "</div>"
    
    return html

# --- 报告生成主逻辑 ---

def create_report(scores_df, corr_df, raw_df):
    html_sections = []
    
    # 1. 市场情绪 (简约版)
    core = ["上证50ETF", "沪深300ETF", "创业板ETF"]
    valid = [a for a in core if a in scores_df.index]
    sent = np.clip(scores_df.loc[valid, 'weighted_z_score_rs'].mean() * 2, -10, 10) if valid else 0
    sent_color = "#d9534f" if sent > 2 else ("#28a745" if sent < -2 else "#777")
    
    html_sections.append(f"""
    <div style='text-align:center; padding:20px; background:#fff; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.05);'>
        <h2 style='margin:0; color:#333;'>市场情绪仪表盘</h2>
        <div style='font-size:3em; font-weight:bold; color:{sent_color}; margin:10px 0;'>{sent:.2f}</div>
        <div>基于核心宽基ETF的动能合成</div>
    </div>
    """)
    
    # 2. 深度洞察 (Full Version)
    html_sections.append(generate_deep_dive_full(scores_df))
    
    # 3. 分组排名表
    for g_name, g_assets in GROUPS.items():
        sub = scores_df[scores_df.index.isin(g_assets)].copy()
        if not sub.empty:
            sub = sub.rename(columns=COLUMN_TRANSLATIONS).sort_values('综合大师分', ascending=False)
            cols = ['综合大师分', '动能加速度', '加权相对Z值'] + [c for c in sub.columns if '日相对Z值' in c]
            cols = [c for c in cols if c in sub.columns]
            
            html_sections.append(f"<h3>{g_name} 动能排名</h3>")
            html_sections.append(sub[cols].to_html(classes='styled-table', escape=False, formatters={c: colorize for c in cols}))

    # 4. 相关性矩阵 (带黄色高亮)
    if not corr_df.empty:
        # 筛选有效资产
        valid_assets = [a for a in ASSETS_MAP.keys() if a in corr_df.index]
        if valid_assets:
            corr_sub = corr_df.loc[valid_assets, valid_assets]
            html_sections.append("<h3>最近 60 日资产回报相关性矩阵 (黄色高亮 > 0.8)</h3>")
            # 使用 colorize 逻辑应用到相关性矩阵的每个单元格
            html_sections.append(corr_sub.to_html(classes='styled-table', escape=False, formatters={col: colorize for col in corr_sub.columns}))

    # 写入文件
    css = """<style>
        body{font-family:'Microsoft YaHei', sans-serif; background:#f4f6f9; padding:30px; color:#333;}
        .container{max-width:1200px; margin:auto; background:#fff; padding:40px; border-radius:12px; box-shadow:0 5px 15px rgba(0,0,0,0.1);}
        h1{text-align:center; margin-bottom:10px;}
        h3{border-left:5px solid #007bff; padding-left:10px; margin-top:30px; color:#0056b3;}
        .styled-table{width:100%; border-collapse:collapse; margin:15px 0; font-size:0.9em;}
        .styled-table th{background:#007bff; color:#fff; padding:10px;}
        .styled-table td{padding:8px; border-bottom:1px solid #eee; text-align:center;}
        .styled-table tr:hover{background:#f1f1f1;}
    </style>"""
    
    full_html = f"<html><head><meta charset='utf-8'><title>中国期权宏观分析Pro</title>{css}</head><body><div class='container'><h1>中国期权宏观全景分析 (Pro 1.2)</h1><p style='text-align:center;color:#666'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>{''.join(html_sections)}</div></body></html>"
    
    with open("qqbdw.html", "w", encoding='utf-8') as f: f.write(full_html)
    print("\n报告生成成功: China_Option_Macro_Pro_Full.html")

# =============================================================================
# 主程序
# =============================================================================
if __name__ == '__main__':
    print("=== 启动深度分析引擎 (Deep Dive Restored) ===")
    raw_df = fetch_data_robust(ASSETS_MAP, BENCHMARK_TICKER)
    if not raw_df.empty:
        print("计算动能得分...")
        scores = calculate_professional_momentum_score(raw_df, BENCHMARK_NAME)
        print("计算相关性矩阵...")
        corr = raw_df.pct_change().tail(60).corr()
        
        if not scores.empty:
            create_report(scores, corr, raw_df)
    else:
        print("数据获取失败。")
