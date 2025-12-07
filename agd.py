# -*- coding: utf-8 -*-
"""
中国A股全景交易决策看板 (BaoStock Pro Max版 - 完全体)
版本: 5.3 (支持多配置文件)
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 全局配置 (固定部分)
# =============================================================================

BENCHMARK_TICKER = 'sh.000300' 
TIME_PERIODS = {'long_term': 60, 'mid_term': 20, 'short_term': 5}
PERIOD_WEIGHTS = {'long_term': 0.6, 'mid_term': 0.3, 'short_term': 0.1}

# 宏观指数和宽基，作为固定分析对象，始终会包含在内
MACRO_INDICATORS = {
    "上证指数": "sh.000001",
    "上证50 (超大盘)": "sh.000016",
    "沪深300 (大盘)": "sh.000300",
    "创业板指 (成长)": "sz.399006",
    "中证500 (中盘)": "sh.000905",
    "中证1000 (小盘)": "sh.000852", 
    "科创50 (硬科技)": "sh.000688"
}

# 报告列名翻译与顺序
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
# 2. BaoStock 数据获取
# =============================================================================
def fetch_data_baostock(tickers, years=2):
    print(f"正在连接 BaoStock 系统，下载 {len(tickers)} 个资产数据...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return pd.DataFrame()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    all_series = {}
    
    total = len(tickers)
    for i, code in enumerate(tickers):
        print(f"[{i+1}/{total}] 下载: {code}", end="\r")
        try:
            rs = bs.query_history_k_data_plus(
                code, "date,close", start_date=start_date, end_date=end_date, frequency="d", adjustflag="2"
            )
            if rs.error_code != '0': continue
            data_list = []
            while (rs.error_code == '0') & rs.next(): data_list.append(rs.get_row_data())
            if data_list:
                df_temp = pd.DataFrame(data_list, columns=rs.fields)
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                df_temp['close'] = df_temp['close'].astype(float)
                df_temp.set_index('date', inplace=True)
                all_series[code] = df_temp['close']
        except Exception as e: print(f"\n下载 {code} 出错: {e}")

    bs.logout()
    print("\n数据下载完成，正在合并清洗...")
    if not all_series: return pd.DataFrame()
    combined_df = pd.DataFrame(all_series)
    combined_df.ffill(inplace=True); combined_df.bfill(inplace=True)
    return combined_df

# =============================================================================
# 3. 计算逻辑 (已修改)
# =============================================================================
def calculate_professional_momentum_score(price_data, benchmark_price, ticker_mapping):
    results = []
    ticker_to_name = {v: k for k, v in ticker_mapping.items()}
    
    for ticker in price_data.columns:
        if ticker == benchmark_price.name: continue
        asset_price = price_data[ticker]
        aligned_benchmark = benchmark_price.reindex(asset_price.index).ffill()
        is_index = ticker in MACRO_INDICATORS.values()
        relative_price = asset_price if is_index else (asset_price / aligned_benchmark).dropna()

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
    df.index = [ticker_to_name.get(t, t) for t in df.index]
    return df

# =============================================================================
# 4. 报告生成模块 (未变动)
# =============================================================================

def generate_market_sentiment_module(all_scores_df):
    html = "<h2>🐉 A股情绪全景 (Market Sentiment)</h2>"
    def get_z(name):
        # 此函数依赖的股票名称相对固定，如果分析池中包含它们，就会被正确计算
        if name in all_scores_df.index:
            return all_scores_df.loc[name, 'weighted_z_score_rs']
        return 0

    tech_growth = (get_z("创业板指 (成长)") + get_z("科创50 (硬科技)")) / 2
    blue_chip = get_z("上证50 (超大盘)")
    speculation = get_z("中证1000 (小盘)")
    broker = max(get_z("中信证券 (券商)"), get_z("东方财富 (互金)"))

    score = (tech_growth * 0.4) + (broker * 0.3) + (speculation * 0.2) + (blue_chip * 0.1)
    score = np.clip(score * 1.5, -10, 10)
    
    if score > 6: s, c = "极度火热 (FOMO)", "#d93025"
    elif score > 2: s, c = "乐观 (Bullish)", "#ffc107"
    elif score > -2: s, c = "震荡 (Neutral)", "#6c757d"
    elif score > -6: s, c = "低迷 (Bearish)", "#28a745"
    else: s, c = "冰点 (Freezing)", "#17a2b8"
    
    html += f"""
    <div style='text-align:center; margin:20px 0; padding:20px; background:#fff; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>
        <div style='font-size:1.5em;'>市场热度: <strong style='color:{c};'>{s}</strong></div>
        <div style='font-size:3.5em; font-weight:bold; margin:15px 0; color:{c}'>{score:.2f}</div>
        <div style='width:80%; margin:auto; background-color:#e9ecef; border-radius:10px; height:25px; position:relative;'>
            <div style='height:100%; width:2px; background-color:#343a40; position:absolute; left:50%;'></div>
            <div style='height:25px; width:25px; background-color:{c}; border:3px solid #fff; border-radius:50%; position:absolute; top:0; left:calc({(score+10)*5}% - 12.5px);'></div>
        </div>
        <p style='margin-top:15px; font-size:0.9em; color:#666;'>成长({tech_growth:.2f}) | 题材({speculation:.2f}) | 旗手({broker:.2f})</p>
    </div>"""
    return html

def generate_deep_exploration_module(all_scores_df):
    html = "<h2>🧠 深度洞察 (AI Narrative)</h2>"
    html += "<h3 style='margin-top:30px; border-bottom: 2px solid #eee; padding-bottom:10px;'>数据深度解读：正反逻辑链</h3>"
    html += "<div style='background-color:#f8f9fa; padding:20px; border-radius:8px; border-left: 5px solid #0056b3;'>"
    
    stocks_df = all_scores_df[~all_scores_df.index.isin(MACRO_INDICATORS.keys())]
    if stocks_df.empty:
        html += "<p>当前股票池为空或数据不足，无法生成深度洞察。</p></div>"
        return html

    def find_stocks(condition):
        return stocks_df[condition].sort_values('acceleration', ascending=False)

    # A. 真·主升浪
    true_bulls = find_stocks((stocks_df['master_score'] > 2) & (stocks_df['acceleration'] > 0.3))
    if not true_bulls.empty:
        top = true_bulls.iloc[0]
        html += f"<div style='margin-bottom:20px;'><h4 style='color:#d93025; margin:0;'>✅ 真·主升浪 (买入/持有)</h4><p><b>标的案例：{top.name}</b></p><ul><li><b>【数据真相】</b>: Alpha高达 <b>{top['master_score']:.2f}</b> (全场领先)，且加速度 <b>+{top['acceleration']:.2f}</b> (还在加速)。</li><li><b>【逻辑判断】</b>: 这是完美的<b>'戴维斯双击'</b>形态。既有长期趋势支撑，短期又在加速上攻。它是当前市场的<b>绝对龙头</b>。</li><li><b>【操作对策】</b>: <b style='color:#d93025'>抱紧大腿</b>。只要不出现加速跌破5日线，就一直持有。</li></ul></div>"

    # B. 高位预警
    danger_high = find_stocks((stocks_df['master_score'] > 2) & (stocks_df['acceleration'] < -0.5))
    if not danger_high.empty:
        top = danger_high.sort_values('acceleration', ascending=True).iloc[0]
        html += f"<div style='margin-bottom:20px;'><h4 style='color:#ffc107; margin:0;'>⚠️ 高位预警 (减仓/止盈)</h4><p><b>标的案例：{top.name}</b></p><ul><li><b>【数据真相】</b>: 长期Alpha依然很高 <b>{top['master_score']:.2f}</b>，但加速度已崩塌至 <b style='color:#28a745'>{top['acceleration']:.2f}</b>。</li><li><b>【逻辑判断】</b>: 这是典型的<b>'强弩之末'</b>。上涨动能衰竭，资金正在撤退，<b>获利了结</b>信号明显。</li><li><b>【操作对策】</b>: <b style='color:#ffc107'>坚决止盈</b>。不要迷恋过去的辉煌，不要去吃最后的一个铜板。</li></ul></div>"

    # C. 超跌反弹
    rebound = find_stocks((stocks_df['master_score'] < -0.5) & (stocks_df['acceleration'] > 0.5))
    if not rebound.empty:
        top = rebound.iloc[0]
        html += f"<div style='margin-bottom:20px;'><h4 style='color:#17a2b8; margin:0;'>⚡ 超跌反弹 (博弈/短线)</h4><p><b>标的案例：{top.name}</b></p><ul><li><b>【数据真相】</b>: 长期Alpha还在水下 <b>{top['master_score']:.2f}</b>，但加速度异军突起 <b style='color:#d93025'>+{top['acceleration']:.2f}</b>。</li><li><b>【逻辑判断】</b>: <b>'困境反转'</b>的首选。跌得太久了，主力资金开始猛烈回补。</li><li><b>【操作对策】</b>: <b style='color:#17a2b8'>右侧试错</b>。适合短线快进快出，一旦加速度转弱立即离场。</li></ul></div>"

    # D. 深不见底
    avoids = find_stocks((stocks_df['master_score'] < -1) & (stocks_df['acceleration'] < -0.2))
    if not avoids.empty:
        top = avoids.sort_values('acceleration', ascending=True).iloc[0]
        html += f"<div><h4 style='color:#28a745; margin:0;'>❌ 深不见底 (回避)</h4><p><b>标的案例：{top.name}</b></p><ul><li><b>【数据真相】</b>: Alpha深绿 <b>{top['master_score']:.2f}</b>，且加速度还在负值区间 <b style='color:#28a745'>{top['acceleration']:.2f}</b>。</li><li><b>【逻辑判断】</b>: <b>'阴跌不止'</b>。不要轻易抄底，飞刀还没落地。</li><li><b>【操作对策】</b>: <b style='color:#28a745'>坚决远离</b>。这类股票是账户亏损的主要来源。</li></ul></div>"
    
    html += "</div>"
    return html

def generate_sector_radar(all_scores_df):
    html = "<h2>📊 板块动能雷达</h2>"
    # 此模块依赖固定的分组，如果新配置的股票不在此列，该分组将不会显示
    groups = {
        "核心宽基": ["沪深300 (大盘)", "创业板指 (成长)", "科创50 (硬科技)", "中证1000 (小盘)"],
        "科技主线": ["中芯国际 (半导体)", "工业富联 (AI算力)", "中际旭创 (CPO)", "立讯精密 (果链)"],
        "赛道反弹": ["隆基绿能 (光伏)", "阳光电源 (储能)", "宁德时代 (锂电)", "比亚迪 (新能源)"],
        "红利/金融": ["长江电力 (水电)", "中国神华 (煤炭)", "中信证券 (券商)", "中国平安 (保险)"],
        "大消费": ["贵州茅台 (白酒)", "美的集团 (家电)", "中国中免 (免税)", "迈瑞医疗 (器械)"]
    }
    pivot_html = "<table class='pivot-table'><thead><tr><th>板块分组</th><th>长期趋势(60d)</th><th>短期趋势(5d)</th><th>动能加速度</th><th>状态</th></tr></thead><tbody>"
    for g_name, assets in groups.items():
        valid = [a for a in assets if a in all_scores_df.index]
        if not valid: continue
        sub_df = all_scores_df.loc[valid]
        lt = sub_df[f'z_score_rs_{TIME_PERIODS["long_term"]}d'].mean()
        st = sub_df[f'z_score_rs_{TIME_PERIODS["short_term"]}d'].mean()
        acc = sub_df['acceleration'].mean()
        
        c_acc = "#d93025" if acc > 0 else "#28a745"
        status = "盘整"
        if lt > 0 and acc > 0.2: status = "📈 进攻"
        elif lt < 0 and acc > 0.3: status = "⚡ 反弹"
        elif lt > 0 and acc < -0.2: status = "📉 调整"
        elif lt < 0 and acc < 0: status = "❄️ 阴跌"
        
        pivot_html += f"<tr><td style='text-align:left;font-weight:bold;'>{g_name}</td><td>{lt:.2f}</td><td>{st:.2f}</td><td><span style='color:{c_acc}'>{acc:.2f}</span></td><td><b>{status}</b></td></tr>"
    html += pivot_html + "</tbody></table>"
    return html

def colorize(val):
    if isinstance(val, (int, float)):
        color = '#d93025' if val > 0 else '#28a745'
        if abs(val) > 1.0:
            bg = '#f8d7da' if val > 0 else '#d4edda'
            return f'<span style="background-color: {bg}; color: {color}; font-weight: bold;">{val:.2f}</span>'
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

def create_html_report(all_html_sections, filename="A股全景分析报告_完全体.html"):
    css = """<style>
        body{font-family:"Microsoft YaHei",sans-serif;padding:2rem;background:#f4f6f9;color:#333}
        h1{text-align:center;color:#d93025;border-bottom:3px solid #d93025;padding-bottom:10px} 
        h2{color:#333;border-left:5px solid #d93025;padding-left:10px;margin-top:30px;background:#fff;padding:15px;border-radius:5px;}
        h3{color:#d93025;margin-top:25px} h4{font-size:1.1em; font-weight:bold;}
        .container{max-width:1300px;margin:auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 6px 15px rgba(0,0,0,.08)}
        .styled-table, .pivot-table{width:100%;border-collapse:collapse;margin:20px 0;}
        .styled-table th, .pivot-table th{background:#d93025;color:#fff;padding:12px;text-align:center}
        .styled-table td, .pivot-table td{padding:10px;border-bottom:1px solid #eee;text-align:center}
        .styled-table tr:hover, .pivot-table tr:hover{background-color:#f1f1f1}
        li{margin-bottom:8px; line-height:1.6;} b{font-weight:700;color:#000}
    </style>"""
    html_t = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>A股深度报告(完全体)</title>{css}</head><body><div class='container'><h1>🇨🇳 A股全景交易决策看板 (v5.3 动态配置版)</h1><p style='text-align:center;color:#888'>数据源: BaoStock | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>{''.join(all_html_sections)}</div></body></html>"
    with open(filename, 'w', encoding='utf-8') as f: f.write(html_t)
    print(f"✅ 报告已生成: {filename}")


# =============================================================================
# 5. 主流程 (新)
# =============================================================================
def process_config(config_file, sector_mapping, output_filename):
    """
    为单个配置文件执行完整的分析和报告生成流程
    """
    print(f"\n===== 正在处理配置: {config_file} =====")
    
    # 1. 准备资产列表
    ticker_mapping = {**MACRO_INDICATORS, **sector_mapping}
    all_analysis_assets = list(set(list(MACRO_INDICATORS.values()) + list(sector_mapping.values())))
    all_tickers = list(set(all_analysis_assets + [BENCHMARK_TICKER]))
    
    # 2. 获取数据
    price_data = fetch_data_baostock(all_tickers)
    
    if price_data.empty or BENCHMARK_TICKER not in price_data.columns:
        print(f"❌ {config_file} 的数据不足。请检查网络或 BaoStock 是否在维护时间。")
        return

    # 3. 计算指标
    benchmark_data = price_data[BENCHMARK_TICKER]
    print("⚡ 正在计算 Alpha 与 动量因子...")
    full_analysis_df = calculate_professional_momentum_score(price_data, benchmark_data, ticker_mapping)
    
    if full_analysis_df.empty:
        print(f"❌ {config_file} 计算得分失败，无法生成报告。")
        return

    st_col = f'z_score_rs_{TIME_PERIODS["short_term"]}d'
    mt_col = f'z_score_rs_{TIME_PERIODS["mid_term"]}d'
    if st_col in full_analysis_df.columns and mt_col in full_analysis_df.columns:
        full_analysis_df['acceleration'] = full_analysis_df[st_col] - full_analysis_df[mt_col]
    else:
        full_analysis_df['acceleration'] = 0
    
    # 4. 生成HTML模块
    html_sections = []
    html_sections.append(generate_market_sentiment_module(full_analysis_df))
    html_sections.append(generate_sector_radar(full_analysis_df))
    html_sections.append(generate_deep_exploration_module(full_analysis_df))
    
    # 5. 生成HTML表格
    categories = [
        (f"🏆 核心个股排名 (vs 沪深300) - {os.path.basename(config_file)}", sector_mapping.values()),
        ("🌍 宽基指数趋势", MACRO_INDICATORS.values())
    ]
    reverse_map = {v: k for k, v in ticker_mapping.items()}

    for title, tickers in categories:
        target_names = [reverse_map.get(t) for t in tickers if reverse_map.get(t) in full_analysis_df.index]
        if target_names:
            subset = full_analysis_df.loc[target_names].sort_values('master_score', ascending=False)
            html_sections.append(generate_html_table(subset, title))

    # 6. 创建最终报告
    create_html_report(html_sections, filename=output_filename)

def main():
    print("🚀 启动 A股全景引擎 (v5.3 - 动态配置版)...")
    
    # ▼▼▼ 第 1 处修改 ▼▼▼
    # 将 startswith('config_') 修改为 startswith('yuconfig_')
    config_files = sorted([f for f in os.listdir('.') if f.startswith('yuconfig_') and f.endswith('.json')])

    if not config_files:
        # 更新提示信息，告诉用户新的命名规则
        print("❌ 未找到任何 `yuconfig_*.json` 配置文件。请在脚本目录创建它们。")
        print("   例如，创建一个名为 'yuconfig_我的自选.json' 的文件，内容格式如下:")
        print("""
        {
          "宁德时代 (锂电)": "sz.300750",
          "比亚迪 (新能源)": "sz.002594"
        }
        """)
        return

    # 遍历所有找到的配置文件
    for config_filename in config_files:
        try:
            with open(config_filename, 'r', encoding='utf-8') as f:
                sector_mapping_data = json.load(f)
            
            # ▼▼▼ 第 2 处修改 ▼▼▼
            # 1. 将 replace('config_', ...) 修改为 replace('yuconfig_', ...)
            report_base_name = config_filename.replace('yuconfig_', '').replace('.json', '')
            
            # 2. 生成HTML文件名 (这行不用变)
            output_report_name = f"{report_base_name}.html"

            # 调用核心处理函数
            process_config(config_filename, sector_mapping_data, output_report_name)

        except json.JSONDecodeError: 
            print(f"❌ 错误: 配置文件 {config_filename} 不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"❌ 处理 {config_filename} 时发生未知错误: {e}")

if __name__ == '__main__':
    main()
