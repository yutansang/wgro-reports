# -*- coding: utf-8 -*-
"""
中国A股小市值·博弈全景看板 (Logic Master Pro)
版本: v8.0 (配置驱动版)
升级: 
1. 股票池从代码中移除，改为从外部 `config_*.json` 文件读取。
2. 脚本会自动扫描并处理所有配置文件，为每个股票池生成独立的报告。
"""

import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os
import glob

warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# 1. 全局配置
# =============================================================================
BENCHMARK_TICKER = 'sh.000300' 
TIME_PERIODS = {'long': 60, 'short': 10} 

COLUMN_TRANSLATIONS = {
    'master_score': '👻 控盘鬼才分', 'avg_turnover': '日均换手%', 
    'chip_solidity': '筹码硬度', 'independence': '独立系数', 
    'period_return': '区间涨幅%', 'volatility': '波动率(%)',
    'trend_slope': '趋势斜率', 'acceleration': '情绪加速度'
}

COLUMN_ORDER = [
    'master_score', 'trend_slope', 'acceleration', 'chip_solidity', 
    'avg_turnover', 'volatility', 'period_return', 'independence'
]

# =============================================================================
# 2. 数据获取 & 计算 (函数接受股票池作为参数)
# =============================================================================
def fetch_data_and_calc(portfolio_name, sector_mapping, days=120):
    all_assets = list(sector_mapping.values()) + [BENCHMARK_TICKER]
    code_to_name = {v: k for k, v in sector_mapping.items()}

    print(f"\n🧠 [v8.0] 正在为组合【{portfolio_name}】扫描 {len(all_assets)} 个节点...")
    bs.login()
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data_store = {}
    
    for i, code in enumerate(all_assets):
        print(f"[{i+1}/{len(all_assets)}] 读取数据流: {code}...", end="\r")
        try:
            rs = bs.query_history_k_data_plus(code, "date,close,high,low,volume,turn,pctChg", start, end, "d", "2")
            if rs.error_code == '0':
                dlist = rs.get_data()
                if not dlist.empty:
                    df = pd.DataFrame(dlist, columns=rs.fields)
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['close', 'high', 'low', 'volume', 'turn', 'pctChg']: 
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    df.set_index('date', inplace=True)
                    data_store[code] = df
        except: pass
    bs.logout()
    
    results = []
    if BENCHMARK_TICKER not in data_store: return pd.DataFrame()
    bench = data_store[BENCHMARK_TICKER]['close'].pct_change().fillna(0)
    
    for code, df in data_store.items():
        if code == BENCHMARK_TICKER: continue
        rdf = df.iloc[-TIME_PERIODS['long']:]
        if len(rdf) < 20: continue
        
        avg_turn = rdf['turn'].mean()
        solidity = (rdf['pctChg'].abs().sum() / rdf['turn'].sum() * 10) if rdf['turn'].sum() > 0 else 0
        s_ret = rdf['close'].pct_change().fillna(0)
        b_ret = bench.reindex(s_ret.index).fillna(0)
        indep = 1 - (s_ret.corr(b_ret) if not np.isnan(s_ret.corr(b_ret)) else 0)
        p_ret = (rdf['close'].iloc[-1] / rdf['close'].iloc[0] - 1) * 100
        volatility = rdf['close'].pct_change().std() * 100 
        
        y = rdf['close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0] / y[0] * 100 
        acc = (rdf['turn'].iloc[-5:].mean() / rdf['turn'].mean()) if rdf['turn'].mean() > 0 else 0

        score = indep * 20 + solidity * 15 + slope * 10
        if avg_turn > 20: score -= 20 
        if volatility < 1.8 and abs(p_ret) < 10: score += 10 
        if slope > 0.4 and avg_turn < 12: score += 10 
        
        results.append({
            'Ticker': code_to_name.get(code, code),
            'master_score': score, 'avg_turnover': avg_turn, 
            'chip_solidity': solidity, 'independence': indep, 
            'period_return': p_ret, 'volatility': volatility,
            'trend_slope': slope, 'acceleration': acc
        })
    
    return pd.DataFrame(results).set_index('Ticker').sort_values('master_score', ascending=False)

# =============================================================================
# 3. 推理引擎 (UI渲染) - 无需修改
# =============================================================================
def hl(val, unit="", type="neutral"):
    base_style = "padding:0px 4px; border-radius:3px; font-weight:bold;"
    if type == "good": style = f"background-color:#fff3cd; color:#856404; {base_style}"
    elif type == "risk": style = f"background-color:#f8d7da; color:#721c24; {base_style}"
    elif type == "cool": style = f"background-color:#d1ecf1; color:#0c5460; {base_style}"
    else: style = f"background-color:#ffff00; color:#000; {base_style}"
    return f"<span style='{style}'>{val:.2f}{unit}</span>"

def analyze_logic(row):
    logic = []
    if row['chip_solidity'] > 4.0: logic.append(f"筹码硬度 {hl(row['chip_solidity'], type='good')} (极佳)")
    elif row['chip_solidity'] < 2.0: logic.append(f"筹码松动 ({hl(row['chip_solidity'], type='risk')})")
    
    if row['trend_slope'] > 0.4: logic.append(f"攻击角度犀利 ({hl(row['trend_slope'], type='good')})")
    elif abs(row['trend_slope']) < 0.15: logic.append(f"横盘极致收敛")
    
    if row['avg_turnover'] < 4: logic.append(f"极度缩量 ({hl(row['avg_turnover'],'%', type='cool')})")
    elif row['acceleration'] > 1.3: logic.append(f"资金正在进场 (加速{hl(row['acceleration'])})")
    
    return "，".join(logic) + "。"

def render_tier_card(title, color, df, icon, desc_func):
    if df.empty: return ""
    html = f"<div class='card' style='border-left: 5px solid {color};'><div class='card-header' style='color:{color}; display:flex; justify-content:space-between;'><span>{icon} {title}</span><span style='font-size:0.8em; opacity:0.7'>共挖掘到 {len(df)} 只</span></div><div class='card-body'>"
    ranks = ["🥇 首选", "🥈 次选", "🥉 备选"]
    for i, (name, row) in enumerate(df.head(3).iterrows()):
        rank_str = ranks[i] if i < 3 else f"No.{i+1}"
        bg_col = "#fafafa" if i > 0 else "#fff"
        border_b = "1px dashed #eee" if i < len(df)-1 and i<2 else "none"
        html += f"<div style='padding:12px; background:{bg_col}; border-bottom:{border_b};'><div style='display:flex; align-items:center; margin-bottom:6px;'><span style='font-weight:bold; color:{color}; margin-right:10px;'>{rank_str}</span><span style='font-size:1.1em; font-weight:bold; color:#333;'>{name}</span><span style='margin-left:auto; font-size:0.85em; background:#eee; padding:2px 8px; border-radius:10px;'>评分: {row['master_score']:.1f}</span></div><div style='color:#555; font-size:0.9em; margin-bottom:4px;'>🔍 <b>微观结构:</b> {analyze_logic(row)}</div><div style='color:#333; font-size:0.95em; line-height:1.5; background:rgba(0,0,0,0.02); padding:5px; border-radius:4px;'>🕵️ <b>推演:</b> {desc_func(row)}</div></div>"
    html += "</div></div>"
    return html

def generate_deep_inference_report(df):
    html = "<h2>🧠 逻辑推理引擎 (v8.0)</h2>"
    
    hunters = df[(df['volatility'] < 2.5) & (df['chip_solidity'] > 2.8) & (df['trend_slope'] > -0.2) & (df['trend_slope'] < 0.35)].sort_values('chip_solidity', ascending=False)
    def hunter_logic(row):
        return f"典型且极致的缩量（换手仅{hl(row['avg_turnover'],'%')}），主力像鳄鱼一样潜伏。高硬度说明散户已离场，极易拉升。" if row['avg_turnover'] < 3 else f"在当前横盘震荡中表现出了惊人的稳定性（波动率{hl(row['volatility'],'%')}）。主力在这一位置有极强的护盘意愿，是个安全的防守反击点。"
    html += render_tier_card("潜伏猎手 (低位埋伏)", "#17a2b8", hunters, "💎", hunter_logic)

    movers = df[(df['trend_slope'] > 0.3) & (df['avg_turnover'] < 16) & (df['avg_turnover'] > 4)].sort_values('trend_slope', ascending=False)
    def mover_logic(row):
        return f"完美的**主升浪结构**。斜率向上 ({hl(row['trend_slope'])})，且筹码异常牢固，主力强控盘。" if row['chip_solidity'] > 5 else f"趋势非常强劲，资金合力正在推升股价。虽然筹码稍显松动，但情绪加速度 ({hl(row['acceleration'])}) 显示新资金接力意愿强。"
    html += render_tier_card("趋势龙头 (右侧进攻)", "#d93025", movers, "🚀", mover_logic)
    
    risks = df[(df['avg_turnover'] > 18) | ((df['trend_slope'] < -0.5) & (df['avg_turnover'] > 8))].sort_values('avg_turnover', ascending=False)
    def risk_logic(row): return f"数据出现异常。日均换手率 {hl(row['avg_turnover'],'%', 'risk')} 处于极高水位，这往往是博傻阶段的尾声。"
    if not risks.empty: html += render_tier_card("高危预警 (规避陷阱)", "#ffc107", risks, "⚠️", risk_logic)

    if hunters.empty and movers.empty: html += "<div class='card'><div class='card-body'>🕵️ 扫描完毕：当前市场极度混沌，未发现符合高胜率模型的标的。</div></div>"
    return html

# =============================================================================
# 4. 报表生成 (函数接受组合名称作为参数)
# =============================================================================
def generate_html_table(df):
    df_d = df.copy()
    cols = [c for c in COLUMN_ORDER if c in df_d.columns]
    df_d = df_d[cols].rename(columns=COLUMN_TRANSLATIONS)
    style_hl = "background-color:#ffff00; color:#000; padding:2px 4px; border-radius:3px; font-weight:bold;"
    def c_common(v, th): return f"<span style='{style_hl if v>th else ''}'>{v:.2f}</span>"
    def c_trend(v): return f"<span style='{style_hl if v>0.4 else ('color:green' if v<-0.2 else '')}'>{v:.2f}</span>"
    formatters = {'👻 控盘鬼才分':lambda x:c_common(x,80),'日均换手%':lambda x:c_common(x,15),'筹码硬度':lambda x:c_common(x,3.5),'趋势斜率':c_trend,'波动率(%)':lambda x:f"{x:.2f}",'情绪加速度':lambda x:c_common(x,1.3),'独立系数':lambda x:f'{x:.2f}','区间涨幅%':lambda x:f'{x:.2f}'}
    return f"<h2>📊 全景博弈数据</h2>{df_d.to_html(classes='styled-table', escape=False, border=0, justify='center', formatters=formatters)}"

def create_report(df, portfolio_name):
    css = """<style>body{font-family:'Segoe UI', 'Microsoft YaHei', sans-serif; padding:20px; background:#f0f2f5; color:#2c3e50;} .container{max-width:1100px; margin:auto; background:#fff; padding:40px; border-radius:12px; box-shadow:0 5px 20px rgba(0,0,0,0.05);} h1{color:#2c3e50; text-align:center; border-bottom:3px solid #2c3e50; padding-bottom:15px; margin-bottom:5px;} h2{margin-top:40px; border-left:5px solid #2c3e50; padding-left:15px; font-size:1.5em; color:#34495e;} .styled-table{width:100%; border-collapse:collapse; margin:20px 0; font-size:0.9em;} .styled-table th{background-color:#2c3e50; color:#ffffff; padding:10px;} .styled-table td{padding:10px; border-bottom:1px solid #ddd; text-align:center;} .card{background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); margin-bottom:25px; overflow:hidden; border:1px solid #eee;} .card-header{padding:12px 20px; font-weight:bold; background:#fafafa; border-bottom:1px solid #eee;} .card-body{padding:10px 20px;}</style>"""
    
    title = f"🇨🇳 A股 · 深度推理报告 ({portfolio_name})"
    report_filename = f"{portfolio_name}.html"
    
    sections = [generate_deep_inference_report(df), generate_html_table(df)]
    
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title>{css}</head><body><div class='container'><h1>{title}</h1><p style='text-align:center;color:#7f8c8d'>Powered by Logic Master v8.0</p>{''.join(sections)}</div></body></html>"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 报告已生成: {report_filename}")

# =============================================================================
# 5. 主程序入口 (循环处理所有配置文件)
# =============================================================================
if __name__ == '__main__':
    # 查找当前目录下所有 'config_*.json' 文件
    config_files = glob.glob('config_*.json')
    
    if not config_files:
        print("❌ 未找到任何配置文件 (例如 'config_tech.json')。请在脚本同目录下创建。")
    else:
        print(f"🔍 发现 {len(config_files)} 个配置文件，即将开始处理...")
        
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            portfolio_name = config_data.get("portfolio_name", "未命名组合")
            sector_mapping = config_data.get("stocks", {})
            
            if not sector_mapping:
                print(f"⚠️ 配置文件 {config_file} 为空或格式错误，已跳过。")
                continue

            df = fetch_data_and_calc(portfolio_name, sector_mapping)
            
            if not df.empty:
                create_report(df, portfolio_name)
            else:
                print(f"❌ 组合【{portfolio_name}】数据不足，无法生成报告。")
                
        except Exception as e:
            print(f"❌ 处理配置文件 {config_file} 时发生错误: {e}")

    print("\n🎉 所有任务已完成。")
