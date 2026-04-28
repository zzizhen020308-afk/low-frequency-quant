#!/usr/bin/env python3
"""
行业数据获取与缓存脚本
从 yfinance 抓取所有股票的所属行业（Sector）并本地缓存
"""
import pandas as pd
import json
import time
import yfinance as yf
from tqdm import tqdm


def fetch_sector_mapping(tickers, cache_file='sector_mapping.json'):
    """
    获取股票代码到行业的映射关系，并本地缓存

    参数:
        tickers: list 股票代码列表
        cache_file: str 缓存文件路径

    返回:
        dict: {ticker: sector_name}
    """
    # 先尝试读取已有缓存
    try:
        with open(cache_file, 'r') as f:
            cached_mapping = json.load(f)
        print(f"✅ 读取到 {len(cached_mapping)} 只股票的行业缓存数据")
    except (FileNotFoundError, json.JSONDecodeError):
        cached_mapping = {}
        print("ℹ️  未找到行业缓存，将从头抓取")

    # 找出还没抓取的股票
    to_fetch = [t for t in tickers if t not in cached_mapping]

    if len(to_fetch) == 0:
        print("✅ 所有股票行业数据已缓存，无需重新抓取")
        return cached_mapping

    print(f"🔍 还需抓取 {len(to_fetch)} 只股票的行业数据...")

    # 批量抓取，加入失败重试机制
    failed_tickers = []
    new_fetches = {}

    for ticker in tqdm(to_fetch, desc="抓取行业数据"):
        try:
            ticker_obj = yf.Ticker(ticker)
            sector = ticker_obj.info.get('sector', 'Unknown')
            new_fetches[ticker] = sector
            time.sleep(0.05)  # 限流，避免被API封禁
        except Exception as e:
            failed_tickers.append(ticker)
            continue

    if len(failed_tickers) > 0:
        print(f"⚠️  有 {len(failed_tickers)} 只股票抓取失败，归入 Unknown")
        for ticker in failed_tickers:
            new_fetches[ticker] = 'Unknown'

    # 合并新旧数据
    final_mapping = {**cached_mapping, **new_fetches}

    # 保存到本地缓存
    with open(cache_file, 'w') as f:
        json.dump(final_mapping, f, indent=2)

    print(f"💾 行业映射已保存到 {cache_file}，共计 {len(final_mapping)} 只股票")

    # 打印行业分布统计
    sector_counts = {}
    for sector in final_mapping.values():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    print("\n📊 行业分布统计:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {count} 只")

    return final_mapping


if __name__ == "__main__":
    # 从价格数据中读取所有股票代码
    prices = pd.read_parquet('sp500_adjusted_close.parquet')
    all_tickers = prices.columns.tolist()
    print(f"股票池共计 {len(all_tickers)} 只股票\n")
    fetch_sector_mapping(all_tickers)
