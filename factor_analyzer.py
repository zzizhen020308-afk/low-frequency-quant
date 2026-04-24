#!/usr/bin/env python3
import pandas as pd
import numpy as np


def load_data(filepath='sp500_adjusted_close.parquet'):
    """读取价格数据"""
    return pd.read_parquet(filepath)


def calculate_momentum(prices, lookback_months=12, skip_months=1):
    """计算动量因子

    参数:
        prices: DataFrame, 价格数据
        lookback_months: int, 回溯期月数
        skip_months: int, 跳过最近月数

    返回:
        Series: 每只股票的动量值
    """
    # 计算交易日数量 (假设每月21个交易日)
    lookback_days = lookback_months * 21
    skip_days = skip_months * 21

    # 动量 = t - skip_days 天前的价格 / t - skip_days - lookback_days 天前的价格
    momentum = prices.shift(skip_days).iloc[-1] / prices.shift(skip_days + lookback_days).iloc[-1] - 1

    return momentum


def calculate_volatility(prices, window_days=20, annualize=True):
    """计算波动率因子

    参数:
        prices: DataFrame, 价格数据
        window_days: int, 滚动窗口天数
        annualize: bool, 是否年化

    返回:
        Series: 每只股票的波动率
    """
    # 计算日收益率
    returns = prices.pct_change(fill_method=None).dropna(how='all')

    # 计算滚动波动率
    rolling_vol = returns.rolling(window=window_days).std()

    # 年化 (乘以 sqrt(252))
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)

    # 取最新一期波动率
    latest_vol = rolling_vol.iloc[-1]

    return latest_vol


def generate_cross_sectional_report(prices, top_n=50):
    """生成截面报告

    参数:
        prices: DataFrame, 价格数据
        top_n: int, 选出动量最高的股票数量

    返回:
        DataFrame: 按波动率排序的候选股票
    """
    # 计算因子
    momentum = calculate_momentum(prices)
    volatility = calculate_volatility(prices)

    # 合并因子
    factors = pd.DataFrame({
        'momentum_12m_skip_1m': momentum,
        'volatility_20d_annual': volatility
    })

    # 移除缺失值
    factors = factors.dropna()

    # 选出动量最高的前 N 只股票
    top_momentum = factors.nlargest(top_n, 'momentum_12m_skip_1m')

    # 按波动率从低到高排序
    result = top_momentum.sort_values('volatility_20d_annual', ascending=True)

    # 添加排名列
    result['momentum_rank'] = range(1, len(result) + 1)

    # 重命名索引列为 ticker
    result.index.name = 'ticker'

    return result


def main():
    print("=" * 60)
    print("股票因子分析")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载价格数据...")
    prices = load_data()
    print(f"  数据形状: {prices.shape[0]} 天 x {prices.shape[1]} 只股票")
    print(f"  日期范围: {prices.index.date[0]} 至 {prices.index.date[-1]}")

    # 计算动量
    print("\n[2/4] 计算12个月动量(剔除最近1个月)...")
    momentum = calculate_momentum(prices)
    print(f"  动量范围: {momentum.min():.2%} ~ {momentum.max():.2%}")
    print(f"  动量中位数: {momentum.median():.2%}")

    # 计算波动率
    print("\n[3/4] 计算20个交易日年化波动率...")
    volatility = calculate_volatility(prices)
    print(f"  波动率范围: {volatility.min():.2%} ~ {volatility.max():.2%}")
    print(f"  波动率中位数: {volatility.median():.2%}")

    # 生成截面报告
    print("\n[4/4] 生成截面报告...")
    report = generate_cross_sectional_report(prices, top_n=50)

    # 保存结果
    output_file = 'top_momentum_candidates.csv'
    report.to_csv(output_file)
    print(f"  结果已保存至: {output_file}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("截面报告摘要")
    print("=" * 60)
    print(f"\n动量最高的前50只股票（按波动率从低到高排序）:")
    print("-" * 60)
    print(f"{'排名':<5} {'代码':<10} {'12个月动量':>15} {'20日年化波动率':>18}")
    print("-" * 60)

    for i, (ticker, row) in enumerate(report.head(20).iterrows(), 1):
        print(f"{i:<5} {ticker:<10} {row['momentum_12m_skip_1m']:>14.2%} {row['volatility_20d_annual']:>17.2%}")

    if len(report) > 20:
        print(f"\n  ... 还有 {len(report) - 20} 只股票，请查看完整 CSV 文件")

    print("\n" + "=" * 60)
    print("统计摘要:")
    print(f"  平均动量: {report['momentum_12m_skip_1m'].mean():.2%}")
    print(f"  平均波动率: {report['volatility_20d_annual'].mean():.2%}")
    print(f"  波动率最低的3只股票: {', '.join(report.head(3).index.tolist())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
