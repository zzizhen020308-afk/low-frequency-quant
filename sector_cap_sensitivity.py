#!/usr/bin/env python3
"""
行业持仓上限（Sector Cap）敏感性测试
测试不同 sector_cap 阈值对策略绩效的影响
"""
import pandas as pd
import numpy as np
from backtest_engine import (
    load_data,
    generate_monthly_rebalance_dates,
    generate_weights,
    run_backtest,
    calculate_performance_metrics
)

def run_sector_cap_test(sector_cap_values):
    """运行不同 sector_cap 阈值的回测"""
    print("=" * 80)
    print("行业持仓上限敏感性测试")
    print("=" * 80)

    # 加载数据
    prices = load_data()
    rebalance_dates = generate_monthly_rebalance_dates(prices)

    results = []

    for sector_cap in sector_cap_values:
        print(f"\n{'=' * 60}")
        print(f"测试 sector_cap = {sector_cap:.0%} (MAX_STOCKS_PER_SECTOR = {int(20 * sector_cap)})")
        print(f"{'=' * 60}")

        # 生成权重
        weights, prices_with_spy = generate_weights(
            prices,
            rebalance_dates,
            momentum_top_n=50,
            final_select_n=20,
            buffer_keep_n=40,
            sector_cap=sector_cap,
            enable_sector_control=True,
            enable_beta_hedging=False
        )

        # 运行回测
        returns = run_backtest(prices_with_spy, weights, transaction_cost=0.001)
        metrics = calculate_performance_metrics(returns)

        # 计算换手率
        daily_weights = weights.ffill()
        weight_changes = daily_weights.diff().abs().sum(axis=1)
        rebalance_turnover = weight_changes[weight_changes > 1e-6]

        results.append({
            'sector_cap': sector_cap,
            'max_stocks_per_sector': int(20 * sector_cap),
            'annual_return': metrics['annual_return'],
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'monthly_win_rate': metrics['monthly_win_rate'],
            'avg_turnover_per_trade': rebalance_turnover.mean(),
            'annual_turnover': rebalance_turnover.mean() * 12
        })

    return results

def print_sensitivity_summary(results):
    """打印敏感性测试汇总"""
    print("\n" + "=" * 80)
    print("行业持仓上限敏感性测试汇总")
    print("=" * 80)

    df = pd.DataFrame(results)
    df = df.sort_values('sector_cap')

    # 格式化输出
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("\n绩效对比表:")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"\nsector_cap = {row['sector_cap']:.0%} (单行业最多 {row['max_stocks_per_sector']} 只):")
        print(f"  年化收益: {row['annual_return']:>8.2%}  |  年化波动: {row['annual_volatility']:>8.2%}")
        print(f"  夏普比率: {row['sharpe_ratio']:>8.2f}  |  卡玛比率: {row['calmar_ratio']:>8.2f}")
        print(f"  最大回撤: {row['max_drawdown']:>8.2%}  |  月胜率:   {row['monthly_win_rate']:>8.2%}")
        print(f"  年化换手: {row['annual_turnover']:>8.2%}")

    # 找到最优值
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    best_calmar = df.loc[df['calmar_ratio'].idxmax()]

    print("\n" + "=" * 80)
    print("最优参数:")
    print("-" * 80)
    print(f"夏普比率最优: sector_cap = {best_sharpe['sector_cap']:.0%}")
    print(f"  年化收益: {best_sharpe['annual_return']:.2%}, 夏普: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"卡玛比率最优: sector_cap = {best_calmar['sector_cap']:.0%}")
    print(f"  年化收益: {best_calmar['annual_return']:.2%}, 卡玛: {best_calmar['calmar_ratio']:.2f}")

    # 保存到 CSV
    df.to_csv('sector_cap_sensitivity_results.csv', index=False)
    print(f"\n详细结果已保存至: sector_cap_sensitivity_results.csv")

    return df

if __name__ == "__main__":
    # 测试的 sector_cap 阈值范围
    sector_cap_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    results = run_sector_cap_test(sector_cap_values)
    df = print_sensitivity_summary(results)
