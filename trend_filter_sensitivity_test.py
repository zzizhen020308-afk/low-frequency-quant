#!/usr/bin/env python3
"""
大盘趋势过滤 - 参数敏感性测试
系统扫描不同过滤方案和参数组合，找出最优配置
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

from backtest_engine import (
    load_data,
    calculate_momentum,
    calculate_volatility,
    generate_monthly_rebalance_dates,
    calculate_performance_metrics,
    run_backtest
)


def generate_weights_with_trend_filter(
        prices,
        rebalance_dates,
        filter_type='sma_cross',
        filter_params=None,
        momentum_top_n=50,
        final_select_n=20,
        buffer_keep_n=40):
    """
    支持多种大盘过滤方案的权重生成函数

    filter_type:
        - 'none': 无过滤（基准）
        - 'sma_cross': 快慢均线交叉
        - 'absolute_momentum': 绝对动量
        - 'dual_momentum': 双动量（1m + 6m）
        - 'volatility_threshold': 波动率阈值过滤
    """
    if filter_params is None:
        filter_params = {}

    momentum = calculate_momentum(prices)
    volatility = calculate_volatility(prices)

    # 大盘代理
    market_index = prices.mean(axis=1)
    bear_market_count = 0

    # 预计算趋势指标（避免在循环中重复计算）
    trend_indicator = None

    if filter_type == 'none':
        # 无过滤：永远是牛市
        pass

    elif filter_type == 'sma_cross':
        # 快慢均线交叉
        short_days = filter_params.get('short_days', 20)
        long_days = filter_params.get('long_days', 100)
        sma_short = market_index.rolling(short_days).mean()
        sma_long = market_index.rolling(long_days).mean()
        trend_indicator = sma_short > sma_long

    elif filter_type == 'absolute_momentum':
        # 绝对动量
        lookback_days = filter_params.get('lookback_days', 126)  # 默认6个月
        market_return = market_index / market_index.shift(lookback_days) - 1
        trend_indicator = market_return > 0

    elif filter_type == 'dual_momentum':
        # 双动量过滤
        short_days = filter_params.get('short_days', 21)   # 1个月
        long_days = filter_params.get('long_days', 126)     # 6个月
        mom_short = market_index / market_index.shift(short_days) - 1
        mom_long = market_index / market_index.shift(long_days) - 1
        # 两个都为正才认为是牛市
        trend_indicator = (mom_short > 0) & (mom_long > 0)

    elif filter_type == 'volatility_threshold':
        # 波动率阈值过滤：高波动时才检查趋势
        vol_window = filter_params.get('vol_window', 60)
        vol_quantile = filter_params.get('vol_quantile', 0.75)
        mom_days = filter_params.get('mom_days', 126)

        market_returns = market_index.pct_change(fill_method=None)
        market_vol = market_returns.rolling(vol_window).std() * np.sqrt(252)
        vol_threshold = market_vol.quantile(vol_quantile)
        mom = market_index / market_index.shift(mom_days) - 1

        # 低波动：永远持仓；高波动：检查动量
        trend_indicator = (market_vol <= vol_threshold) | (mom > 0)

    # 权重矩阵初始化
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    first_rebalance_date = rebalance_dates[0]
    weights.loc[:first_rebalance_date - pd.Timedelta(days=1), :] = 0.0

    previous_holdings = set()

    for rebalance_date in rebalance_dates:
        # ========== 大盘风控检查 ==========
        if filter_type == 'none':
            is_bull_market = True
        else:
            is_bull_market = trend_indicator.loc[rebalance_date]

        if not is_bull_market:
            # 熊市：全部空仓
            weights.loc[rebalance_date, :] = 0.0
            previous_holdings = set()
            bear_market_count += 1
            continue

        # ========== 选股逻辑 ==========
        date_momentum = momentum.loc[rebalance_date].dropna()
        date_volatility = volatility.loc[rebalance_date].dropna()
        common_tickers = date_momentum.index.intersection(date_volatility.index)

        if len(common_tickers) == 0:
            previous_holdings = set()
            continue

        # 动量前N
        top_momentum = date_momentum[common_tickers].nlargest(momentum_top_n).index

        # 波动率排序
        candidate_volatility = date_volatility[top_momentum].sort_values(ascending=True)
        candidate_tickers = candidate_volatility.index.tolist()

        # 缓冲带机制
        selected = set()
        for ticker in previous_holdings:
            if ticker in candidate_tickers:
                rank = candidate_tickers.index(ticker)
                if rank < buffer_keep_n:
                    selected.add(ticker)

        # 填补空缺
        needed = final_select_n - len(selected)
        if needed > 0:
            for ticker in candidate_tickers:
                if ticker not in selected:
                    selected.add(ticker)
                    if len(selected) >= final_select_n:
                        break

        # 权重分配（等权重）
        weights.loc[rebalance_date, :] = 0.0
        if len(selected) > 0:
            weight_per_stock = 1.0 / len(selected)
            weights.loc[rebalance_date, list(selected)] = weight_per_stock

        previous_holdings = selected

    # 计算空仓时间占比
    bear_ratio = bear_market_count / len(rebalance_dates)

    return weights, bear_ratio


def run_single_test(prices, rebalance_dates, filter_type, filter_params):
    """运行单个参数组合测试"""
    weights, bear_ratio = generate_weights_with_trend_filter(
        prices, rebalance_dates, filter_type, filter_params
    )
    returns = run_backtest(prices, weights, transaction_cost=0.001)
    metrics = calculate_performance_metrics(returns)

    # 计算换手率
    daily_weights = weights.ffill()
    weight_changes = daily_weights.diff().abs().sum(axis=1)
    rebalance_turnover = weight_changes[weight_changes > 1e-6]
    annual_turnover = rebalance_turnover.mean() * 12 if len(rebalance_turnover) > 0 else 0

    return {
        'filter_type': filter_type,
        'params': str(filter_params),
        'bear_ratio': bear_ratio,
        'annual_return': metrics['annual_return'],
        'annual_volatility': metrics['annual_volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'calmar_ratio': metrics['calmar_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'monthly_win_rate': metrics['monthly_win_rate'],
        'annual_turnover': annual_turnover
    }


def main():
    print("=" * 80)
    print("大盘趋势过滤 - 参数敏感性测试")
    print("=" * 80)

    # 加载数据
    print("\n[1/3] 加载数据...")
    prices = load_data()
    rebalance_dates = generate_monthly_rebalance_dates(prices)
    print(f"  数据范围: {prices.index.date[0]} 至 {prices.index.date[-1]}")
    print(f"  调仓次数: {len(rebalance_dates)}")

    # 定义参数扫描空间
    print("\n[2/3] 定义参数扫描空间...")

    test_cases = []

    # ========== 基准：无过滤 ==========
    test_cases.append(('none', {}))

    # ========== 方案1：绝对动量 ==========
    print("  - 绝对动量: 测试窗口 21, 42, 63, 84, 126, 189, 252 天")
    for days in [21, 42, 63, 84, 126, 189, 252]:
        test_cases.append(('absolute_momentum', {'lookback_days': days}))

    # ========== 方案2：快慢均线交叉 ==========
    print("  - 均线交叉: 测试 10/50, 20/50, 20/100, 20/150, 30/100, 50/200")
    for short, long_d in [(10, 50), (20, 50), (20, 100), (20, 150), (30, 100), (50, 200)]:
        test_cases.append(('sma_cross', {'short_days': short, 'long_days': long_d}))

    # ========== 方案3：双动量 ==========
    print("  - 双动量: 测试 (1m, 3m), (1m, 6m), (3m, 6m), (3m, 9m)")
    for short, long_d in [(21, 63), (21, 126), (63, 126), (63, 189)]:
        test_cases.append(('dual_momentum', {'short_days': short, 'long_days': long_d}))

    # ========== 方案4：波动率阈值过滤 ==========
    print("  - 波动率阈值: 测试不同分位数和动量窗口组合")
    for quantile in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        for mom_days in [63, 126, 189]:
            test_cases.append(('volatility_threshold', {
                'vol_window': 60,
                'vol_quantile': quantile,
                'mom_days': mom_days
            }))

    print(f"\n  总共测试 {len(test_cases)} 个参数组合")

    # 运行所有测试
    print("\n[3/3] 运行参数扫描...")
    results = []

    for i, (filter_type, params) in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {filter_type}: {params}")
        try:
            result = run_single_test(prices, rebalance_dates, filter_type, params)
            results.append(result)
        except Exception as e:
            print(f"    ❌ 错误: {e}")

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 计算相对于基准的改善
    baseline = df[df['filter_type'] == 'none'].iloc[0]
    df['excess_return'] = df['annual_return'] - baseline['annual_return']
    df['sharpe_improvement'] = df['sharpe_ratio'] - baseline['sharpe_ratio']
    df['drawdown_reduction'] = df['max_drawdown'] - baseline['max_drawdown']  # 负值更好

    # 打印结果汇总
    print("\n" + "=" * 80)
    print("参数敏感性测试结果汇总")
    print("=" * 80)

    print(f"\n📊 基准（无过滤）:")
    print(f"  年化收益: {baseline['annual_return']:.2%}")
    print(f"  年化波动: {baseline['annual_volatility']:.2%}")
    print(f"  夏普比率: {baseline['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {baseline['max_drawdown']:.2%}")

    # 按夏普比率排序
    print("\n" + "=" * 80)
    print("🏆 TOP 10 表现（按夏普比率排序）")
    print("=" * 80)
    top_sharpe = df.sort_values('sharpe_ratio', ascending=False).head(10)
    for i, (_, row) in enumerate(top_sharpe.iterrows()):
        print(f"\n{i+1}. {row['filter_type']}")
        print(f"   参数: {row['params']}")
        print(f"   空仓占比: {row['bear_ratio']:.1%}")
        print(f"   年化收益: {row['annual_return']:.2%} (超额: {row['excess_return']:+.2%})")
        print(f"   年化波动: {row['annual_volatility']:.2%}")
        print(f"   夏普比率: {row['sharpe_ratio']:.2f} (改善: {row['sharpe_improvement']:+.2f})")
        print(f"   最大回撤: {row['max_drawdown']:.2%} (变化: {row['drawdown_reduction']:+.2%})")

    # 按最大回撤排序
    print("\n" + "=" * 80)
    print("🛡️  TOP 10 表现（按最大回撤排序，回撤越小越好）")
    print("=" * 80)
    top_drawdown = df.sort_values('max_drawdown', ascending=True).head(10)
    for i, (_, row) in enumerate(top_drawdown.iterrows()):
        print(f"\n{i+1}. {row['filter_type']}")
        print(f"   参数: {row['params']}")
        print(f"   空仓占比: {row['bear_ratio']:.1%}")
        print(f"   年化收益: {row['annual_return']:.2%} (超额: {row['excess_return']:+.2%})")
        print(f"   最大回撤: {row['max_drawdown']:.2%} (减少: {row['drawdown_reduction']:+.2%})")
        print(f"   夏普比率: {row['sharpe_ratio']:.2f}")

    # 分方案汇总
    print("\n" + "=" * 80)
    print("📈 各方案最佳表现汇总")
    print("=" * 80)

    for filter_type in df['filter_type'].unique():
        if filter_type == 'none':
            continue
        sub_df = df[df['filter_type'] == filter_type]
        best_sharpe = sub_df.loc[sub_df['sharpe_ratio'].idxmax()]
        best_dd = sub_df.loc[sub_df['max_drawdown'].idxmin()]

        print(f"\n{filter_type}:")
        print(f"  最佳夏普: {best_sharpe['sharpe_ratio']:.2f} (参数: {best_sharpe['params']})")
        print(f"    收益: {best_sharpe['annual_return']:.2%}, 回撤: {best_sharpe['max_drawdown']:.2%}")
        print(f"  最佳回撤: {best_dd['max_drawdown']:.2%} (参数: {best_dd['params']})")
        print(f"    收益: {best_dd['annual_return']:.2%}, 夏普: {best_dd['sharpe_ratio']:.2f}")

    # 保存完整结果
    df.to_csv('trend_filter_sensitivity_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 完整测试结果已保存至: trend_filter_sensitivity_results.csv")

    # 核心结论
    print("\n" + "=" * 80)
    print("🎯 核心结论")
    print("=" * 80)

    any_improvement = (df['sharpe_improvement'] > 0.01).any()
    if not any_improvement:
        print("\n  ❌ 所有大盘过滤方案均未能显著改善夏普比率")
        print("     建议：暂时不要启用大盘趋势过滤，维持 v1.2 基准策略")
    else:
        best = df.iloc[df['sharpe_improvement'].idxmax()]
        print(f"\n  ✅ 找到最优方案: {best['filter_type']}")
        print(f"     参数: {best['params']}")
        print(f"     夏普改善: {best['sharpe_improvement']:+.2f}")
        print(f"     收益变化: {best['excess_return']:+.2%}")
        print(f"     回撤变化: {best['drawdown_reduction']:+.2%}")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    return df


if __name__ == "__main__":
    results_df = main()
