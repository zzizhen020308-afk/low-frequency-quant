#!/usr/bin/env python3
"""
参数鲁棒性检验 - 数据窥探偏差（Data Snooping Bias）分析
在最优参数的"附近邻域"进行网格搜索，验证是否存在"参数平原"

核心思想：
    - 如果最优参数周围表现都稳定 → 参数鲁棒，可实盘
    - 如果只有某个单点表现好，周围都很差 → 过拟合，实盘必崩
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from backtest_engine import (
    load_data,
    calculate_momentum,
    calculate_volatility,
    generate_monthly_rebalance_dates,
    calculate_performance_metrics,
    run_backtest
)


def generate_weights_volatility_filter(
        prices,
        rebalance_dates,
        vol_window=60,
        vol_quantile=0.7,
        mom_days=63,
        momentum_top_n=50,
        final_select_n=20,
        buffer_keep_n=40):
    """
    波动率阈值过滤（本测试的唯一方案）

    逻辑：
        - 计算大盘 60 天滚动波动率
        - 计算历史波动率分位数阈值
        - 低波动环境：永远持仓（不做趋势判断）
        - 高波动环境：用动量判断牛熊，熊市时空仓
    """
    momentum = calculate_momentum(prices)
    volatility = calculate_volatility(prices)

    # 大盘代理（等权重指数）
    market_index = prices.mean(axis=1)
    market_returns = market_index.pct_change(fill_method=None)

    # 预计算波动率指标
    market_vol = market_returns.rolling(vol_window).std() * np.sqrt(252)
    vol_threshold = market_vol.quantile(vol_quantile)

    # 预计算动量指标
    market_mom = market_index / market_index.shift(mom_days) - 1

    # 趋势判断：低波动环境永远持仓，高波动环境检查动量
    # True = 牛市/持仓，False = 熊市/空仓
    trend_indicator = (market_vol <= vol_threshold) | (market_mom > 0)

    bear_market_count = 0

    # 权重矩阵初始化
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    first_rebalance_date = rebalance_dates[0]
    weights.loc[:first_rebalance_date - pd.Timedelta(days=1), :] = 0.0

    previous_holdings = set()

    for rebalance_date in rebalance_dates:
        # ========== 大盘风控检查 ==========
        is_bull_market = trend_indicator.loc[rebalance_date]

        if not is_bull_market:
            # 熊市：全部空仓
            weights.loc[rebalance_date, :] = 0.0
            previous_holdings = set()
            bear_market_count += 1
            continue

        # ========== 选股逻辑（仅在牛市执行） ==========
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

        # ========== 缓冲带机制 ==========
        selected = set()

        # 优先保留老仓位
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

        # 等权重分配
        weights.loc[rebalance_date, :] = 0.0
        if len(selected) > 0:
            weight_per_stock = 1.0 / len(selected)
            weights.loc[rebalance_date, list(selected)] = weight_per_stock

        previous_holdings = selected

    # 空仓时间占比
    bear_ratio = bear_market_count / len(rebalance_dates)

    return weights, bear_ratio


def run_single_parameter_test(prices, rebalance_dates, vol_window, vol_quantile, mom_days):
    """运行单个参数组合测试"""
    weights, bear_ratio = generate_weights_volatility_filter(
        prices, rebalance_dates, vol_window, vol_quantile, mom_days
    )

    returns = run_backtest(prices, weights, transaction_cost=0.001)
    metrics = calculate_performance_metrics(returns)

    return {
        'vol_window': vol_window,
        'vol_quantile': vol_quantile,
        'mom_days': mom_days,
        'bear_ratio': bear_ratio,
        'annual_return': metrics['annual_return'],
        'annual_volatility': metrics['annual_volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'calmar_ratio': metrics['calmar_ratio'],
        'max_drawdown': metrics['max_drawdown'],
    }


def main():
    print("=" * 80)
    print("参数鲁棒性检验 - 数据窥探偏差分析")
    print("Parameter Robustness Test - Data Snooping Bias Analysis")
    print("=" * 80)

    # 加载数据
    print("\n[1/5] 加载数据...")
    prices = load_data()
    rebalance_dates = generate_monthly_rebalance_dates(prices)
    print(f"  数据范围: {prices.index.date[0]} 至 {prices.index.date[-1]}")
    print(f"  调仓次数: {len(rebalance_dates)}")

    # ========== 参数网格设置 ==========
    print("\n[2/5] 定义参数网格...")

    vol_window_fixed = 60  # 固定为60天
    vol_quantiles = [0.60, 0.65, 0.70, 0.75, 0.80]  # 5个水平
    mom_days_list = [42, 63, 84, 105, 126]           # 5个水平（2个月 ~ 6个月）

    total_tests = len(vol_quantiles) * len(mom_days_list)
    print(f"  参数网格: vol_quantile × mom_days = {len(vol_quantiles)} × {len(mom_days_list)} = {total_tests} 组")
    print(f"  vol_quantile: {vol_quantiles}")
    print(f"  mom_days: {mom_days_list}")

    # 运行所有测试
    print(f"\n[3/5] 运行参数网格搜索（共 {total_tests} 组）...")

    results = []
    for i, vol_q in enumerate(vol_quantiles):
        for j, mom_d in enumerate(mom_days_list):
            test_num = i * len(mom_days_list) + j + 1
            print(f"  [{test_num}/{total_tests}] vol_quantile={vol_q:.2f}, mom_days={mom_d}")

            result = run_single_parameter_test(
                prices, rebalance_dates, vol_window_fixed, vol_q, mom_d
            )
            results.append(result)

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 保存完整结果
    df.to_csv('parameter_robustness_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 完整测试结果已保存至: parameter_robustness_results.csv")

    # ========== 构建热力图矩阵 ==========
    print("\n[4/5] 构建热力图矩阵...")

    # 夏普比率矩阵
    sharpe_matrix = pd.pivot_table(
        df,
        index='vol_quantile',
        columns='mom_days',
        values='sharpe_ratio'
    )
    sharpe_matrix = sharpe_matrix.sort_index(ascending=False)  # Y轴从大到小，符合阅读习惯

    # 最大回撤矩阵（取绝对值，方便热力图显示：颜色越深 = 回撤越小 = 越好）
    drawdown_matrix = pd.pivot_table(
        df,
        index='vol_quantile',
        columns='mom_days',
        values='max_drawdown'
    )
    drawdown_matrix = drawdown_matrix.sort_index(ascending=False)

    # 年化收益矩阵
    return_matrix = pd.pivot_table(
        df,
        index='vol_quantile',
        columns='mom_days',
        values='annual_return'
    )
    return_matrix = return_matrix.sort_index(ascending=False)

    # 空仓时间矩阵
    bear_ratio_matrix = pd.pivot_table(
        df,
        index='vol_quantile',
        columns='mom_days',
        values='bear_ratio'
    )
    bear_ratio_matrix = bear_ratio_matrix.sort_index(ascending=False)

    # ========== 绘制热力图 ==========
    print("\n[5/5] 绘制热力图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('参数鲁棒性检验 - 波动率阈值过滤方案\nParameter Robustness Heatmap',
                 fontsize=16, fontweight='bold', y=0.98)

    # 图1：夏普比率热力图
    ax1 = axes[0, 0]
    sns.heatmap(sharpe_matrix,
                annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.80,  # 以基准夏普0.80为中心
                ax=ax1, cbar_kws={'label': 'Sharpe Ratio'})
    ax1.set_title('夏普比率热力图\nSharpe Ratio Heatmap', fontsize=12, fontweight='bold')
    ax1.set_xlabel('动量窗口 (天)\nmom_days (days)')
    ax1.set_ylabel('波动率分位数\nvol_quantile')
    # 标注最优参数位置
    ax1.scatter(1.5, 1.5, marker='*', s=300, color='white', edgecolor='black', linewidth=2,
                label='最优参数 (Original Best)')
    ax1.legend(loc='upper right', fontsize=8)

    # 图2：最大回撤热力图
    ax2 = axes[0, 1]
    sns.heatmap(drawdown_matrix * 100,  # 转为百分比
                annot=True, fmt='.1f', cmap='RdYlGn_r',  # 反向色阶：回撤越小越好
                center=-25,
                ax=ax2, cbar_kws={'label': 'Max Drawdown (%)'})
    ax2.set_title('最大回撤热力图 (%)\nMax Drawdown Heatmap', fontsize=12, fontweight='bold')
    ax2.set_xlabel('动量窗口 (天)\nmom_days (days)')
    ax2.set_ylabel('波动率分位数\nvol_quantile')
    ax2.scatter(1.5, 1.5, marker='*', s=300, color='white', edgecolor='black', linewidth=2)

    # 图3：年化收益热力图
    ax3 = axes[1, 0]
    sns.heatmap(return_matrix * 100,
                annot=True, fmt='.1f', cmap='RdYlGn',
                center=12,
                ax=ax3, cbar_kws={'label': 'Annual Return (%)'})
    ax3.set_title('年化收益热力图 (%)\nAnnual Return Heatmap', fontsize=12, fontweight='bold')
    ax3.set_xlabel('动量窗口 (天)\nmom_days (days)')
    ax3.set_ylabel('波动率分位数\nvol_quantile')
    ax3.scatter(1.5, 1.5, marker='*', s=300, color='white', edgecolor='black', linewidth=2)

    # 图4：空仓时间占比热力图
    ax4 = axes[1, 1]
    sns.heatmap(bear_ratio_matrix * 100,
                annot=True, fmt='.1f', cmap='Blues',
                ax=ax4, cbar_kws={'label': 'Bear Market Time (%)'})
    ax4.set_title('空仓时间占比 (%)\nBear Market Time Ratio', fontsize=12, fontweight='bold')
    ax4.set_xlabel('动量窗口 (天)\nmom_days (days)')
    ax4.set_ylabel('波动率分位数\nvol_quantile')
    ax4.scatter(1.5, 1.5, marker='*', s=300, color='white', edgecolor='black', linewidth=2)

    plt.tight_layout()
    plt.savefig('parameter_robustness_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"💾 热力图已保存至: parameter_robustness_heatmap.png")
    plt.close()

    # ========== 控制台总结分析 ==========
    print("\n" + "=" * 80)
    print("📊 参数鲁棒性分析总结")
    print("=" * 80)

    # 基准参数
    baseline_sharpe = 0.80  # 无过滤基准
    baseline_drawdown = -0.3847  # 无过滤基准

    print(f"\n📌 基准（无过滤）:")
    print(f"   夏普比率: {baseline_sharpe:.2f}")
    print(f"   最大回撤: {baseline_drawdown:.2%}")

    print(f"\n📌 最优参数（原测试）: vol_quantile=0.70, mom_days=63")
    best = df[(df['vol_quantile'] == 0.70) & (df['mom_days'] == 63)].iloc[0]
    print(f"   夏普比率: {best['sharpe_ratio']:.2f} (vs 基准: +{best['sharpe_ratio']-baseline_sharpe:.2f})")
    print(f"   最大回撤: {best['max_drawdown']:.2%} (vs 基准: {best['max_drawdown']-baseline_drawdown:+.2%})")
    print(f"   空仓时间: {best['bear_ratio']:.1%}")

    # ========== 鲁棒性统计 ==========
    print(f"\n" + "=" * 80)
    print("🎯 鲁棒性存活检验")
    print("=" * 80)

    # 定义"存活"标准：夏普 >= 0.75 且 回撤 <= -25%
    sharpe_threshold = 0.75
    drawdown_threshold = -0.25

    df['survived'] = (df['sharpe_ratio'] >= sharpe_threshold) & (df['max_drawdown'] <= drawdown_threshold)
    survival_count = df['survived'].sum()
    survival_rate = survival_count / total_tests

    print(f"\n✅ 存活标准: 夏普 >= {sharpe_threshold:.2f} 且 最大回撤 <= {drawdown_threshold:.0%}")
    print(f"   存活组合数: {survival_count} / {total_tests}")
    print(f"   鲁棒性存活率: {survival_rate:.1%}")

    # 更严格标准
    strict_sharpe = 0.80
    strict_drawdown = -0.25
    df['survived_strict'] = (df['sharpe_ratio'] >= strict_sharpe) & (df['max_drawdown'] <= strict_drawdown)
    strict_survival_count = df['survived_strict'].sum()
    strict_survival_rate = strict_survival_count / total_tests

    print(f"\n🔥 严格存活标准: 夏普 >= {strict_sharpe:.2f} 且 最大回撤 <= {strict_drawdown:.0%}")
    print(f"   存活组合数: {strict_survival_count} / {total_tests}")
    print(f"   严格鲁棒性存活率: {strict_survival_rate:.1%}")

    # ========== 参数平原分析 ==========
    print(f"\n" + "=" * 80)
    print("🗺️ 参数平原分析")
    print("=" * 80)

    print(f"\n🏆 所有存活组合（夏普 >= 0.75 且 回撤 <= -25%）:")
    survived_df = df[df['survived']].sort_values('sharpe_ratio', ascending=False)
    for _, row in survived_df.iterrows():
        marker = " ⭐ 原最优参数" if (row['vol_quantile'] == 0.70 and row['mom_days'] == 63) else ""
        print(f"   ({row['vol_quantile']:.2f}, {row['mom_days']:3d}): "
              f"夏普={row['sharpe_ratio']:.2f}, 回撤={row['max_drawdown']:.2%}, "
              f"空仓={row['bear_ratio']:.1%}{marker}")

    # ========== 参数敏感性分析 ==========
    print(f"\n" + "=" * 80)
    print("📈 参数敏感性分析")
    print("=" * 80)

    # 按 vol_quantile 分组统计
    print(f"\n按 vol_quantile 分组（平均表现）:")
    by_quantile = df.groupby('vol_quantile').agg({
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'bear_ratio': 'mean'
    }).sort_index()
    for idx, row in by_quantile.iterrows():
        print(f"   vol_quantile={idx:.2f}: 夏普={row['sharpe_ratio']:.2f}, "
              f"回撤={row['max_drawdown']:.2%}, 空仓={row['bear_ratio']:.1%}")

    # 按 mom_days 分组统计
    print(f"\n按 mom_days 分组（平均表现）:")
    by_mom = df.groupby('mom_days').agg({
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'bear_ratio': 'mean'
    }).sort_index()
    for idx, row in by_mom.iterrows():
        print(f"   mom_days={idx:3d}: 夏普={row['sharpe_ratio']:.2f}, "
              f"回撤={row['max_drawdown']:.2%}, 空仓={row['bear_ratio']:.1%}")

    # ========== 最终结论 ==========
    print(f"\n" + "=" * 80)
    print("🎯 最终结论")
    print("=" * 80)

    if survival_rate >= 0.6:
        print(f"\n✅ 【参数鲁棒性良好】")
        print(f"   存活率 {survival_rate:.1%} >= 60%，存在明显的参数平原")
        print(f"   最优参数不是孤点，周围表现稳定，可以实盘")
    elif survival_rate >= 0.3:
        print(f"\n⚠️  【参数鲁棒性中等】")
        print(f"   存活率 {survival_rate:.1%} 在 30%-60% 之间")
        print(f"   建议谨慎使用，可考虑取参数平原的中心值而非单点最优")
    else:
        print(f"\n❌ 【参数鲁棒性差 - 过拟合风险高！】")
        print(f"   存活率 {survival_rate:.1%} < 30%，很可能是数据窥探的结果")
        print(f"   强烈建议不要实盘！要么改进方案，要么放弃大盘过滤")

    print(f"\n💡 推荐实盘参数选择策略:")
    print(f"   不要选单点最优参数，应选择参数平原的中心位置")
    print(f"   推荐范围: vol_quantile = 0.65-0.75, mom_days = 42-84")

    print("\n" + "=" * 80)
    print("参数鲁棒性检验完成!")
    print("=" * 80)

    return df


if __name__ == "__main__":
    results_df = main()
