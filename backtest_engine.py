#!/usr/bin/env python3
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='sp500_adjusted_close.parquet'):
    """加载价格数据"""
    prices = pd.read_parquet(filepath)
    prices.index = pd.to_datetime(prices.index)
    return prices


def calculate_momentum(prices, lookback_days=252, skip_days=21):
    """计算动量因子"""
    momentum = prices.shift(skip_days) / prices.shift(skip_days + lookback_days) - 1
    return momentum


def calculate_volatility(prices, window_days=20):
    """计算波动率因子"""
    returns = prices.pct_change(fill_method=None)
    volatility = returns.rolling(window=window_days).std() * np.sqrt(252)
    return volatility


def generate_monthly_rebalance_dates(prices):
    """生成每月调仓日期（每月第一个交易日）"""
    month_ends = prices.resample('ME').last().index
    rebalance_dates = []
    for month_end in month_ends:
        next_trading_day = prices.index[prices.index > month_end]
        if len(next_trading_day) > 0:
            rebalance_dates.append(next_trading_day[0])
    min_history_days = 252 + 21
    rebalance_dates = [d for d in rebalance_dates if d >= prices.index[min_history_days]]
    return pd.DatetimeIndex(rebalance_dates)


def generate_weights(prices, rebalance_dates, momentum_top_n=50, final_select_n=20, buffer_keep_n=40, weight_scheme='equal'):
    """在每个调仓日生成目标权重（引入缓冲带机制降低换手率）

    参数:
        buffer_keep_n: 缓冲带阈值，老仓位在波动率排名前N名内则优先保留
        weight_scheme: str, 权重方案:
            - 'equal': 等权重（默认，v1.0 基准，推荐使用）
            - 'inverse_volatility': 波动率倒数加权（v1.1 测试版，效果不佳）
    """
    print(f"计算因子并生成持仓权重（缓冲带模式）...")
    print(f"  调仓次数: {len(rebalance_dates)}")
    print(f"  缓冲带阈值: 波动率排名前 {buffer_keep_n} 名优先保留老仓位")

    # 权重方案显示
    weight_scheme_names = {
        'equal': '等权重（Equal Weight）- 推荐',
        'inverse_volatility': '波动率倒数加权（Inverse Volatility）- 效果不佳'
    }
    print(f"  权重方案: {weight_scheme_names.get(weight_scheme, weight_scheme)}")

    momentum = calculate_momentum(prices)
    volatility = calculate_volatility(prices)

    # 初始化全为 NaN，确保非调仓日引擎不产生任何交易动作
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    # 在第一个调仓日之前，明确将所有股票权重设为 0（空仓）
    first_rebalance_date = rebalance_dates[0]
    weights.loc[:first_rebalance_date - pd.Timedelta(days=1), :] = 0.0

    # 记录上一期持仓，用于缓冲带逻辑
    previous_holdings = set()

    for rebalance_date in rebalance_dates:
        date_momentum = momentum.loc[rebalance_date].dropna()
        date_volatility = volatility.loc[rebalance_date].dropna()
        common_tickers = date_momentum.index.intersection(date_volatility.index)

        if len(common_tickers) == 0:
            previous_holdings = set()
            continue

        # ========== 第一步：选出动量最高的前 momentum_top_n 只股票 ==========
        top_momentum = date_momentum[common_tickers].nlargest(momentum_top_n).index

        # ========== 第二步：按波动率从小到大排序这 50 只候选股 ==========
        candidate_volatility = date_volatility[top_momentum].sort_values(ascending=True)
        candidate_tickers = candidate_volatility.index.tolist()

        # ========== 核心缓冲带逻辑：优先保留老仓位 ==========
        selected = set()

        # 检查上一期持仓：如果在候选池中且波动率排名在 buffer_keep_n 以内，则优先保留
        for ticker in previous_holdings:
            if ticker in candidate_tickers:
                rank = candidate_tickers.index(ticker)
                if rank < buffer_keep_n:
                    selected.add(ticker)

        # ========== 核心缓冲带逻辑：替补新仓位填补空缺 ==========
        needed = final_select_n - len(selected)
        if needed > 0:
            # 按波动率排名顺序，挑选尚未入选的股票填补空缺
            for ticker in candidate_tickers:
                if ticker not in selected:
                    selected.add(ticker)
                    if len(selected) >= final_select_n:
                        break

        selected_list = list(selected)

        # ========== 权重分配 ==========
        # 调仓日当天，先将所有股票目标权重设为 0.0（不在 selected 中的将被引擎自动平仓）
        weights.loc[rebalance_date, :] = 0.0

        if len(selected_list) > 0:
            if weight_scheme == 'equal':
                # 等权重
                weight_per_stock = 1.0 / len(selected_list)
                weights.loc[rebalance_date, selected_list] = weight_per_stock

            elif weight_scheme == 'inverse_volatility':
                # 波动率倒数加权（不推荐）
                selected_vol = date_volatility[selected_list]
                epsilon = 1e-6
                smoothed_vol = selected_vol + epsilon
                inv_vol = 1.0 / smoothed_vol
                inv_vol_sum = inv_vol.sum()
                risk_parity_weights = inv_vol / inv_vol_sum
                weights.loc[rebalance_date, selected_list] = risk_parity_weights

            else:
                raise ValueError(f"不支持的权重方案: {weight_scheme}")

        # 更新上一期持仓记录，供下一个调仓日使用
        previous_holdings = selected

    # 绝对不能 ffill()，否则每天都会产生调仓手续费！

    return weights


def calculate_performance_metrics(returns, risk_free_rate=0.0):
    """计算完整的绩效指标"""
    returns = returns.copy()

    # 累计收益率
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    # 年化收益率
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)

    # 夏普比率
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # 卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 日胜率
    daily_win_rate = (returns > 0).mean()

    # 月度胜率
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_returns > 0).mean()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'daily_win_rate': daily_win_rate,
        'monthly_win_rate': monthly_win_rate,
        'cumulative_returns': cumulative,
        'drawdown': drawdown
    }


def run_backtest(prices, weights, transaction_cost=0.001):
    """运行回测（使用 vectorbt 原生订单引擎重构）"""
    print(f"\n运行回测...")
    print(f"  单边交易成本: {transaction_cost:.1%}")

    # 使用 from_orders 配合 targetpercent 实现目标权重调仓
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type='targetpercent',  # 告诉引擎传入的 size 是目标资产占比
        group_by=True,              # 核心：将 500 只股票合并为一个资产组合计算净值
        cash_sharing=True,          # 核心：允许平仓释放的资金用于买入新股票
        call_seq='auto',            # 自动决定买卖顺序（通常是先平仓后建仓）
        fees=transaction_cost,      # 交易成本（滑点+佣金）
        freq='1D',                  # 数据频率为日
        init_cash=100000.0          # 假设初始资金为 10 万美元
    )

    # portfolio.returns() 会返回扣除各项成本后的每日真实组合收益率（Series）
    # 这个返回值可以直接无缝接入你现有的 calculate_performance_metrics 函数
    return portfolio.returns()


def print_backtest_results(metrics):
    """打印回测结果"""
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)

    print(f"\n总收益率:       {metrics['total_return']:>10.2%}")
    print(f"年化收益率:     {metrics['annual_return']:>10.2%}")
    print(f"年化波动率:     {metrics['annual_volatility']:>10.2%}")
    print(f"夏普比率:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"卡玛比率:       {metrics['calmar_ratio']:>10.2f}")
    print(f"最大回撤:       {metrics['max_drawdown']:>10.2%}")
    print(f"日胜率:         {metrics['daily_win_rate']:>10.2%}")
    print(f"月胜率:         {metrics['monthly_win_rate']:>10.2%}")

    print("\n" + "=" * 60)


def plot_results(metrics):
    """绘制回测结果图表"""
    print("\n生成回测图表...")

    # 累计收益曲线
    fig = metrics['cumulative_returns'].vbt.plot(
        title='动量波动率复合策略 - 累计收益曲线',
        trace_kwargs=dict(name='策略净值', line=dict(width=2))
    )
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='净值',
        showlegend=True
    )
    fig.show()

    # 回撤曲线
    drawdown_fig = metrics['drawdown'].vbt.plot(
        title='动量波动率复合策略 - 回撤曲线',
        trace_kwargs=dict(name='回撤', line=dict(color='red', width=2), fill='tozeroy')
    )
    drawdown_fig.update_layout(
        xaxis_title='日期',
        yaxis_title='回撤',
        showlegend=True
    )
    drawdown_fig.show()


def main(plot_charts=False):
    print("=" * 60)
    print("动量波动率复合策略回测引擎")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载价格数据...")
    prices = load_data()
    print(f"  数据形状: {prices.shape[0]} 天 x {prices.shape[1]} 只股票")
    print(f"  日期范围: {prices.index.date[0]} 至 {prices.index.date[-1]}")

    # 生成调仓日期
    print("\n[2/5] 生成月度调仓日期...")
    rebalance_dates = generate_monthly_rebalance_dates(prices)
    print(f"  调仓日期数量: {len(rebalance_dates)}")
    print(f"  首个调仓日: {rebalance_dates[0].date()}")
    print(f"  最后调仓日: {rebalance_dates[-1].date()}")

    # 生成权重
    print("\n[3/5] 生成持仓权重...")
    weights = generate_weights(
        prices,
        rebalance_dates,
        momentum_top_n=50,
        final_select_n=20,
        weight_scheme='equal'  # ✅ 默认使用等权重（推荐）
        # weight_scheme='inverse_volatility'  # 如需测试其他方案，取消注释这行
    )

    # 【优化2】使用 vectorbt 回测后再统计真实持仓
    # 先运行回测
    print("\n[4/5] 运行回测...")
    returns = run_backtest(
        prices,
        weights,
        transaction_cost=0.001
    )

    # 计算绩效指标
    print("\n[5/5] 计算绩效指标...")
    metrics = calculate_performance_metrics(returns)

    # 【优化3】计算并打印换手率统计
    print("\n[补充] 计算换手率统计...")
    # 用 ffill 还原每日真实持仓，然后计算权重变化
    daily_weights = weights.ffill()
    # 计算每笔调仓的换手率
    weight_changes = daily_weights.diff().abs().sum(axis=1)
    rebalance_turnover = weight_changes[weight_changes > 1e-6]  # 只保留有调仓的日期

    print(f"  调仓次数: {len(rebalance_turnover)} 次")
    print(f"  平均单次换手率: {rebalance_turnover.mean():.2%}")
    print(f"  年化换手率: {rebalance_turnover.mean() * 12:.2%}")
    print(f"  累计交易成本: {rebalance_turnover.sum() * 0.001:.2%}")

    # 打印结果
    print_backtest_results(metrics)

    # 保存回测结果
    print("\n保存回测结果...")
    metrics['cumulative_returns'].to_csv('strategy_cumulative_returns.csv', header=['cumulative_return'])
    print("  累计收益已保存至: strategy_cumulative_returns.csv")

    # 绘制图表
    if plot_charts:
        plot_results(metrics)

    print("\n回测完成!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = main(plot_charts=False)
