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



def generate_weights(prices, rebalance_dates, momentum_top_n=50, final_select_n=20, buffer_keep_n=40):
    """
    【v3.0 Beta Hedging】大盘贝塔对冲架构 —— 终极进化版

    核心理念：不再做空个股来对冲！那会损失掉空头个股的Alpha。
    正确做法：多头保留选股Alpha，空头只剥离市场Beta。

    设计哲学：
        Alpha + Beta  = 传统多头策略收益
        Alpha - Beta*K = Beta对冲后的纯Alpha + 方向判断收益
        其中K是动态对冲系数，由四象限状态机决定

    大盘四象限与动态对冲系数：
        状态 1（低波动 + 动量向上）：Long 100%, Short 0%   → 净敞口 +100%
            牛市：全力进攻，不浪费子弹在空头上，享受Beta收益

        状态 2（高波动 + 动量向上）：Long 70%, Short -30% → 净敞口 +40%
            高位震荡：保留70%股票Alpha，但用30%大盘空头对冲尾部风险

        状态 3（低波动 + 动量向下）：Long 50%, Short -50% → 净敞口 0%
            阴跌市：纯市场中性模式，完全剥离大盘Beta，只赚选股Alpha

        状态 4（高波动 + 动量向下）：Long 20%, Short -80% → 净敞口 -60%
            崩盘/股灾：深度净做空，从暴跌中主动获利

    双层风控缓冲带：
        第一层：净敞口缓冲带（±15%）—— 防止状态边界反复横跳
        第二层：多头个股缓冲带（前40名保留）—— 减少无谓换仓摩擦

    空头执行方式：
        不再做空个股！直接做空大盘指数代理（SPY_Proxy）
        做空仓位=纯Beta暴露，不会损失个股Alpha

    参数:
        momentum_top_n: 动量筛选阈值（前N名进入候选池）
        final_select_n: 满仓时多头持仓数（默认20只，单腿5%权重）
        buffer_keep_n: 多头个股缓冲带阈值（前N名老仓位优先保留）
    """
    print(f"计算因子并生成持仓权重...")
    print(f"  调仓次数: {len(rebalance_dates)}")
    print(f"  策略模式: 【v3.0 Beta Hedging 大盘贝塔对冲架构】")
    print(f"    动量窗口: 52 天 (跳过最近 21 天)")
    print(f"    波动率窗口: 60 天")
    print(f"    波动率分位数阈值: 67.5%")
    print(f"    净敞口缓冲带: ±15%")
    print(f"    多头缓冲带: 前 {buffer_keep_n} 名优先保留")
    print(f"    空头执行: 大盘指数代理 (SPY_Proxy)")

    # ========== 第一步：构建大盘代理资产 ==========
    # 将等权重大盘指数作为一个合成资产加入price矩阵
    # 这是我们的空头对冲工具
    market_index = prices.mean(axis=1)
    prices_with_spy = prices.copy()
    prices_with_spy['SPY_Proxy'] = market_index
    spy_column_name = 'SPY_Proxy'

    # ========== 预计算因子 ==========
    # 统一使用 52-21 动量窗口：与大盘风控信号对齐
    momentum = calculate_momentum(prices_with_spy, lookback_days=52, skip_days=21)
    volatility = calculate_volatility(prices_with_spy)

    # ========== 大盘风控因子预计算 ==========
    market_returns = market_index.pct_change(fill_method=None)

    # 大盘 60 天滚动波动率
    market_vol = market_returns.rolling(60).std() * np.sqrt(252)
    vol_threshold = market_vol.quantile(0.675)
    is_low_vol = market_vol <= vol_threshold

    # 大盘 52 天绝对动量
    market_mom = market_index / market_index.shift(52) - 1
    is_mom_up = market_mom > 0

    # ========== 权重矩阵初始化 ==========
    # NaN = 非调仓日，vectorbt 引擎会自动沿用前值
    # 注意权重矩阵要包含 SPY_Proxy
    weights = pd.DataFrame(np.nan, index=prices_with_spy.index, columns=prices_with_spy.columns)

    # 第一个调仓日之前明确空仓
    first_rebalance_date = rebalance_dates[0]
    weights.loc[:first_rebalance_date - pd.Timedelta(days=1), :] = 0.0

    # ========== 状态变量 ==========
    previous_long_holdings = set()   # 上一期多头持仓
    previous_net_exposure = 1.0      # 上一期净敞口（用于第一层缓冲带）

    # 记录上期实际的 long_target / short_target
    prev_long_target = 1.0
    prev_short_target = 0.0

    fixed_weight_per_stock = 1.0 / final_select_n  # 固定单只 5%

    # 统计各状态出现次数
    state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    buffer_skip_count = 0  # 被净敞口缓冲带拦下的调整次数

    # ========== 主循环：每个调仓日执行 ==========
    for rebalance_date in rebalance_dates:

        # 注意：排除 SPY_Proxy 作为个股被选入多头
        all_tickers = momentum.columns
        stock_tickers = [t for t in all_tickers if t != spy_column_name]

        date_momentum = momentum.loc[rebalance_date].dropna()
        date_volatility = volatility.loc[rebalance_date].dropna()

        # 只考虑真实股票（不包括SPY代理）的因子有效性
        common_tickers = date_momentum.index.intersection(date_volatility.index)
        common_tickers = [t for t in common_tickers if t != spy_column_name]

        if len(common_tickers) == 0:
            weights.loc[rebalance_date, :] = 0.0
            previous_long_holdings = set()
            previous_net_exposure = 0.0
            continue

        # ============================================================
        # 【第一步：大盘四象限状态机 + 动态多空目标】
        # ============================================================
        current_low_vol = is_low_vol.loc[rebalance_date]
        current_mom_up = is_mom_up.loc[rebalance_date]

        # 计算理论多空目标比例
        if current_low_vol and current_mom_up:
            state = 1
            theory_long_target = 1.0   # 状态1：牛市满仓
            theory_short_target = 0.0
        elif (not current_low_vol) and current_mom_up:
            state = 2
            theory_long_target = 0.7   # 状态2：高位震荡，30% 对冲
            theory_short_target = -0.3
        elif current_low_vol and (not current_mom_up):
            state = 3
            theory_long_target = 0.5   # 状态3：阴跌市，纯市场中性
            theory_short_target = -0.5
        else:
            state = 4
            theory_long_target = 0.2   # 状态4：崩盘期，净做空获利
            theory_short_target = -0.8

        theory_net_exposure = theory_long_target + theory_short_target

        # ============================================================
        # 【第二步：净敞口缓冲带（Net Exposure Buffer）】
        # 防止大盘状态在边界上反复横跳导致的频繁调仓
        # ============================================================
        if abs(theory_net_exposure - previous_net_exposure) < 0.15:
            # 变化幅度 < 15%，拒绝切换，沿用上期多空目标
            final_long_target = prev_long_target
            final_short_target = prev_short_target
            buffer_skip_count += 1
        else:
            # 变化幅度足够大，执行状态切换
            final_long_target = theory_long_target
            final_short_target = theory_short_target
            state_counts[state] += 1

        # ============================================================
        # 【第三步：计算本期多头持仓数量】
        # ============================================================
        N_long = max(0, int(final_select_n * final_long_target))

        # ============================================================
        # 【第四步：多头选股 + 缓冲带】
        # ============================================================
        final_long = set()

        if N_long > 0:
            # 第一步：动量前 50 名进入多头候选池
            # 注意：只从真实股票中选，不包括 SPY_Proxy
            valid_mom = date_momentum[common_tickers]
            top_momentum_long = valid_mom.nlargest(momentum_top_n).index

            # 第二步：候选池按波动率从小到大排序
            long_candidate_vol = date_volatility[top_momentum_long].sort_values(ascending=True)
            long_candidate_tickers = long_candidate_vol.index.tolist()

            # 缓冲带逻辑：优先保留老多头
            for ticker in previous_long_holdings:
                if ticker in long_candidate_tickers:
                    rank = long_candidate_tickers.index(ticker)
                    if rank < buffer_keep_n:
                        final_long.add(ticker)

            # 按波动率排名填补空缺
            needed_long = N_long - len(final_long)
            if needed_long > 0:
                for ticker in long_candidate_tickers:
                    if ticker not in final_long:
                        final_long.add(ticker)
                        if len(final_long) >= N_long:
                            break

        # ========== 权重分配 ==========
        # 调仓日先清零所有仓位
        weights.loc[rebalance_date, :] = 0.0

        # 多头分配：每只固定 +5%
        if len(final_long) > 0:
            weights.loc[rebalance_date, list(final_long)] = fixed_weight_per_stock

        # 空头分配：直接分配给 SPY_Proxy
        # 注意：final_short_target 是负数（如 -0.3）
        if final_short_target < 0:
            weights.loc[rebalance_date, spy_column_name] = final_short_target

        # ========== 更新状态变量 ==========
        previous_long_holdings = final_long
        previous_net_exposure = final_long_target + final_short_target
        prev_long_target = final_long_target
        prev_short_target = final_short_target

    # ========== 打印统计结果 ==========
    total = len(rebalance_dates)
    print(f"\n  【大盘四象限状态分布】")
    print(f"    状态1（低波+向上→100%净多）: {state_counts[1]} 次 ({state_counts[1]/total:.1%})")
    print(f"    状态2（高波+向上→40%净多）: {state_counts[2]} 次 ({state_counts[2]/total:.1%})")
    print(f"    状态3（低波+向下→0%净敞口）: {state_counts[3]} 次 ({state_counts[3]/total:.1%})")
    print(f"    状态4（高波+向下→60%净空）: {state_counts[4]} 次 ({state_counts[4]/total:.1%})")
    print(f"\n  【净敞口缓冲带效果】")
    print(f"    被拦下的调整次数: {buffer_skip_count} 次 ({buffer_skip_count/total:.1%})")

    return weights, prices_with_spy

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
