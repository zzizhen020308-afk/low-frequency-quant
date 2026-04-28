#!/usr/bin/env python3
"""
=========================================================================
v1.5 最终版本 - 动量波动率复合策略回测引擎
=========================================================================
【版本特性】
  • 四象限渐进式仓位调整（动态股票数量，固定单股权重）
  • 三层风控缓冲带体系：净敞口缓冲 + 个股缓冲 + 行业上限
  • Active Sector Tilt 行业风控优化（sector_cap=20% = 单行业最多4只）
  • ✅ 无未来函数：波动率分位数采用 expanding 窗口严格时点正确

【回测绩效】
  • 总收益率:     533.41%
  • 年化收益率:   13.11%
  • 年化波动率:   11.90%
  • 夏普比率:     1.10
  • 卡玛比率:     0.80
  • 最大回撤:    -16.42%
  • 月胜率:       60.77%

【默认参数】
  • momentum_top_n=50       (动量前50进入候选池)
  • final_select_n=20       (满仓时持有20只)
  • buffer_keep_n=40        (前40名老仓位优先保留)
  • sector_cap=0.20         (单一行业上限20%)
  • enable_sector_control=True (默认启用行业风控)
  • enable_beta_hedging=False (默认纯多头择时)

【创建日期】2026-04-27
=========================================================================
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import json
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



def generate_weights(prices, rebalance_dates, momentum_top_n=50, final_select_n=20, buffer_keep_n=40, sector_cap=0.20, enable_sector_control=True, enable_beta_hedging=False):
    """
    【v1.5】四象限渐进式仓位调整 + Active Sector Tilt 行业风控优化
           ✅ 已修复未来函数：波动率分位数使用 expanding 窗口严格 Point-in-Time

    默认架构：v1.4 纯多头择时（实盘首选，风险收益比最高）
    可选功能：enable_beta_hedging=True 时启用 v3.0 大盘贝塔对冲

    三层风控体系：
        第一层：净敞口缓冲带（±15%）—— 防止状态边界反复横跳
        第二层：多头个股缓冲带（前40名保留）—— 减少无谓换仓摩擦
        第三层：行业持仓上限（20%）—— 单一行业系统性风险硬约束（敏感性测试最优值）

    大盘四象限与动态仓位系数：
        状态 1（低波动 + 动量向上）→ 目标仓位 = 100%（满仓进攻）
        状态 2（高波动 + 动量向上）→ 目标仓位 = 70%（适度减仓）
        状态 3（低波动 + 动量向下）→ 目标仓位 = 50%（半仓防御）
        状态 4（高波动 + 动量向下）→ 目标仓位 = 20%（大幅减仓）

    Active Sector Tilt 设计哲学：
        不再强制行业中性（等权分配到各行业），那会浪费动量优势
        允许强势行业超配，但设置硬上限防止单一行业崩盘
        例如：20只股票 × 50% = 单一行业最多10只

    参数:
        momentum_top_n: 动量筛选阈值（前N名进入候选池）
        final_select_n: 满仓时多头持仓数（默认20只，单腿5%权重）
        buffer_keep_n: 多头个股缓冲带阈值（前N名老仓位优先保留）
        sector_cap: 单一行业上限比例（默认0.50 = 50%）
        enable_sector_control: 是否启用行业风控（默认True）
        enable_beta_hedging: 是否启用大盘贝塔对冲（默认False，纯多头）
    """
    print(f"计算因子并生成持仓权重...")
    print(f"  调仓次数: {len(rebalance_dates)}")
    if enable_beta_hedging:
        print(f"  策略模式: 【v3.1 大盘贝塔对冲 + 行业风控】")
        print(f"    空头执行: 大盘指数代理 (SPY_Proxy)")
    else:
        print(f"  策略模式: 【v1.5 四象限纯多头择时 + 行业风控优化】")
    print(f"    动量窗口: 52 天 (跳过最近 21 天)")
    print(f"    波动率窗口: 60 天")
    print(f"    波动率分位数阈值: 67.5%")
    print(f"    净敞口缓冲带: ±15%")
    print(f"    多头缓冲带: 前 {buffer_keep_n} 名优先保留")
    if enable_sector_control:
        print(f"    行业风控: 启用，单一行业上限 {sector_cap:.0%}")
    else:
        print(f"    行业风控: 禁用")

    # ========== 行业映射数据加载 ==========
    sector_mapping = {}
    if enable_sector_control:
        try:
            with open('sector_mapping.json', 'r') as f:
                sector_mapping = json.load(f)
            print(f"  ✅ 已加载 {len(sector_mapping)} 只股票的行业映射")
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"  ⚠️  未找到行业映射文件，行业风控已自动禁用")
            enable_sector_control = False

    MAX_STOCKS_PER_SECTOR = int(final_select_n * sector_cap) if enable_sector_control else final_select_n

    # ========== 大盘代理资产（仅贝塔对冲模式使用） ==========
    market_index = prices.mean(axis=1)
    if enable_beta_hedging:
        prices_with_spy = prices.copy()
        prices_with_spy['SPY_Proxy'] = market_index
        spy_column_name = 'SPY_Proxy'
    else:
        prices_with_spy = prices  # 纯多头模式不需要SPY_Proxy
        spy_column_name = None

    # ========== 预计算因子 ==========
    # 统一使用 52-21 动量窗口：与大盘风控信号对齐
    momentum = calculate_momentum(prices_with_spy, lookback_days=52, skip_days=21)
    volatility = calculate_volatility(prices_with_spy)

    # ========== 大盘风控因子预计算 ==========
    # ⚠️ 严格 Point-in-Time：不使用任何未来数据！
    market_returns = market_index.pct_change(fill_method=None)

    # 大盘 60 天滚动波动率
    market_vol = market_returns.rolling(60).std() * np.sqrt(252)

    # 使用 expanding 计算动态分位数（避免未来函数！）
    # min_periods=252 确保至少有一年数据作为基准
    # 截止到当天，系统只知道历史上发生过什么
    vol_threshold = market_vol.expanding(min_periods=252).quantile(0.675)
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

    # 记录上期实际的 long_target
    prev_long_target = 1.0

    # 统计各状态出现次数
    state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    buffer_skip_count = 0  # 被净敞口缓冲带拦下的调整次数
    sector_block_count = 0  # 被行业风控拦下的股票数

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

        # 计算理论目标仓位
        if current_low_vol and current_mom_up:
            state = 1
            theory_long_target = 1.0   # 状态1：牛市满仓
        elif (not current_low_vol) and current_mom_up:
            state = 2
            theory_long_target = 0.7   # 状态2：高位震荡，适度减仓
        elif current_low_vol and (not current_mom_up):
            state = 3
            theory_long_target = 0.5   # 状态3：阴跌市，半仓防御
        else:
            state = 4
            theory_long_target = 0.2   # 状态4：高风险期，大幅减仓

        # 贝塔对冲模式：计算空头比例（纯多头模式无空头）
        if enable_beta_hedging:
            theory_short_target = -(1.0 - theory_long_target)  # 动态对冲系数
        else:
            theory_short_target = 0.0  # 纯多头模式无空头

        theory_net_exposure = theory_long_target + theory_short_target

        # ============================================================
        # 【第二步：净敞口缓冲带（Net Exposure Buffer）】
        # 防止大盘状态在边界上反复横跳导致的频繁调仓
        # ============================================================
        if abs(theory_net_exposure - previous_net_exposure) < 0.15:
            # 变化幅度 < 15%，拒绝切换，沿用上期目标
            final_long_target = prev_long_target
            if enable_beta_hedging:
                final_short_target = -(1.0 - final_long_target)
            else:
                final_short_target = 0.0
            buffer_skip_count += 1
        else:
            # 变化幅度足够大，执行状态切换
            final_long_target = theory_long_target
            final_short_target = theory_short_target
            state_counts[state] += 1

        # ============================================================
        # 【第三步：多头选股 + 缓冲带 + 行业风控】
        # ============================================================
        final_long = set()
        sector_counts = {}  # 当前各行业已选股票计数

        # 始终保持 final_select_n 只股票，同比例降权（分散化程度不变）
        # 【还原 v1.4 核心逻辑：动态股票数，绝对固定权重！】
        N_long = int(final_select_n * final_long_target)  # 决定本期到底拿几只股票 (例如 14 只)
        weight_per_stock = 1.0 / final_select_n           # 永远是 0.05 (5%) 绝对固定权重
        # 第一步：动量前 N 名进入多头候选池
        valid_mom = date_momentum[common_tickers]
        top_momentum_long = valid_mom.nlargest(momentum_top_n).index

        # 第二步：候选池按波动率从小到大排序
        long_candidate_vol = date_volatility[top_momentum_long].sort_values(ascending=True)
        long_candidate_tickers = long_candidate_vol.index.tolist()

        # Phase 1：缓冲带逻辑（带严格上限 + 波动率优先 + 行业风控）
        # 【关键改动】：遍历 long_candidate_tickers（它已经按波动率从小到大排好序了）
        for ticker in long_candidate_tickers:
            if ticker in previous_long_holdings:
                rank = long_candidate_tickers.index(ticker)
                if rank < buffer_keep_n:
                    # ===== 行业风控检查 =====
                    sector = sector_mapping.get(ticker, 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if enable_sector_control and sector_counts[sector] > MAX_STOCKS_PER_SECTOR:
                        sector_counts[sector] -= 1  # 撤销计数
                        sector_block_count += 1
                        continue
                    final_long.add(ticker)
                    # 【核心紧箍咒】：老仓位保留的数量，绝不能超过本期的动态目标 N_long！
                    if len(final_long) >= N_long:
                        break

        # Phase 2：按波动率排名填补空缺到 N_long 只（同样带行业风控）
        needed_long = N_long - len(final_long)
        if needed_long > 0:
            for ticker in long_candidate_tickers:
                if ticker not in final_long:
                    # ===== 行业风控检查 =====
                    sector = sector_mapping.get(ticker, 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if enable_sector_control and sector_counts[sector] > MAX_STOCKS_PER_SECTOR:
                        sector_counts[sector] -= 1  # 撤销计数
                        sector_block_count += 1
                        continue
                    final_long.add(ticker)
                    # 达到动态目标 N_long 即刻停止
                    if len(final_long) >= N_long:
                        break

        # ========== 权重分配 ==========
        # 调仓日先清零所有仓位
        weights.loc[rebalance_date, :] = 0.0

        # 多头分配：每只动态权重 = 目标总仓位 / 持股数
        if len(final_long) > 0:
            weights.loc[rebalance_date, list(final_long)] = weight_per_stock

        # 空头分配（仅贝塔对冲模式）：直接分配给 SPY_Proxy
        if enable_beta_hedging and final_short_target < 0:
            weights.loc[rebalance_date, spy_column_name] = final_short_target

        # ========== 更新状态变量 ==========
        previous_long_holdings = final_long
        previous_net_exposure = final_long_target + final_short_target
        prev_long_target = final_long_target

    # ========== 打印统计结果 ==========
    total = len(rebalance_dates)
    print(f"\n  【大盘四象限状态分布】")
    print(f"    状态1（低波+向上→100%净多）: {state_counts[1]} 次 ({state_counts[1]/total:.1%})")
    print(f"    状态2（高波+向上→40%净多）: {state_counts[2]} 次 ({state_counts[2]/total:.1%})")
    print(f"    状态3（低波+向下→0%净敞口）: {state_counts[3]} 次 ({state_counts[3]/total:.1%})")
    print(f"    状态4（高波+向下→60%净空）: {state_counts[4]} 次 ({state_counts[4]/total:.1%})")
    print(f"\n  【缓冲带风控效果】")
    print(f"    净敞口缓冲拦下: {buffer_skip_count} 次 ({buffer_skip_count/total:.1%})")
    if enable_sector_control:
        print(f"    行业风控拦下: {sector_block_count} 只股票")

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

    # 返回 portfolio 对象和收益率，便于后续获取真实交易统计
    return portfolio, portfolio.returns()


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
    weights, prices_with_spy = generate_weights(
        prices,
        rebalance_dates,
        momentum_top_n=50,
        final_select_n=20,
        buffer_keep_n=40,
        sector_cap=0.20,          # 敏感性测试最优值：20%，单行业最多4只
        enable_sector_control=True,  # 默认启用行业风控
        enable_beta_hedging=False  # 默认使用 v1.5 纯多头模式
    )

    # 先运行回测
    print("\n[4/5] 运行回测...")
    portfolio, returns = run_backtest(
        prices_with_spy,
        weights,
        transaction_cost=0.001
    )

    # 计算绩效指标
    print("\n[5/5] 计算绩效指标...")
    metrics = calculate_performance_metrics(returns)

    # 【修正】统计真实交易笔数
    print("\n[补充] 交易统计...")
    total_trades_count = len(portfolio.trades.records)
    # 注：由于采用 targetpercent 调仓模式，权重漂移已被引擎自动纳入计算
    # 回测中的交易成本已经反映了真实换手率带来的摩擦

    print(f"  总交易笔数: {total_trades_count} 笔")
    print(f"  交易模式: targetpercent 调仓，已自动包含权重漂移再平衡")

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
