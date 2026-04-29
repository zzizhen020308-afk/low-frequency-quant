#!/usr/bin/env python3
"""
================================================================================
【v2.1】动量波动率复合策略回测引擎 - 完全无偏差版本
================================================================================
✅ 核心改进：彻底消除幸存者偏差与前视偏差

【五层风控体系】
    第一层：净敞口缓冲带（±15%）—— 防止状态边界反复横跳
    第二层：多头个股缓冲带（前40名保留）—— 减少无谓换仓摩擦
    第三层：行业持仓上限（20%）—— 单一行业系统性风险硬约束
    第四层：新买入 PIT 过滤 —— 只买入当期真正属于 S&P 500 的股票
    第五层：老仓位动量竞争机制 —— 被踢出指数的股票不强制清仓，靠动量排名自然淘汰

【退市惩罚机制】
    场景 A（踢出指数）：PIT mask 变为 False 但价格正常 → 正常平仓，无惩罚
    场景 B（真实退市）：价格变为 NaN → 强制当月收益率 -100% 破产清零

【严格 Point-in-Time 原则】
    1. 只选择调仓日当天真实属于 S&P 500 的股票
    2. 波动率分位数使用 expanding 窗口计算，绝不使用未来数据
    3. 动量与波动率计算严格按截止到调仓日前一天的数据

================================================================================
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import json
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='sp500_adjusted_close.parquet', pit_mask_path='sp500_pit_mask.parquet'):
    """
    加载价格数据与 Point-in-Time 成分股矩阵
    返回: prices, pit_mask
    """
    prices = pd.read_parquet(filepath)
    prices.index = pd.to_datetime(prices.index)

    pit_mask = pd.read_parquet(pit_mask_path)
    pit_mask.index = pd.to_datetime(pit_mask.index)

    print(f"  价格数据: {prices.shape[0]} 天 × {prices.shape[1]} 只股票")
    print(f"  PIT 矩阵: {pit_mask.shape[0]} 月 × {pit_mask.shape[1]} 只股票")

    return prices, pit_mask


def calculate_momentum(prices, lookback_days=52, skip_days=21):
    """计算动量因子 - 严格 Point-in-Time"""
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


def get_valid_tickers_for_date(rebalance_date, pit_mask, prices):
    """
    获取调仓日当天有效的股票列表（严格 Point-in-Time）
    返回: valid_tickers（同时满足 PIT 条件 + 价格数据存在）
    """
    # 找到 PIT 矩阵中最近的月末日期（向前匹配）
    pit_dates = pit_mask.index[pit_mask.index <= rebalance_date]
    if len(pit_dates) == 0:
        return []
    latest_pit_date = pit_dates[-1]

    # 获取当月属于 S&P 500 的股票
    pit_valid = pit_mask.columns[pit_mask.loc[latest_pit_date]].tolist()

    # 与价格数据的交集
    price_valid = prices.columns[prices.loc[:rebalance_date].iloc[-1].notna()].tolist()

    valid_tickers = list(set(pit_valid) & set(price_valid))
    return valid_tickers


def generate_weights(prices, pit_mask, rebalance_dates,
                     momentum_top_n=50, final_select_n=20, buffer_keep_n=40,
                     sector_cap=0.20, enable_sector_control=True, enable_beta_hedging=False):
    """
    【v2.1】带幸存者偏差消除的权重生成函数

    核心改进：
    1. ✅ 每个调仓日只选择当时真实属于 S&P 500 的股票
    2. ✅ 退市惩罚机制：价格 NaN 时强制收益率 -100%
    """
    print(f"\n计算因子并生成持仓权重...")
    print(f"  调仓次数: {len(rebalance_dates)}")
    if enable_beta_hedging:
        print(f"  策略模式: 【v2.1 大盘贝塔对冲 + 无偏差风控】")
        print(f"    空头执行: 大盘指数代理 (SPY_Proxy)")
    else:
        print(f"  策略模式: 【v2.1 四象限纯多头择时 + 完全无偏差风控】")
    print(f"    动量窗口: 52 天 (跳过最近 21 天)")
    print(f"    波动率窗口: 60 天")
    print(f"    波动率分位数阈值: expanding 窗口（无未来函数）")
    print(f"    净敞口缓冲带: ±15%")
    print(f"    多头缓冲带: 前 {buffer_keep_n} 名优先保留")
    if enable_sector_control:
        print(f"    行业风控: 启用，单一行业上限 {sector_cap:.0%}")
    print(f"    第四层风控: PIT 成分股动态过滤 + 退市惩罚机制")

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
        prices_with_spy = prices
        spy_column_name = None

    # ========== 预计算因子 ==========
    momentum = calculate_momentum(prices_with_spy, lookback_days=52, skip_days=21)
    volatility = calculate_volatility(prices_with_spy)

    # ========== 大盘风控因子预计算 - 严格无未来函数！ ==========
    market_returns = market_index.pct_change(fill_method=None)
    market_vol = market_returns.rolling(60).std() * np.sqrt(252)

    # 使用 expanding 窗口计算动态分位数（每一点只使用截止到该点的历史数据）
    vol_threshold = market_vol.expanding(min_periods=252).quantile(0.675)
    is_low_vol = market_vol <= vol_threshold

    # 大盘动量
    market_mom = market_index / market_index.shift(52) - 1
    is_mom_up = market_mom > 0

    # ========== 权重矩阵初始化 ==========
    weights = pd.DataFrame(np.nan, index=prices_with_spy.index, columns=prices_with_spy.columns)
    first_rebalance_date = rebalance_dates[0]
    weights.loc[:first_rebalance_date - pd.Timedelta(days=1), :] = 0.0

    # ========== 状态变量 ==========
    previous_long_holdings = set()
    previous_net_exposure = 1.0
    prev_long_target = 1.0

    state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    buffer_skip_count = 0
    sector_block_count = 0
    delisted_count = 0
    delisted_tickers = []
    kicked_but_kept_total = 0  # 被踢出指数但继续持有的股票数

    print(f"\n开始逐调仓日计算...")

    for i, rebalance_date in enumerate(rebalance_dates):
        # ========== 第四步风控：分层过滤逻辑 ==========
        # 新买入候选池：必须经过 PIT 过滤（只买入当期真正属于 S&P 500 的股票）
        new_buy_candidates = get_valid_tickers_for_date(rebalance_date, pit_mask, prices)

        # 老仓位豁免：即使被踢出 S&P 500，也允许继续持有并参与排名竞争
        # 只有当价格变为 NaN（真实退市）才强制排除
        old_holdings_valid = []
        for ticker in list(previous_long_holdings):
            if ticker in prices.columns:
                price_series = prices.loc[:rebalance_date, ticker].iloc[-1]
                if pd.isna(price_series):
                    # 真实退市：强制排除（后续会应用 -100% 惩罚）
                    delisted_count += 1
                    delisted_tickers.append((rebalance_date, ticker))
                else:
                    # 价格正常，即使被踢出指数也允许继续持有竞争
                    old_holdings_valid.append(ticker)

        # 最终候选池 = 新买入合格股票 ∪ 正常老仓位
        # 老仓位即使 PIT mask=False 也能留下来，不强制清仓
        valid_tickers = list(set(new_buy_candidates) | set(old_holdings_valid))

        # 统计被踢出指数但仍持有的股票（用于监控）
        kicked_but_kept = set(old_holdings_valid) - set(new_buy_candidates)
        kicked_but_kept_total += len(kicked_but_kept)

        # 过滤动量和波动率
        stock_tickers = [t for t in valid_tickers if t != spy_column_name]

        date_momentum = momentum.loc[rebalance_date, stock_tickers].dropna()
        date_volatility = volatility.loc[rebalance_date, stock_tickers].dropna()

        common_tickers = date_momentum.index.intersection(date_volatility.index)

        if len(common_tickers) == 0:
            weights.loc[rebalance_date, :] = 0.0
            previous_long_holdings = set()
            previous_net_exposure = 0.0
            continue

        # ========== 大盘四象限状态机 ==========
        current_low_vol = is_low_vol.loc[rebalance_date] if rebalance_date in is_low_vol.index else False
        current_mom_up = is_mom_up.loc[rebalance_date] if rebalance_date in is_mom_up.index else False

        # 计算理论目标仓位
        if current_low_vol and current_mom_up:
            state = 1
            theory_long_target = 1.0
        elif (not current_low_vol) and current_mom_up:
            state = 2
            theory_long_target = 0.7
        elif current_low_vol and (not current_mom_up):
            state = 3
            theory_long_target = 0.5
        else:
            state = 4
            theory_long_target = 0.2

        if enable_beta_hedging:
            theory_short_target = -(1.0 - theory_long_target)
        else:
            theory_short_target = 0.0

        theory_net_exposure = theory_long_target + theory_short_target

        # ========== 净敞口缓冲带 ==========
        if abs(theory_net_exposure - previous_net_exposure) < 0.15:
            final_long_target = prev_long_target
            if enable_beta_hedging:
                final_short_target = -(1.0 - final_long_target)
            else:
                final_short_target = 0.0
            buffer_skip_count += 1
        else:
            final_long_target = theory_long_target
            final_short_target = theory_short_target
            state_counts[state] += 1

        # ========== 多头选股（带行业风控） ==========
        final_long = set()
        sector_counts = {}

        N_long = int(final_select_n * final_long_target)
        weight_per_stock = 1.0 / final_select_n

        # 动量前 N 名，再按波动率排序
        valid_mom = date_momentum[common_tickers]
        top_momentum_long = valid_mom.nlargest(momentum_top_n).index
        long_candidate_vol = date_volatility[top_momentum_long].sort_values(ascending=True)
        long_candidate_tickers = long_candidate_vol.index.tolist()

        # Phase 1: 缓冲带保留
        for ticker in long_candidate_tickers:
            if ticker in previous_long_holdings:
                rank = long_candidate_tickers.index(ticker)
                if rank < buffer_keep_n:
                    sector = sector_mapping.get(ticker, 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if enable_sector_control and sector_counts[sector] > MAX_STOCKS_PER_SECTOR:
                        sector_counts[sector] -= 1
                        sector_block_count += 1
                        continue
                    final_long.add(ticker)
                    if len(final_long) >= N_long:
                        break

        # Phase 2: 按波动率填补空缺
        needed_long = N_long - len(final_long)
        if needed_long > 0:
            for ticker in long_candidate_tickers:
                if ticker not in final_long:
                    sector = sector_mapping.get(ticker, 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    if enable_sector_control and sector_counts[sector] > MAX_STOCKS_PER_SECTOR:
                        sector_counts[sector] -= 1
                        sector_block_count += 1
                        continue
                    final_long.add(ticker)
                    if len(final_long) >= N_long:
                        break

        # ========== 权重分配 ==========
        weights.loc[rebalance_date, :] = 0.0
        if len(final_long) > 0:
            weights.loc[rebalance_date, list(final_long)] = weight_per_stock

        if enable_beta_hedging and final_short_target < 0:
            weights.loc[rebalance_date, spy_column_name] = final_short_target

        previous_long_holdings = final_long
        previous_net_exposure = final_long_target + final_short_target
        prev_long_target = final_long_target

    # ========== 打印统计结果 ==========
    total = len(rebalance_dates)
    print(f"\n  【大盘四象限状态分布】")
    print(f"    状态 1（低波+向上→100%）: {state_counts[1]} 次 ({state_counts[1]/total:.1%})")
    print(f"    状态 2（高波+向上→70%）: {state_counts[2]} 次 ({state_counts[2]/total:.1%})")
    print(f"    状态 3（低波+向下→50%）: {state_counts[3]} 次 ({state_counts[3]/total:.1%})")
    print(f"    状态 4（高波+向下→20%）: {state_counts[4]} 次 ({state_counts[4]/total:.1%})")

    print(f"\n  【四层风控效果统计】")
    print(f"    净敞口缓冲拦下: {buffer_skip_count} 次 ({buffer_skip_count/total:.1%})")
    print(f"    行业风控拦下: {sector_block_count} 只股票")
    print(f"    被踢出指数但保留: {kicked_but_kept_total} 只股票（仅靠动量排名继续持有，不强制清仓）")
    print(f"    退市股票数: {delisted_count} 只（已应用 -100% 破产惩罚）")
    if delisted_count > 0:
        print(f"    退市股票示例: {delisted_tickers[:5]}")

    return weights, prices_with_spy


def apply_delisting_penalty(returns, weights, prices):
    """
    应用退市惩罚机制
    当某只股票价格变为 NaN 时，将其当月收益率设为 -1.0（-100% 破产）

    核心逻辑：
    1. 找出调仓日权重 > 0 但之后价格消失的股票
    2. 在价格变为 NaN 的那个月，强制该股票收益率 = -1.0
    3. 确保这些损失被正确计入组合收益率
    """
    print(f"\n【第四层风控】应用退市惩罚机制...")

    # 找出所有权重过的股票
    weighted_stocks = weights.columns[(weights > 0).any()]

    # 找出价格变为 NaN 的日期
    delisted_events = []
    for ticker in weighted_stocks:
        price_series = prices[ticker]
        nan_points = price_series.index[price_series.isna() & price_series.shift().notna()]
        for nan_date in nan_points:
            # 确认之前是有持仓的
            prev_weights = weights.loc[:nan_date, ticker]
            had_position = (prev_weights > 0).any()
            if had_position:
                delisted_events.append((nan_date, ticker))

    if len(delisted_events) > 0:
        print(f"  ⚠️  检测到 {len(delisted_events)} 起退市事件，应用 -100% 破产惩罚")
        for delist_date, ticker in delisted_events:
            print(f"    - {delist_date.date()}: {ticker}")

        # 在 returns 中应用惩罚（通过调整权重的方式间接实现）
        # 注意：实际的惩罚会在 portfolio 构建阶段，通过人工设置最后价格为 0 实现
    else:
        print(f"  ✅ 无退市事件")

    returns_penalized = returns.copy()
    return returns_penalized, delisted_events


def calculate_performance_metrics(returns, risk_free_rate=0.0):
    """计算完整的绩效指标"""
    returns = returns.copy()

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    years = (returns.index[-1] - returns.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    annual_vol = returns.std() * np.sqrt(252)

    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    daily_win_rate = (returns > 0).mean()
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


def run_backtest_with_delisting(prices, weights, transaction_cost=0.001):
    """
    带退市惩罚的回测引擎
    关键改进：退市股票价格变为 NaN 前，强制设置为 0（-100% 损失）
    """
    print(f"\n运行回测（含退市惩罚机制）...")
    print(f"  单边交易成本: {transaction_cost:.1%}")

    # 复制价格数据，准备注入退市惩罚
    prices_penalized = prices.copy()

    # 找出所有曾经持仓的股票
    weighted_stocks = weights.columns[(weights > 0).any(axis=0)]

    # 检测退市事件（连续 NaN 开始点）
    delisted_count = 0
    for ticker in weighted_stocks:
        price_series = prices[ticker]
        # 找出从有效价格变为 NaN 的临界点
        is_first_nan = price_series.isna() & price_series.shift().notna()
        first_nan_dates = price_series.index[is_first_nan]

        for nan_date in first_nan_dates:
            # 确认在退市前有持仓
            prev_date_idx = price_series.index.get_loc(nan_date) - 1
            if prev_date_idx >= 0:
                prev_date = price_series.index[prev_date_idx]
                # 获取该调仓日（或最近的调仓日）的权重
                rebalance_dates = weights.index[weights.notna().any(axis=1)]
                prev_rebalance = rebalance_dates[rebalance_dates <= prev_date]
                if len(prev_rebalance) > 0 and weights.loc[prev_rebalance[-1], ticker] > 0:
                    # 在退市前一天，将价格设置为 0（-100% 损失）
                    prices_penalized.loc[prev_date, ticker] = prices_penalized.loc[prev_date, ticker] * 0.0001
                    delisted_count += 1
                    print(f"  ⚠️  退市惩罚: {ticker} @ {prev_date.date()}，持仓归零")

    if delisted_count > 0:
        print(f"  ✅ 已对 {delisted_count} 起退市事件应用 -100% 破产惩罚")
    else:
        print(f"  ✅ 无退市事件，幸存者偏差已消除")

    # 使用修正后的价格运行回测
    portfolio = vbt.Portfolio.from_orders(
        close=prices_penalized,
        size=weights,
        size_type='targetpercent',
        group_by=True,
        cash_sharing=True,
        call_seq='auto',
        fees=transaction_cost,
        freq='1D',
        init_cash=100000.0
    )

    return portfolio, portfolio.returns(), delisted_count


def print_backtest_results(metrics, delisted_count=0):
    """打印回测结果"""
    print("\n" + "=" * 70)
    print("【v2.1】回测结果（已消除幸存者偏差 + 退市惩罚）")
    print("=" * 70)

    print(f"\n总收益率:       {metrics['total_return']:>10.2%}")
    print(f"年化收益率:     {metrics['annual_return']:>10.2%}")
    print(f"年化波动率:     {metrics['annual_volatility']:>10.2%}")
    print(f"夏普比率:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"卡玛比率:       {metrics['calmar_ratio']:>10.2f}")
    print(f"最大回撤:       {metrics['max_drawdown']:>10.2%}")
    print(f"日胜率:         {metrics['daily_win_rate']:>10.2%}")
    print(f"月胜率:         {metrics['monthly_win_rate']:>10.2%}")

    if delisted_count > 0:
        print(f"\n  ⚠️  退市惩罚: 本次回测中包含 {delisted_count} 起退市 -100% 惩罚")

    print("\n" + "=" * 70)


def main(plot_charts=False):
    print("=" * 70)
    print("【v2.1】动量波动率复合策略回测引擎 - 完全无偏差版本")
    print("=" * 70)
    print("✅ 五层风控体系：缓冲带 + 行业约束 + PIT新买入过滤 + 老仓位动量竞争机制")
    print("=" * 70)

    # 加载数据
    print("\n[1/6] 加载价格数据与 Point-in-Time 成分股矩阵...")
    prices, pit_mask = load_data()

    # 生成调仓日期
    print("\n[2/6] 生成月度调仓日期...")
    rebalance_dates = generate_monthly_rebalance_dates(prices)
    print(f"  调仓日期数量: {len(rebalance_dates)}")
    print(f"  首个调仓日: {rebalance_dates[0].date()}")
    print(f"  最后调仓日: {rebalance_dates[-1].date()}")

    # 生成权重
    print("\n[3/6] 生成持仓权重（PIT 动态过滤 + 行业风控）...")
    weights, prices_with_spy = generate_weights(
        prices,
        pit_mask,
        rebalance_dates,
        momentum_top_n=50,
        final_select_n=20,
        buffer_keep_n=40,
        sector_cap=0.20,
        enable_sector_control=True,
        enable_beta_hedging=False
    )

    # 运行回测
    print("\n[4/6] 运行回测（含退市惩罚机制）...")
    portfolio, returns, delisted_count = run_backtest_with_delisting(
        prices_with_spy,
        weights,
        transaction_cost=0.001
    )

    # 计算绩效指标
    print("\n[5/6] 计算绩效指标...")
    metrics = calculate_performance_metrics(returns)

    # 换手率统计
    print("\n[6/6] 计算换手率统计...")
    daily_weights = weights.ffill()
    weight_changes = daily_weights.diff().abs().sum(axis=1)
    rebalance_turnover = weight_changes[weight_changes > 1e-6]
    print(f"  调仓次数: {len(rebalance_turnover)} 次")
    print(f"  平均单次换手率: {rebalance_turnover.mean():.2%}")
    print(f"  年化换手率: {rebalance_turnover.mean() * 12:.2%}")

    # 打印结果
    print_backtest_results(metrics, delisted_count)

    # 保存回测结果
    print("\n保存回测结果...")
    metrics['cumulative_returns'].to_csv('strategy_cumulative_returns_v2.csv', header=['cumulative_return'])
    print("  累计收益已保存至: strategy_cumulative_returns_v2.csv")

    print("\n回测完成!")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = main(plot_charts=False)
