#!/usr/bin/env python3
"""
=========================================================================
标普 500 Point-in-Time (PiT) 动态成分股矩阵构建器
=========================================================================
【核心功能】
  从维基百科抓取 S&P 500 历史成分股变更记录，通过逆向推演构建
  精确的日频/月频成分股布尔矩阵，彻底消除回测中的幸存者偏差。

【算法原理】
  1. 以当前 (2026年) S&P 500 成分股为基准（存活者集合）
  2. 抓取维基百科所有历史变更事件 (Date, Added, Removed)
  3. 从今天倒推回 15 年前：
     - 遇到 Added 事件 → 从当天集合中删除该股票（因为它是当天才加入的）
     - 遇到 Removed 事件 → 向当天集合中添加该股票（因为它是当天才被删除的）

【输出文件】
  sp500_pit_mask.parquet - 月频成分股布尔矩阵 (FFill)
  索引: 日期 (月末), 列: 股票代码, 值: True/False (是否属于 S&P 500)

【参考文献】
  https://en.wikipedia.org/wiki/List_of_S%26P_500_constituent_changes
=========================================================================
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# 模拟浏览器请求头，避免 403 Forbidden
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

def read_html_with_user_agent(url):
    """使用自定义 User-Agent 读取 HTML 表格，避免 403 错误"""
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req) as response:
        html_content = response.read()
    return pd.read_html(html_content)

# =========================================================================
# 配置参数
# =========================================================================
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CURRENT_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START_DATE = pd.Timestamp('2011-01-01')  # 回测起始日期前推
END_DATE = pd.Timestamp('2026-04-30')     # 当前日期

# =========================================================================
# Step 1: 抓取当前 S&P 500 成分股列表（作为逆向推演的起点）
# =========================================================================
def fetch_current_constituents():
    """抓取当前 S&P 500 成分股列表"""
    print("=" * 80)
    print("[Step 1/4] 抓取 S&P 500 当前成分股列表...")
    print("=" * 80)

    try:
        tables = read_html_with_user_agent(CURRENT_SP500_URL)
        # 第一个表格通常是成分股列表
        df = tables[0]
        print(f"  找到 {len(df)} 只当前成分股")

        # 尝试识别股票代码列（维基百科的列名可能有变化）
        possible_cols = ['Symbol', 'Ticker', 'Ticker symbol', '股票代码']
        ticker_col = None
        for col in possible_cols:
            if col in df.columns:
                ticker_col = col
                break

        if ticker_col is None:
            print(f"  ⚠️  无法识别股票代码列，列名如下: {df.columns.tolist()}")
            # 尝试猜测：列名包含 'symbol' 或 'ticker'
            for col in df.columns:
                if 'symbol' in col.lower() or 'ticker' in col.lower():
                    ticker_col = col
                    break

        if ticker_col is None:
            raise ValueError("无法找到股票代码列")

        tickers = df[ticker_col].str.strip().unique().tolist()
        print(f"  成功提取 {len(tickers)} 只股票代码")
        print(f"  前10只: {tickers[:10]}")

        return set(tickers)

    except Exception as e:
        print(f"  ❌ 抓取失败: {e}")
        print(f"  尝试使用备份方案...")
        # 备份：使用我们的价格数据中列名作为当前成分股的近似
        prices = pd.read_parquet('sp500_adjusted_close.parquet')
        tickers = prices.columns.tolist()
        print(f"  使用价格数据作为备份，共 {len(tickers)} 只股票")
        return set(tickers)

# =========================================================================
# Step 2: 抓取历史成分股变更记录
# =========================================================================
def fetch_constituent_changes():
    """抓取维基百科 S&P 500 成分股变更历史"""
    print("\n" + "=" * 80)
    print("[Step 2/4] 抓取 S&P 500 历史成分股变更记录...")
    print("=" * 80)

    try:
        tables = read_html_with_user_agent(WIKIPEDIA_URL)
        print(f"  维基百科页面共找到 {len(tables)} 个表格")

        # 找出包含变更记录的表格（通常有 'Date', 'Added', 'Removed' 列）
        change_tables = []
        for i, table in enumerate(tables):
            # 检查列名是否包含关键词
            col_str = ' '.join(str(c).lower() for c in table.columns)
            if ('date' in col_str and
                ('add' in col_str or 'ticker' in col_str) and
                ('remov' in col_str or 'delete' in col_str)):
                print(f"  ✅ 表格 #{i}: 疑似变更表格，形状 {table.shape}")
                change_tables.append(table)

        if not change_tables:
            print(f"  ⚠️  未找到标准变更表格，尝试所有包含 'Date' 列的表格")
            for i, table in enumerate(tables):
                if len(table) > 10:  # 至少有一些数据
                    for col in table.columns:
                        if 'date' in str(col).lower():
                            print(f"  ✅ 表格 #{i}: 包含 Date 列，形状 {table.shape}")
                            change_tables.append(table)
                            break

        # 合并所有变更表格
        all_changes = []
        for table in change_tables:
            cleaned = clean_change_table(table)
            if cleaned is not None and len(cleaned) > 0:
                all_changes.append(cleaned)

        if not all_changes:
            raise ValueError("未能成功提取任何变更记录")

        # 合并并按日期排序
        changes_df = pd.concat(all_changes, ignore_index=True)
        changes_df = changes_df.sort_values('Date').reset_index(drop=True)

        print(f"\n  ✅ 成功提取 {len(changes_df)} 条变更记录")
        print(f"  时间范围: {changes_df['Date'].min()} 至 {changes_df['Date'].max()}")
        print(f"\n  前5条记录预览:")
        print(changes_df.head().to_string())

        return changes_df

    except Exception as e:
        print(f"  ❌ 抓取变更记录失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_change_table(table):
    """清洗单个变更表格"""
    try:
        df = table.copy()

        # 处理多层列名（维基百科表格常见格式）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns]
        else:
            df.columns = [str(c).strip() for c in df.columns]

        # 找出 Date 列
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'Date' in col:
                date_col = col
                break

        if date_col is None:
            return None

        # 找出 Added 列
        added_col = None
        for col in df.columns:
            cl = col.lower()
            if ('add' in cl or 'Added_' in col) and 'remove' not in cl:
                added_col = col
                break

        # 找出 Removed 列
        removed_col = None
        for col in df.columns:
            cl = col.lower()
            if 'remov' in cl or 'Removed_' in col or 'delete' in cl or 'drop' in cl:
                removed_col = col
                break

        if added_col is None and removed_col is None:
            return None

        # 只保留我们需要的列
        keep_cols = []
        col_mapping = {}
        if date_col:
            keep_cols.append(date_col)
            col_mapping[date_col] = 'Date'
        if added_col:
            keep_cols.append(added_col)
            col_mapping[added_col] = 'Added'
        if removed_col:
            keep_cols.append(removed_col)
            col_mapping[removed_col] = 'Removed'

        df = df[keep_cols].copy()
        df = df.rename(columns=col_mapping)

        # 解析日期
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # 清洗股票代码
        if 'Added' in df.columns:
            df['Added'] = df['Added'].astype(str).str.strip()
            df.loc[df['Added'].str.lower().isin(['nan', 'none', '']), 'Added'] = None
            # 移除备注（括号中的内容）
            df['Added'] = df['Added'].str.replace(r'\s*\([^)]*\)', '', regex=True)

        if 'Removed' in df.columns:
            df['Removed'] = df['Removed'].astype(str).str.strip()
            df.loc[df['Removed'].str.lower().isin(['nan', 'none', '']), 'Removed'] = None
            df['Removed'] = df['Removed'].str.replace(r'\s*\([^)]*\)', '', regex=True)

        # 移除无效行（没有任何变更信息）
        has_added = 'Added' in df.columns and df['Added'].notna().any()
        has_removed = 'Removed' in df.columns and df['Removed'].notna().any()
        if not has_added and not has_removed:
            return None

        df = df.sort_values('Date').reset_index(drop=True)
        return df

    except Exception as e:
        print(f"    ⚠️  清洗表格时出错: {e}")
        return None

# =========================================================================
# Step 3: 逆向推演构建历史成分股矩阵
# =========================================================================
def build_historical_mask(current_tickers, changes_df):
    """逆向推演构建 Point-in-Time 成分股矩阵"""
    print("\n" + "=" * 80)
    print("[Step 3/4] 逆向推演构建历史成分股矩阵...")
    print("=" * 80)

    # 收集所有出现过的股票代码
    all_tickers = set(current_tickers)
    if 'Added' in changes_df.columns:
        all_tickers.update(changes_df['Added'].dropna().unique())
    if 'Removed' in changes_df.columns:
        all_tickers.update(changes_df['Removed'].dropna().unique())

    all_tickers = sorted([t for t in all_tickers if t and str(t).lower() != 'nan'])
    print(f"  历史上共出现 {len(all_tickers)} 只 S&P 500 股票")

    # 创建完整的日期范围（从 START_DATE 到 END_DATE）
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

    # 初始化成分股矩阵
    mask_df = pd.DataFrame(False, index=dates, columns=all_tickers)

    # =========================================================================
    # 正确的逆向推演算法：
    #
    # 时间轴: 过去 <========================================= 现在
    #          START_DATE       change_date    change_date    END_DATE
    #
    # 推演逻辑:
    #   1. 从 END_DATE 开始，已知 current_tickers 是当前成分股
    #   2. 按时间倒序处理变更（从现在向过去走）
    #   3. 遇到 Added 事件: 该股票是 change_date 才加入的，
    #                       所以 change_date 之前它不在指数里
    #   4. 遇到 Removed 事件: 该股票是 change_date 才被删除的，
    #                         所以 change_date 之前它还在指数里
    # =========================================================================

    # 当前状态集合（END_DATE 时的状态）
    current_set = set(current_tickers)

    # 按倒序处理变更事件日期（从最新到最老）
    changes_grouped = changes_df.groupby('Date')
    unique_dates = sorted(changes_grouped.groups.keys(), reverse=True)

    # 初始化：END_DATE 及之后（但我们的数据到 END_DATE 结束）
    last_date = END_DATE

    print(f"\n  开始倒推处理 {len(unique_dates)} 个变更日...")

    for i, change_date in enumerate(unique_dates):
        if change_date < START_DATE:
            continue

        # 显示进度
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    处理进度: {i+1}/{len(unique_dates)} 个变更日 (当前: {change_date.date()})")

        # Step 1: 设置 [change_date, last_date] 区间的状态
        # （因为还没处理 change_date 的变更，所以 current_set 是 change_date 之后的状态）
        date_slice = mask_df.index[(mask_df.index >= change_date) & (mask_df.index <= last_date)]
        if len(date_slice) > 0:
            mask_df.loc[date_slice, list(current_set)] = True

        # Step 2: 应用变更（倒推）- 更新 current_set 为 change_date 之前的状态
        day_changes = changes_grouped.get_group(change_date)
        for _, row in day_changes.iterrows():
            # Added 事件：该股票是 change_date 才加入的，之前没有
            if 'Added' in row and pd.notna(row['Added']):
                ticker = row['Added']
                current_set.discard(ticker)  # 移除：变更日之前它不在指数

            # Removed 事件：该股票是 change_date 才被删除的，之前有
            if 'Removed' in row and pd.notna(row['Removed']):
                ticker = row['Removed']
                current_set.add(ticker)  # 添加：变更日之前它还在指数

        last_date = change_date - pd.Timedelta(days=1)

    # 处理 START_DATE 到第一个变更日（或最后一个变更日到开始日）之间的区间
    if last_date >= START_DATE:
        date_slice = mask_df.index[(mask_df.index >= START_DATE) & (mask_df.index <= last_date)]
        if len(date_slice) > 0:
            valid_tickers = [t for t in current_set if t in mask_df.columns]
            mask_df.loc[date_slice, valid_tickers] = True

    print(f"\n  ✅ 历史成分股矩阵构建完成")
    print(f"  矩阵形状: {mask_df.shape}")

    # 统计每月股票数量（验证合理性）
    monthly_count = mask_df.resample('ME').sum().sum(axis=1)
    print(f"\n  历史成分股数量统计:")
    print(f"    最小值: {monthly_count.min():.0f} 只")
    print(f"    最大值: {monthly_count.max():.0f} 只")
    print(f"    平均值: {monthly_count.mean():.0f} 只")
    print(f"    最新值: {monthly_count.iloc[-1]:.0f} 只")

    # 合理性检查：应该在 ~500 只左右
    if monthly_count.mean() < 400 or monthly_count.mean() > 600:
        print(f"  ⚠️  警告: 平均成分股数量异常，可能数据有问题")

    return mask_df

# =========================================================================
# Step 4: 重采样为月度频率并保存
# =========================================================================
def resample_and_save(mask_df):
    """重采样为月度频率并保存"""
    print("\n" + "=" * 80)
    print("[Step 4/4] 重采样为月度频率并保存...")
    print("=" * 80)

    # 使用月末频率，向前填充（保证调仓日是正确的）
    mask_monthly = mask_df.resample('ME').last().ffill()

    print(f"  原始日频矩阵: {mask_df.shape}")
    print(f"  月度矩阵: {mask_monthly.shape}")
    print(f"  月度日期范围: {mask_monthly.index[0].date()} 至 {mask_monthly.index[-1].date()}")

    # 保存
    output_file = 'sp500_pit_mask.parquet'
    mask_monthly.to_parquet(output_file)
    print(f"\n  ✅ 文件已保存: {output_file}")

    # 额外保存日频矩阵作为备份（可选）
    daily_file = 'sp500_pit_mask_daily.parquet'
    mask_df.to_parquet(daily_file)
    print(f"  ✅ 日频备份已保存: {daily_file}")

    # 生成一份统计报告
    monthly_count = mask_monthly.sum(axis=1)
    print(f"\n{'='*80}")
    print("成分股数量统计（月度）")
    print(f"{'='*80}")
    print(monthly_count.describe().to_string())

    return mask_monthly

# =========================================================================
# 主函数
# =========================================================================
def main():
    print("\n" + "=" * 80)
    print("S&P 500 Point-in-Time 动态成分股矩阵构建器")
    print("=" * 80)
    print(f"  时间范围: {START_DATE.date()} 至 {END_DATE.date()}")
    print(f"  数据源: 维基百科")
    print("=" * 80 + "\n")

    # Step 1: 获取当前成分股
    current_tickers = fetch_current_constituents()

    # Step 2: 获取历史变更记录
    changes_df = fetch_constituent_changes()
    if changes_df is None or len(changes_df) == 0:
        print("\n❌ 无法获取变更记录，程序退出")
        return

    # Step 3: 构建历史成分股矩阵
    mask_df = build_historical_mask(current_tickers, changes_df)

    # Step 4: 重采样并保存
    mask_monthly = resample_and_save(mask_df)

    print("\n" + "=" * 80)
    print("✅ 全部任务完成！")
    print("=" * 80)
    print("\n输出文件:")
    print("  1. sp500_pit_mask.parquet       - 月频成分股布尔矩阵 (FFill)")
    print("  2. sp500_pit_mask_daily.parquet - 日频成分股布尔矩阵 (备份)")
    print("\n使用方法:")
    print("  - 回测中，对于每个调仓日，只选择 mask_monthly.loc[date] == True 的股票")
    print("  - 与现有价格数据取交集，确保只使用当时真实属于 S&P 500 的股票")
    print("=" * 80)

    return mask_monthly

if __name__ == "__main__":
    mask_monthly = main()
