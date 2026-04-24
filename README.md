# 📈 美股动量波动率复合策略研究

一个基于 S&P 500 成分股的量化动量策略研究项目。

## 📁 项目结构

```
low-frequency-quant/
├── data/                          # 数据目录（git 忽略）
│   └── sp500_adjusted_close.parquet  # S&P 500 成分股日收盘价
│
├── fetch_sp500_data.py            # 数据获取脚本
├── factor_analyzer.py             # 因子分析脚本
├── backtest_engine.py             # 回测引擎（核心）
│
├── STRATEGY_PERFORMANCE_TRACKER.md  # 📊 策略表现跟踪器（重要！）
└── README.md                      # 本文件
```

## 🎯 策略逻辑

### 核心思想
结合**动量因子**和**波动率因子**的复合策略：
1. **动量筛选**：选出过去 12 个月（剔除最近 1 个月）表现最好的前 50 只股票
2. **波动率筛选**：从中选出 20 只波动率最低的股票（实现低波动增强）
3. **权重分配**：
   - v1.0：等权重（Equal Weight）
   - v1.1：波动率倒数加权（Inverse Volatility Weighting）
4. **调仓频率**：每月第一个交易日

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv env
source env/bin/activate

# 安装依赖
pip install pandas numpy vectorbt yfinance requests beautifulsoup4 pyarrow
```

### 2. 获取数据
```bash
python fetch_sp500_data.py
```

### 3. 运行回测
```bash
python backtest_engine.py
```

## 📊 当前策略表现（v1.0 基准）

| 指标 | 数值 |
|------|------|
| **总收益率** | 539.61% |
| **年化收益率** | 13.18% |
| **年化波动率** | 18.15% |
| **夏普比率** | 0.73 |
| **最大回撤** | -40.51% |
| **月胜率** | 55.80% |

> **详细的版本对比和优化记录请查看 [STRATEGY_PERFORMANCE_TRACKER.md](./STRATEGY_PERFORMANCE_TRACKER.md)**

## 🎯 优化路线图

- [x] v1.0：基准策略（等权重）
- [x] v1.1：波动率倒数加权（Inverse Volatility Weighting）
- [ ] 动量/波动率综合排名
- [ ] 降低调仓频率测试
- [ ] 大盘趋势过滤
- [ ] 行业中性化
- [ ] 个股止损规则

## 📝 研究文档

- **[STRATEGY_PERFORMANCE_TRACKER.md](./STRATEGY_PERFORMANCE_TRACKER.md)** - 策略版本迭代记录，所有优化的绩效对比都在这里！

## 🤝 关于

这是一个量化策略研究项目，用于学习和探索多因子选股策略。

> ⚠️ **免责声明**：本项目仅供学习和研究使用，不构成任何投资建议。
