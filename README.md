# Behavioral-Finance-Factor

一个面向 A 股的行为金融因子研究资料与计算 Notebook。项目围绕投资者注意力、羊群效应、参考点、前景理论、分析师锚定和注意力溢出等机制，提供 27 个指标的公式说明、通俗解读，以及相应的米筐数据计算示例。

> 本目录用于研究与学习，不构成投资建议。因子的经济解释不等于可交易结论，所有信号均须在明确的股票池、可得数据时点和交易成本假设下重新检验。

## 文件说明

```text
Behavioral-Finance-Factor/
├── README.md                         # 本说明
├── 行为金融因子.ipynb                 # 因子计算与示例调用
├── 行为金融因子.md                    # 27 个指标的公式、变量定义和逻辑
└── 行为金融因子_通俗解读与故事版.md    # 同一组指标的直觉化解读
```

建议按以下顺序阅读：

1. 先看 [行为金融因子_通俗解读与故事版.md](行为金融因子_通俗解读与故事版.md)，建立因子所描述的投资者行为直觉；
2. 再看 [行为金融因子.md](行为金融因子.md)，确认公式、输入数据和方向解释；
3. 最后运行 `行为金融因子.ipynb`，按需调用对应的计算函数。

## 快速开始

Notebook 使用 `numpy`、`pandas` 和米筐 `rqdatac`。其中的安装单元使用过 `rqdatac==3.4.2`。

```bash
python -m pip install numpy pandas "rqdatac==3.4.2"
jupyter notebook 行为金融因子.ipynb
```

运行前需要在当前环境完成米筐数据服务的初始化与授权。打开 Notebook 后，请按章节从上到下运行：开头的辅助函数会被后续多个因子复用；每个因子章节都在计算函数后给出示例参数和结果预览。

第一次尝试建议缩短时间区间、使用较小股票池，以先确认数据权限、字段名称、输出形状和运行时间。

## 已实现的计算主题

Notebook 以 `calculate_*` 函数组织计算逻辑。下表列出其中的主要入口；函数的完整参数、默认值和计算细节以 Notebook 中的 docstring 为准。

| 主题 | 代表性函数 | 所刻画的行为机制 |
| --- | --- | --- |
| 异常收益与交易注意力 | `calculate_abnretd_daily`、`calculate_abnretavg_daily`、`calculate_abnvold_daily`、`calculate_abnvolavg_daily`、`calculate_attn_daily`、`calculate_turnavg_daily` | 极端收益、异常成交量和交易活跃度带来的有限注意力 |
| 市场与局部羊群效应 | `calculate_cssd_beta_factor`、`calculate_csad_beta_factor`、`calculate_small_board_csad`、`calculate_center_board_csad`、`calculate_center_board_csad_ratio_daily` | 市场或相似股票群体在极端行情中的收益趋同性 |
| 参考点与前景理论 | `calculate_cgo_daily`、`calculate_tk_daily`、`calculate_pb_tk_daily` | 浮盈浮亏、损失厌恶与风险偏好的状态依赖 |
| 预期偏差与溢出 | `calculate_caf_ep_daily`、`calculate_spill_daily`、`calculate_turnover_spill_daily` | 分析师锚定、行业内热点扩散及注意力传导 |
| 关注事件与机构行为 | `calculate_attention_events_daily`、`calculate_lsv_daily` | 涨跌停 / 龙虎榜事件与基金交易的同步性 |
| 回归型注意力捕捉 | `calculate_indus_abnd_attention_daily`、`calculate_indus_vol_attention_daily`、`calculate_mkt_turnover_attention_daily` | 个股收益对行业异常收益、行业波动和市场成交额的敏感度 |

文档还收录了个人与机构交易热度、SLSV 等指标的定义。是否能够直接计算，取决于相应的微观交易或基金持仓数据是否可用。

## 典型因子：ABNRETD

最大异常日收益率（ABNRETD）衡量一段窗口内，个股日收益与市场日收益的最大绝对偏离：

$$
\mathrm{ABNRETD}_{i,T} =
\max_{t=T-m+1}^{T}\left|r_{i,t}-\bar r_t\right|
$$

其中 $r_{i,t}$ 是个股日收益，$\bar r_t$ 是市场日收益，$m$ 通常取一个月左右的交易日窗口。因子值越高，表示个股曾出现更显著的异常日波动，常被视为投资者注意力的代理变量。

Notebook 中的 `calculate_abnretd_daily` 使用两步计算：先根据全市场有效股票的日收益构造市场等权收益，再按股票计算滚动窗口内的最大绝对异常收益。示例默认使用 20 日窗口，并返回索引为 `(date, order_book_id)`、列名为 `ABNRETD` 的面板数据。

ABNRETD 本身不规定做多或做空方向。若研究“高关注度导致短期溢价、之后反转”的假设，应以未来收益、分组表现和可交易约束验证低因子组与高因子组的相对表现。

## 数据需求

不同模块从米筐读取的字段和接口并不相同：

| 数据 | 主要用途 |
| --- | --- |
| 日度收盘价、成交量、成交额、交易日历 | 收益率、异常收益、异常成交量、注意力事件 |
| 换手率 | CGO、ATTN、TURNAVG 与换手率注意力溢出 |
| 市值、账面市值比、无风险利率 | 市场 / 规模 / 价值 / 动量控制变量 |
| 行业分类 | 行业内注意力、行业波动与溢出效应 |
| 分析师一致预期 PE | CAF_EP；Notebook 中该章节标注为付费接口 |
| 基金持仓与披露日期 | LSV；Notebook 中该章节标注为付费接口 |

所有数据都应按当时可得口径使用。尤其应处理停牌、复权、上市时间、成分股历史变化和财报 / 持仓披露滞后，避免把事后数据带入信号日。

## 输出与使用方式

多数个股因子返回两层索引的 `pandas.DataFrame`：

```text
index:   (date, order_book_id)
column:  对应因子名称
```

CSSD 与基础 CSAD 一类的市场状态模型则返回日期索引的时间序列。具体列名会直接体现在相应函数的 docstring 和示例单元中。将因子用于截面检验前，建议至少完成以下检查：

- 信号只使用信号日及之前能够观察到的数据；
- 同一交易日的股票池与行业归属符合历史时点；
- 剔除或单独处理停牌、涨跌停、极端复权和缺失观测；
- 对因子做必要的去极值、标准化和行业 / 市值暴露检验；
- 使用样本外时期、调仓频率、换手率和交易成本评估稳健性。

## 性能与权限提示

- 全 A 股长样本的数据下载、逐日滚动回归和行业分组会消耗较多时间与内存。
- “小型板块”CSAD 需要构造股票间相关性；股票池越大，计算量增长越快。
- 回归型注意力因子需要同时构建市场、规模、价值与动量控制变量，较适合先在短样本上验证。
- 分析师预期和基金持仓章节依赖付费或权限受限接口；无权限时，其余仅依赖行情的数据模块仍可单独研究。

## 许可

本目录未包含单独的开源许可证。分发、商用或引用其中内容前，请先向项目维护者确认授权，并清楚记录数据来源、股票池、样本期和参数口径。
