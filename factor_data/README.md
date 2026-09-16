# 行为金融因子：原始值批量计算包

本目录包含全部 22 个行为金融因子。每个脚本只保留原始因子值的计算与 CSV
保存，不包含画图、因子检验或回测代码。

输出为宽表：行是交易日，列是 order_book_id，第一列固定命名为 datetime。
默认计算区间为 2019-01-01 至 2026-08-31。

解压 ricequant_factor_data_20100101_20260910.zip 后，目录应包含：
data/fund_holdings、data/get_comp_indicators、
data/get_instrument_industry_monthly。

基金类、CAF_EP 与所有行业相关因子使用这份本地数据；其他日频行情数据由
RiceQuant 运行环境的 rqdatac 获取。
