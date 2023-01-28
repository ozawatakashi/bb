# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:13:17 2022

@author: ozawa
"""

import pandas as pd
import numpy as np
import os
import time
import glob
import pandas_datareader.data as web
import datetime
from dateutil.relativedelta import relativedelta
os.chdir("G:/マイドライブ/01_自己研鑽/03_財務分析/01_元データ/")

#%%
# =============================================================================
# 関数群
# =============================================================================

def make_YM_IDX(stock_sr, M = 2):
    """
    Parameters
    ----------
    stock_sr : 日付をindexで持った日次の終値Series
    M : int

    Returns
    -------
    stock : 月次（最終営業日ごと、Mカ月後）に変換した終値Series
    """
    stock = stock_sr.copy()
    stock["YM"] = [x[:-3] for x in stock.index.astype(str)]
    stock = stock.drop_duplicates(subset = "YM", keep = "last")
    stock = stock.dropna()
    stock["YM"] = pd.to_datetime(stock["YM"]).apply(lambda x: x + relativedelta(months = -M))
    stock = stock.set_index("YM")
    return stock
#%%
# =============================================================================
# 取得した財務データを結合する
# =============================================================================
years = glob.glob("20*")
list0_df = []
for year in years:
    list1_df = []
    for file in glob.glob(year + "/fy*.csv"):
        list1_df.append(pd.read_csv(file, index_col = 0).set_index(["コード", "年度"]))
    list0_df.append(pd.concat(list1_df, axis = 1))

df = pd.concat(list0_df, axis = 0)
df = df.dropna(axis = 0)
df = df.replace("-", "0")
df = df.astype(np.float32)
df = df.reset_index()
df["年度"] = pd.to_datetime(df["年度"])
df.to_csv("../03_加工データ/財務情報.csv", encoding = "CP932")
#%%
# =============================================================================
# 株価データの整形
# =============================================================================
df = pd.read_csv("../03_加工データ/財務情報.csv", encoding = "CP932", index_col = 0)
df_code = pd.read_csv("../03_加工データ/コード一覧.csv", encoding = "CP932")
list_code = list(df_code["コード"])

df_stock = pd.read_csv("../03_加工データ/株価終値.csv", encoding = "CP932")
df_stock_add = pd.read_csv("../03_加工データ/株価終値_追加分.csv", encoding = "CP932")
list_df_stock = []
for i in range(2, 4002):
    date = df_stock[str(i)]
    date = date[date != '--------']
    date = date.iloc[2:]
    date_add = df_stock_add[str(i)]
    date_add = date_add[date_add != '--------']
    date_add = date_add.iloc[2:]
    date = pd.concat([date, date_add])
    date = pd.to_datetime(date)
    
    if len(date.index.unique()) == 1: continue
    price = df_stock[str(i+0.5)]
    price = price[price != '--------']
    price = price.iloc[2:]
    price_add = df_stock_add[str(i+0.5)]
    price_add = price_add[price_add != '--------']
    price_add = price_add.iloc[2:]
    price = pd.concat([price, price_add])

    stock = pd.Series(price.values, index = date.values, name = list_code[i-2]).astype(np.float64)
    ind = stock.index.dropna()
    if len(ind) <= 1:continue
    stock = stock.dropna()
    list_df_stock.append(stock)

df_stock = pd.concat(list_df_stock, axis = 1).sort_index()
df_stock.to_csv("../03_加工データ/株価終値_整形後.csv", encoding = "CP932")

#%%
# =============================================================================
# 結合済みの財務データと株価終値をLEFT結合する
# 1年間(N週)の終値リターンと、直近N週営業日間の終値リターンを結合する
# =============================================================================
N = 26

df = pd.read_csv("../03_加工データ/財務情報.csv", encoding = "CP932", index_col = 0)
df_stock = pd.read_csv("../03_加工データ/株価終値_整形後.csv", encoding = "CP932", index_col = 0)

df_stock_percent_after = df_stock.pct_change(periods = N).shift(-N)
df_stock_percent_before = df_stock.pct_change(periods = N)
list_code = list(df_stock.columns)

df["年度"] = pd.to_datetime(df["年度"])
# df["年度"].apply(lambda x: x + relativedelta(months = 1, days = -1))
list_df = []
for i, code in enumerate(list_code):
    try:
        stock_future_ret = make_YM_IDX(df_stock_percent_after[[code]])
        stock_before_ret = make_YM_IDX(df_stock_percent_before[[code]])
        stock_one = make_YM_IDX(df_stock[[code]])
    except:
        print("\n" + str(code) + "なし")
        continue
    stock_future_ret.columns = ["future_ret"]
    stock_before_ret.columns = ["before_ret"]
    stock_one.columns = ["stock_price"]
    df_tmp = df.loc[df["コード"] == int(code), :].copy()
    df_tmp = df_tmp.merge(stock_future_ret, how = "left", left_on = "年度", right_index = True)
    df_tmp = df_tmp.merge(stock_one, how = "left", left_on = "年度", right_index = True)
    df_tmp = df_tmp.merge(stock_before_ret, how = "left", left_on = "年度", right_index = True)
    list_df.append(df_tmp)
    print("\r", int(100*i/len(list_code)),"% finish", end = "", sep = "")

df_main = pd.concat(list_df)
# df_main = df_main.dropna()
df_main.to_csv("../03_加工データ/分析用データ_half_year.csv", encoding = "CP932")

print(df_main["future_ret"].astype(np.float32).describe(percentiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))
df_main.loc[df_main["future_ret"] >= 10, "future_ret"] = 10
df_main.loc[df_main["future_ret"] <= -4, "future_ret"] = -4
df_main["future_ret"].hist(bins = 1000)

#%%
# =============================================================================
# 結合済みの財務データと株価終値をLEFT結合する
# 1年間(N週)の終値リターンと、直近N週営業日間の終値リターンを結合する
# =============================================================================
N = 52

df = pd.read_csv("../03_加工データ/財務情報.csv", encoding = "CP932", index_col = 0)
df_stock = pd.read_csv("../03_加工データ/株価終値_整形後.csv", encoding = "CP932", index_col = 0)

df_stock_percent_after = df_stock.pct_change(periods = N).shift(-N)
df_stock_percent_before = df_stock.pct_change(periods = N)
list_code = list(df_stock.columns)

df["年度"] = pd.to_datetime(df["年度"])
# df["年度"].apply(lambda x: x + relativedelta(months = 1, days = -1))
list_df = []
for i, code in enumerate(list_code):
    try:
        stock_future_ret = make_YM_IDX(df_stock_percent_after[[code]])
        stock_before_ret = make_YM_IDX(df_stock_percent_before[[code]])
        stock_one = make_YM_IDX(df_stock[[code]])
    except:
        print("\n" + str(code) + "なし")
        continue
    stock_future_ret.columns = ["future_ret"]
    stock_before_ret.columns = ["before_ret"]
    stock_one.columns = ["stock_price"]
    df_tmp = df.loc[df["コード"] == int(code), :].copy()
    df_tmp = df_tmp.merge(stock_future_ret, how = "left", left_on = "年度", right_index = True)
    df_tmp = df_tmp.merge(stock_one, how = "left", left_on = "年度", right_index = True)
    df_tmp = df_tmp.merge(stock_before_ret, how = "left", left_on = "年度", right_index = True)
    list_df.append(df_tmp)
    print("\r", int(100*i/len(list_code)),"% finish", end = "", sep = "")

df_main = pd.concat(list_df)
# df_main = df_main.dropna()
df_main.to_csv("../03_加工データ/分析用データ.csv", encoding = "CP932")

print(df_main["future_ret"].astype(np.float32).describe(percentiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))
df_main.loc[df_main["future_ret"] >= 10, "future_ret"] = 10
df_main.loc[df_main["future_ret"] <= -4, "future_ret"] = -4
df_main["future_ret"].hist(bins = 1000)

