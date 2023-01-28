# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:37:24 2022

@author: ozawa
"""

import pandas as pd
import numpy as np
import sweetviz as sv
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

os.chdir("C:/Users/ozawa/OneDrive/ドキュメント/07_データサイエンス関連/03_財務分析/03_加工データ/")
#%%
df_m = pd.read_csv("分析用データ.csv", encoding = "CP932", index_col = 0)
df_m.loc[df_m["future_ret"] >= 10, "future_ret"] = 10
df_m.loc[df_m["future_ret"] <= -4, "future_ret"] = -4
df_m.loc[df_m["Adj Close"] >= 10, "Adj Close"] = 10
df_m.loc[df_m["Adj Close"] <= -4, "Adj Close"] = -4

df_info = pd.read_excel("../01_元データ/銘柄情報.xls")
df_kind = df_info[["コード","17業種区分", "33業種区分", "銘柄名"]]

df_m = df_m.merge(df_kind, how = "left", on = "コード")
df_m = df_m.dropna(axis = 0, subset = ["銘柄名"])
df_m.to_csv("分析用データ_ver2.csv", encoding = "CP932")
#%%
list_kind = np.unique((df_m["33業種区分"]))
kind = list_kind[1]
use_cols = df_m.columns[df_m.dtypes == np.float64]


df_tmp = df_m.loc[df_m["33業種区分"] == kind, :]
for col in use_cols:
    x_tick = pd.qcut(df_tmp[col], 20, duplicates = "drop")
    df_gp = df_tmp.groupby(x_tick).mean()[[col, "future_ret"]]
    plt.plot(col, "future_ret", data = df_gp)
    plt.title(col)
    plt.legend()
    plt.show()
    

#%%
my_report = sv.analyze([df_m, "main data"], "future_ret")
my_report.show_html("sweetviz_report.html")
#%%
df_m.groupby("33業種区分").count()["future_ret"]
