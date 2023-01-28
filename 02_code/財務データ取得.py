# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 06:13:15 2022

@author: ozawa
"""

import pandas as pd
import numpy as np
import requests
import io
import os
import time
import pandas_datareader.data as web
import glob
os.chdir("G:/マイドライブ/01_自己研鑽/03_財務分析/01_元データ/")

#%%
# =============================================================================
# 年次の財務指標データをirbankから取得
# =============================================================================

for year in np.arange(10, 22):
    url = "https://f.irbank.net/files/00" + str(year) + "/"
    os.makedirs("20" + str(year), exist_ok = True)

    for kind in ["fy-balance-sheet.csv", "fy-cash-flow-statement.csv", "fy-profit-and-loss.csv", "fy-stock-dividend.csv"]:        
        time.sleep(80) # 時間制限があるため、毎回80秒のタイムスリープ
        response = requests.get(url + kind)
        df = pd.read_csv(io.BytesIO(response.content), header = 1)
        df.to_csv("20" + str(year) + "/" + kind)

#%%
YEAR = 22
url = "https://f.irbank.net/files/0000/"
os.makedirs("20" + str(YEAR), exist_ok = True)

for kind in ["fy-balance-sheet.csv", "fy-cash-flow-statement.csv", "fy-profit-and-loss.csv", "fy-stock-dividend.csv"]:        
    response = requests.get(url + kind)
    df = pd.read_csv(io.BytesIO(response.content), header = 1)
    df.to_csv("20" + str(YEAR) + "/" + kind)
    time.sleep(20) # 時間制限があるため、毎回80秒のタイムスリープ

#%%
# =============================================================================
# pandas_readerを使って、yahooファイナンスから上場企業の日次株価データを取得(エラーにはならないが時間制限でストップするため結構時間かかる)
# =============================================================================
exists_code = [x.split("\\")[-1][:-4] for x in glob.glob("個別銘柄_終値/*")]

df_stock_info = pd.read_csv("銘柄情報.csv", encoding = "CP932")
use_flg = df_stock_info["33業種コード"] != "-"
list_code = list(df_stock_info.loc[use_flg, "コード"])
code = 2317
start_date = "2005/01"
end_date = "2022/11"
for i, code in enumerate(list_code):
#    if str(code) in exists_code:
#        continue
    try:
        stock_p = web.DataReader(str(code) + ".T", "Google", start = start_date, end = end_date)
        stock_p.to_csv("個別銘柄_終値/" + str(code) + ".csv", encoding = "CP932")
    except:
        print("\n" + str(code) + ".Tは取得できず")
    print("\r" + str(code) + ":" + str(int(i*100/len(list_code))), end = "")
#%%
# =============================================================================
# 上記で取得した個別銘柄の株価を終値のみ抽出して1つのcsvにまとめる
# =============================================================================
list_code = glob.glob("個別銘柄_終値/*")

list_stock = []
for code in list_code:
    tmp = pd.read_csv(code, encoding = "CP932", index_col = 0)
    tmp = tmp["Adj Close"]
    tmp.name = code.split("\\")[-1][:-4]
    list_stock.append(tmp)

df_stock = pd.concat(list_stock,axis=1)
df_stock = df_stock.sort_index()
df_stock.to_csv("../03_加工データ/各銘柄終値.csv", encoding = "CP932")
