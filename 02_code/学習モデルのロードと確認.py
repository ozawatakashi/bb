# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 02:25:57 2022

@author: takashi.ozawa
"""

import pandas as pd
import numpy as np
import sweetviz as sv
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
import japanize_matplotlib
from optuna.integration import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import shap
import random as rn
import pickle
import category_encoders as ce
from sklearn.metrics import roc_curve, roc_auc_score
os.chdir("G:/マイドライブ/01_自己研鑽/03_財務分析/03_加工データ/")
#%%
# =============================================================================
# データ読み込み
# =============================================================================
df_m = pd.read_csv("分析用データ.csv", encoding = "CP932", index_col = 0)
df_m = df_m.drop_duplicates(subset = ["コード", "年度"], keep = "first")

df_stock_info = pd.read_csv("../01_元データ/銘柄情報.csv", encoding = "CP932")
#%%
# =============================================================================
# 説明変数の作成
# =============================================================================
df_m = df_m.merge(df_stock_info[["コード", "銘柄名", "33業種区分", "17業種区分"]], how = "left", on = "コード")

df_m["PER"] = df_m["stock_price"]/df_m["EPS"]
df_m["PBR"] = df_m["stock_price"]/df_m["BPS"]
df_m["ミックス係数"] = df_m["PER"] * df_m["PBR"]
df_m["理論株価"] = df_m["PER"] * df_m["BPS"] * df_m["ROE"]
df_m["理論株価倍率"] = df_m["stock_price"] / df_m["理論株価"]
df_m["ROS"] = df_m["経常利益"]/df_m["売上高"]
df_m["OIM"] = df_m["営業利益"]/df_m["売上高"]
df_m["売上高純利益率"] = df_m["純利益"]/df_m["売上高"]
df_m["総資産経常利益率"] = df_m["経常利益"]/df_m["総資産"]
df_m["総資産営業利益率"] = df_m["営業利益"]/df_m["総資産"]
df_m["総資産回転率"] = df_m["売上高"]/df_m["総資産"]
df_m["配当利回り"] = df_m["一株配当"] / df_m["stock_price"]
df_m["設備投資率_総資産"] = df_m["設備投資"] / df_m["総資産"]
df_m["設備投資率_純資産"] = df_m["設備投資"] / df_m["純資産"]
df_m["設備投資率_売上高"] = df_m["設備投資"] / df_m["売上高"]
df_m["設備投資率_純利益"] = df_m["設備投資"] / df_m["純利益"]
df_m["アクルーアル"] = df_m["純利益"] - df_m["営業CF"]
df_m["アクルーアル比率"] = df_m["アクルーアル"] / df_m["総資産"]
df_m["FCF"] = df_m["営業CF"] + df_m["投資CF"]
df_m["キャッシュ利益比率"] = df_m["営業CF"] / df_m["営業利益"]
df_m["33業種区分_cat"] = df_m["33業種区分"].copy()

col_cat = ["33業種区分_cat"]
ce_oe = ce.OrdinalEncoder(cols = col_cat, handle_unknown='impute') #Label Encodingの実施
df_m = ce_oe.fit_transform(df_m) #文字を序数に変換

#値を1の始まりから0の始まりにする
for i in col_cat:
    df_m[i] = df_m[i] - 1
    df_m[i] = df_m[i].astype("category")

col_calc_diff = ["ミックス係数",
                 "配当利回り",
                 "総資産経常利益率"]
col_diff = [x + "_前年差" for x in col_calc_diff]
df_m[col_diff] = df_m.groupby("コード")[col_calc_diff].diff()

col_calc_pct = ["理論株価",
                "BPS", "営業CF", "売上高"]
col_pct = [x + "_変化率" for x in col_calc_pct]
df_m[col_pct] = df_m.groupby("コード")[col_calc_pct].pct_change()

df_m["OIM_p_売上高_変化率"] = df_m["OIM"] + df_m["売上高_変化率"]

col_calc_pct = ["理論株価",
                "BPS", "営業CF"]
col_pct = [x + "_変化率" for x in col_calc_pct]

#%%
# =============================================================================
# 目的変数処理
# =============================================================================
df_m = df_m.merge(df_m.groupby(["年度", "17業種区分"])["future_ret"].mean().rename("year_ret"),
                  how = "left", on = ["年度", "17業種区分"])
df_m["target"] = np.sign(df_m["future_ret"])
df_m.loc[df_m["target"] == -1, "target"] = 0

df_m["month"] = [x.split("-")[1] for x in df_m["年度"]]
#%%
# =============================================================================
# 分析用データの作成
# =============================================================================
TRAIN_END_DATE = "2018-12-31"
TEST_START_DATE = "2019-12-31"
# =============================================================================
# X_col = ['総資産', 'ROE', 'ROA','配当利回り',
#          '自己資本比率', '営業CFマージン',
#        '配当性向', '総還元性向', '純資産配当率',
#        'before_ret','PER', 'PBR', 
#        'ROS', 'OIM', '総資産経常利益率',
#        '総資産回転率', '33業種区分_cat']
# =============================================================================
X_col = ['配当利回り', '営業CFマージン',
       'before_ret','PBR', "ROE",
       '総資産経常利益率', "総資産営業利益率",
       '総資産回転率', "設備投資率_純資産", "ミックス係数",
       "アクルーアル比率", "キャッシュ利益比率",
       "FCF"
       ] + col_diff + col_pct

df_use = df_m.dropna(subset = X_col)


X_train = df_use.query("年度 <= @TRAIN_END_DATE")[X_col]
X_test = df_use.dropna(subset = "target").query("年度 > @TEST_START_DATE")[X_col]
y_train = df_use.query("年度 <= @TRAIN_END_DATE")["target"]
y_test = df_use.dropna(subset = "target").query("年度 > @TEST_START_DATE")["target"]

df_test = df_use.query("年度 > @TEST_START_DATE").dropna(subset = ["target"])
df_all_test = df_use.query("年度 > @TEST_START_DATE")
#%%
with open("../05_model/上昇モデル/best_boosters_ver3.pkl", "rb") as fin:
    best_clf_up = pickle.load(fin)

#%%

# =============================================================================
# 直近の有力候補を確認(上昇モデル)
# =============================================================================
df_all_test["predict"] = np.mean([clf.predict(df_all_test[X_col]) for clf in best_clf_up], axis = 0)

df_all_test = df_all_test.sort_values(["年度" ,"predict"], ascending = False)
df_all_test["rank_each_month"] = df_all_test.groupby("年度")["predict"].rank(ascending = False)

explainers = [shap.TreeExplainer(ml) for ml in best_clf_up]
shap_values = [exp.shap_values(df_all_test[X_col]) for exp in explainers]
shap_values = np.mean(shap_values, axis = 0)

shap_values = shap_values[1]
base_value = np.mean([exp(df_all_test.iloc[0:1,:][X_col]).base_values[0][1] for exp in explainers])

print("======最新年月のデータ数====== \n ", df_all_test["年度"].value_counts().sort_index()[-1:])

df_best = df_all_test.iloc[:10, :]
print("======有力候補の銘柄======\n", df_best[["年度", "predict", "銘柄名", "33業種区分"]])
df_best_2 = df_all_test.query("年度 == '2022-08-01'").iloc[:10, :]
print("======先月の有力候補銘柄======\n", df_best_2[["年度", "predict", "銘柄名", "33業種区分"]])

df_shap_values = pd.DataFrame(shap_values, columns = X_col)

# best10
for i in range(10):
    shap_plt = df_shap_values.iloc[i, :].sort_values(key = abs)
    shap_plt["BASE"] = base_value
    shap_plt["TOTAL"] = shap_plt.sum() + base_value
    plt.barh(shap_plt.index, shap_plt)
    plt.title("上昇:" + df_all_test.iloc[i, :].loc["銘柄名"] + "\n " + df_all_test.iloc[i, :].loc["33業種区分"])
    #plt.savefig("../04_output/図示/最新の有力候補/" + str(i) + "_"  + df_all_test.iloc[i, :].loc["銘柄名"] + "_shap.jpg")
    plt.show()

list_imp = [clf.feature_importance(importance_type = "gain") for clf in best_clf_up]
arr_imp = np.mean(list_imp, axis = 0)
sr_imp = pd.Series(arr_imp, index = best_clf_up[0].feature_name())
sr_imp.sort_values().plot(kind = "barh")
plt.title("gini importance")
#plt.savefig("../04_output/図示/gini_importance.jpg")
plt.show()
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve


pred_train = np.mean([clf.predict(X_train) for clf in best_clf_up], axis = 0)
precision, recall, thresholds = precision_recall_curve(y_train, pred_train)

threshold = pd.Series(pred_train).describe(percentiles = [0.9])["90%"]

prec = precision_score(y_train, (pred_train > threshold)*1)

plt.plot(recall, precision, label='PR curve')
plt.axhline(prec, color = "k", linestyle = "dashed", linewidth = 1)
plt.legend()
plt.title('PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()
#%%
threshold = pd.Series(pred_train).describe(percentiles = [0.9])["90%"]

df_test["predict"] = np.mean([clf.predict(df_test[X_col]) for clf in best_clf_up], axis = 0)

df_test = df_test.sort_values(["年度"])
df_test["pred_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = False)
df_test["pred_worst_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = True)
print("======各月の上位1銘柄======\n", df_test.query("pred_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
print("======各月の下位1銘柄======\n", df_test.query("pred_worst_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])

df_top5 = df_test.query("(pred_rank_each_month  <= 5) & (predict >= @threshold)").groupby("年度")["future_ret"].mean()

df_worst5 = df_test.query("pred_worst_rank_each_month  <= 3").groupby("年度")["future_ret"].mean()

top_title = "TOP3 MEAN : " + str(np.round(df_top5.mean()*100, 1)) + "%"
all_title = "ALL MEAN : " + str(np.round(df_test.groupby("年度")["future_ret"].mean().mean()*100, 1)) + "%"
worst_title = "WORST3 MEAN : " + str(np.round(df_worst5.mean()*100, 1)) + "%"

df_top5.plot(x = "年度", kind = "bar", label = "TOP 3", color = "blue", align = "edge")
df_test.groupby("年度")["future_ret"].mean().plot(x = "年度", label = "ALL", color = "black")
df_worst5.plot(x = "年度", label = "WORST 3", kind = "bar", color = "red")
plt.axhline(0, color = "k", linestyle = "dashed", linewidth = 1)
plt.title(top_title + "\n" + all_title + "\n" + worst_title)
plt.legend()
plt.show()

# df_top5.to_csv("../04_output/上昇予測_test.csv", encoding = "CP932")
#%%
# =============================================================================
# 有力候補購入時の株価リターンと予測値の関係
# =============================================================================
pred_mean = 0.593
x_range = np.arange(1, 2.55, 0.05)

idx = 6490

df_one = df_all_test.loc[[idx], :].copy()
df_analyze = pd.concat([df_one]*len(x_range), axis = 0)

df_analyze["before_ret"] += (x_range - 1)
df_analyze["stock_price"] *= x_range
df_analyze["PER"] *= x_range
df_analyze["PBR"] *= x_range
df_analyze["配当利回り"] /= x_range

df_analyze["predict"] = np.mean([clf.predict(df_analyze[X_col]) for clf in best_clf_up], axis = 0)
df_analyze.index = (x_range*100 - 100).astype(int)

df_analyze["predict"].plot()
plt.axhline(pred_mean, color = "k", linestyle = "dashed", linewidth = 1, label = "Mean:" + str(np.round(pred_mean, 3)))
plt.title("株価成長率と予測値の変動\n" + df_analyze["銘柄名"].iloc[0])
plt.xlabel("株価リターン(%)")
plt.legend()
#%%
# =============================================================================
# 学習データ内の確認
# =============================================================================
df_m["predict"] = np.mean([clf.predict(df_m[X_col]) for clf in best_clf_up], axis = 0)
df_m.to_csv("../04_output/予測結果.csv", encoding = "Cp932")
df_auc = df_m.dropna(subset = ["target"]).query("年度 >= '20191231'")

fpr, tpr, thresholds = roc_curve(df_auc["target"], df_auc["predict"])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color = "r", linestyle = "dashed")
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.title("AUC : " + str(roc_auc_score(df_auc["target"], df_auc["predict"])))
plt.show()
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
df_m["predict"] = np.mean([clf.predict(df_m[X_col]) for clf in best_clf_up], axis = 0)
# df_m.to_csv("../04_output/予測結果.csv", encoding = "Cp932")
df_auc = df_m.dropna(subset = ["target"]).query("年度 >= '20191231'")

threshold = df_m.dropna(subset = ["target"]).query("年度 < '20181231'")["predict"].mean()

df_auc["decision"] = np.sign(df_auc["predict"] - threshold)
df_auc.loc[df_auc["decision"] == -1, "decision"] = 0

print("正解率 : " + str(accuracy_score(df_auc["target"], df_auc["decision"])))
print("Presicion : " + str(precision_score(df_auc["target"], df_auc["decision"])))
print("Recall : " + str(recall_score(df_auc["target"], df_auc["decision"])))

print("\n============閾値高め============\n")
df_auc["decision"] = np.sign(df_auc["predict"] - 0.77)
df_auc.loc[df_auc["decision"] == -1, "decision"] = 0
print("正解率 : " + str(accuracy_score(df_auc["target"], df_auc["decision"])))
print("Presicion : " + str(precision_score(df_auc["target"], df_auc["decision"])))
print("Recall : " + str(recall_score(df_auc["target"], df_auc["decision"])))
print("割合 : ", np.round(df_auc["decision"].mean()*100, 1) , "%", sep = "")

#df_auc["predict"].hist(bins = 100, color = "r", edgecolor = "k", alpha = 0.4, density = True)
#df_all_test["predict"].hist(bins = 100, color = "c", edgecolor = "k", alpha = 0.4, density = True)

#%%
# =============================================================================
# 決算月ごとに、上位3銘柄、全購入、下位3銘柄で比べる
# =============================================================================

df_test["predict"] = np.mean([clf.predict(df_test[X_col]) for clf in best_clf_up], axis = 0)

df_test = df_test.sort_values(["年度"])
df_test["pred_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = False)
df_test["pred_worst_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = True)
print("======各月の上位1銘柄======\n", df_test.query("pred_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
print("======各月の下位1銘柄======\n", df_test.query("pred_worst_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
df_top5 = df_test.query("(pred_rank_each_month  <= 5) & (predict <= 0.75)").groupby("年度")["future_ret"].mean()
df_worst5 = df_test.query("pred_worst_rank_each_month  <= 3").groupby("年度")["future_ret"].mean()

top_title = "TOP3 MEAN : " + str(np.round(df_top5.mean()*100, 1)) + "%"
all_title = "ALL MEAN : " + str(np.round(df_test.groupby("年度")["future_ret"].mean().mean()*100, 1)) + "%"
worst_title = "WORST3 MEAN : " + str(np.round(df_worst5.mean()*100, 1)) + "%"

df_top5.plot(x = "年度", kind = "bar", label = "TOP 3", color = "blue", align = "edge")
df_test.groupby("年度")["future_ret"].mean().plot(x = "年度", label = "ALL", color = "black")
df_worst5.plot(x = "年度", label = "WORST 3", kind = "bar", color = "red")
plt.axhline(0, color = "k", linestyle = "dashed", linewidth = 1)
plt.title(top_title + "\n" + all_title + "\n" + worst_title)
plt.legend()
plt.show()
#%%
df_m["predict"] = np.mean([clf.predict(df_m[X_col]) for clf in best_clf_up], axis = 0)
pred_mean_gp = df_m.dropna(subset = ["target"]).query("年度 < '20181231'").groupby("17業種区分")["predict"].mean()

df_gp = df_test.groupby(["年度", "17業種区分"])["predict"].mean().reset_index()
df_gp = df_gp.loc[df_gp["17業種区分"] != '-', :]
#for x in df_gp["17業種区分"].unique():
#    df_gp.loc[df_gp["17業種区分"] == x, "predict"] -= pred_mean_gp[x]
df_gp["rank"] = df_gp.groupby("年度")["predict"].rank(ascending = False)
df_gp_rank = df_gp.query("rank == 1")

df_gp_p = pd.read_csv("../03_加工データ/業界ごと時系列.csv", encoding = "CP932", index_col = 0)
df_gp_p.index = pd.to_datetime(df_gp_p.index)
df_gp_ret = df_gp_p.pct_change(12).shift(-10).dropna()
df_gp_ret = df_gp_ret.iloc[:-2,:]

df_gp_ret.index = df_gp_rank["年度"]
df_gp_ret = df_gp_ret.rename(columns = {"金融":"金融（除く銀行）"})
df_gp_ret = df_gp_ret.stack().reset_index()
df_gp_ret.columns = ["year", "kubun", "return"]
df_gp_ret["rank"] = df_gp_ret.groupby("year")["return"].rank(ascending = False)

df_gp["return"] = np.nan
df_gp["return_rank"] = np.nan

df_gp["17業種区分"] = df_gp["17業種区分"].apply(lambda x: x.split(" ")[0])
for i in df_gp.index:
    year, kubun = df_gp.loc[i, ["年度", "17業種区分"]]
    sign = (df_gp_ret["year"] == year) & (df_gp_ret["kubun"] == kubun)
    df_gp.loc[i, "return"] = df_gp_ret.loc[sign, "return"].iloc[0]
    df_gp.loc[i, "return_rank"] = df_gp_ret.loc[sign, "rank"].iloc[0]
    
df_gp[["return", "predict"]].corr(method = "spearman")

df_gp["per"] = np.nan
for i in df_gp.index:
    year = df_gp.loc[i, "年度"]
    rank = df_gp.loc[i, "rank"]
    rank_len = len(df_gp.loc[df_gp["年度"] == year, :])
    if rank == 1:df_gp.loc[i, "per"] = 1
    elif rank == rank_len:df_gp.loc[i, "per"] = 3
    else :df_gp.loc[i, "per"] = 2

df_gp.groupby("per")["return_rank"].mean()

df_gp.groupby(["年度", "per"])["return_rank"].mean().reset_index().pivot_table(values = ["return_rank"], index = "年度", columns = ["per"]).plot()
df_gp.groupby(["年度", "per"])["return"].mean().reset_index().pivot_table(values = ["return"], index = "年度", columns = ["per"]).plot()
#df_gp.groupby("per")["return"].mean().plot(kind = "bar")
#%%
