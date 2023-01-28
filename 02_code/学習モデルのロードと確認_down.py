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
df_m = pd.read_csv("分析用データ_half_year.csv", encoding = "CP932", index_col = 0)
df_m = df_m.drop_duplicates(subset = ["コード", "年度"], keep = "first")

df_stock_info = pd.read_csv("../01_元データ/銘柄情報.csv", encoding = "CP932")
#%%
# =============================================================================
# 説明変数の作成
# =============================================================================
df_m = df_m.merge(df_stock_info[["コード", "銘柄名", "33業種区分", "17業種区分"]], how = "left", on = "コード")

df_m["PER"] = df_m["stock_price"]/df_m["EPS"]
df_m["PBR"] = df_m["stock_price"]/df_m["BPS"]
df_m["ROS"] = df_m["経常利益"]/df_m["売上高"]
df_m["OIM"] = df_m["営業利益"]/df_m["売上高"]
df_m["総資産経常利益率"] = df_m["経常利益"]/df_m["総資産"]
df_m["総資産回転率"] = df_m["売上高"]/df_m["総資産"]
df_m["配当利回り"] = df_m["一株配当"] / df_m["stock_price"]
df_m["33業種区分_cat"] = df_m["33業種区分"].copy()

col_cat = ["33業種区分_cat"]
ce_oe = ce.OrdinalEncoder(cols = col_cat, handle_unknown='impute') #Label Encodingの実施
df_m = ce_oe.fit_transform(df_m) #文字を序数に変換

#値を1の始まりから0の始まりにする
for i in col_cat:
    df_m[i] = df_m[i] - 1
    df_m[i] = df_m[i].astype("category")
#%%
# =============================================================================
# 目的変数処理
# =============================================================================
df_m = df_m.merge(df_m.groupby(["年度", "17業種区分"])["future_ret"].mean().rename("year_ret"),
                  how = "left", on = ["年度", "17業種区分"])
df_m["target"] = np.sign(df_m["future_ret"])*-1
df_m.loc[df_m["target"] == -1, "target"] = 0

df_m["month"] = [x.split("-")[1] for x in df_m["年度"]]
df_use = df_m.copy() #.query("month == '03'")

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
X_col = ['総資産', '配当利回り',
         '自己資本比率', '営業CFマージン',
       '純資産配当率', 'before_ret','PER',
       'PBR', '総資産経常利益率', 
       '総資産回転率', '33業種区分_cat'
       ]

df_use = df_use.dropna(subset = X_col)


X_train = df_use.query("年度 <= @TRAIN_END_DATE")[X_col]
X_test = df_m.dropna(subset = "target").query("年度 > @TEST_START_DATE")[X_col]
y_train = df_use.query("年度 <= @TRAIN_END_DATE")["target"]
y_test = df_m.dropna(subset = "target").query("年度 > @TEST_START_DATE")["target"]

df_test = df_m.query("年度 > @TEST_START_DATE").dropna(subset = ["target"])
df_all_test = df_m.query("年度 > @TEST_START_DATE")
#%%
with open("../05_model/下落モデル/best_boosters.pkl", "rb") as fin:
    best_clf_down = pickle.load(fin)
#%%
# =============================================================================
# 直近のワーストを確認(下落モデル)
# =============================================================================

df_all_test["predict"] = np.mean([clf.predict(df_all_test[X_col]) for clf in best_clf_down], axis = 0)
df_all_test = df_all_test.sort_values(["年度" ,"predict"], ascending = False)
df_all_test["rank_each_month"] = df_all_test.groupby("年度")["predict"].rank(ascending = False)

explainers = [shap.TreeExplainer(ml) for ml in best_clf_down]
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
    plt.title("下落:" + df_all_test.iloc[i, :].loc["銘柄名"] + "\n " + df_all_test.iloc[i, :].loc["33業種区分"])
    #plt.savefig("../04_output/図示/最新の有力候補/" + str(i) + "_"  + df_all_test.iloc[i, :].loc["銘柄名"] + "_shap.jpg")
    plt.show()

list_imp = [clf.feature_importance(importance_type = "gain") for clf in best_clf_down]
arr_imp = np.mean(list_imp, axis = 0)
sr_imp = pd.Series(arr_imp, index = best_clf_down[0].feature_name())
sr_imp.sort_values().plot(kind = "barh")
plt.title("gini importance")
#plt.savefig("../04_output/図示/gini_importance.jpg")
plt.show()
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve


pred_train = np.mean([clf.predict(X_train) for clf in best_clf_down], axis = 0)
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

df_test["predict"] = np.mean([clf.predict(df_test[X_col]) for clf in best_clf_down], axis = 0)

df_test = df_test.sort_values(["年度"])
df_test["pred_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = False)
df_test["pred_worst_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = True)
print("======各月の上位1銘柄======\n", df_test.query("pred_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
print("======各月の下位1銘柄======\n", df_test.query("pred_worst_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])

df_top5 = df_test.query("(pred_rank_each_month  <= 5) & (predict >= @threshold)").groupby("年度")["future_ret"].mean()

top_title = "TOP3 MEAN : " + str(np.round(df_top5.mean()*100*-1, 1)) + "%"
all_title = "ALL MEAN : " + str(np.round(df_test.groupby("年度")["future_ret"].mean().mean()*100, 1)) + "%"

(df_top5*-1).plot(x = "年度", kind = "bar", label = "TOP 3", color = "blue", align = "edge")
df_test.groupby("年度")["future_ret"].mean().plot(x = "年度", label = "ALL", color = "black")

plt.axhline(0, color = "k", linestyle = "dashed", linewidth = 1)
plt.xticks(rotation= 90)
plt.title(top_title + "\n" + all_title)
plt.legend()
plt.show()

df_top5.to_csv("../04_output/下落予測_test.csv", encoding = "CP932")
df_test.groupby("年度")["future_ret"].mean().to_csv("../04_output/ベンチマーク予測_test.csv", encoding = "CP932")
#%%
df_test.groupby("年度")["predict"].count().plot(kind = "bar")
plt.axhline(200, color = "k", linestyle = "dashed", linewidth = 1)

#%%
# =============================================================================
# 有力候補購入時の株価リターンと予測値の関係
# =============================================================================
print("======有力候補の銘柄======\n", df_best[["年度", "predict", "銘柄名", "33業種区分"]])
pred_mean = 0.432
x_range = np.arange(1, 0.05, -0.05)

idx = 38299

df_one = df_all_test.loc[[idx], :].copy()
df_analyze = pd.concat([df_one]*len(x_range), axis = 0)

df_analyze["before_ret"] += (x_range - 1)
df_analyze["stock_price"] *= x_range
df_analyze["PER"] *= x_range
df_analyze["PBR"] *= x_range
df_analyze["配当利回り"] /= x_range

df_analyze["predict"] = np.mean([clf.predict(df_analyze[X_col]) for clf in best_clf_down], axis = 0)
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
df_m["predict"] = np.mean([clf.predict(df_m[X_col]) for clf in best_clf_down], axis = 0)
df_m.to_csv("../04_output/予測結果_down.csv", encoding = "Cp932")
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
from sklearn.metrics import accuracy_score, precision_score, recall_score

df_m["predict"] = np.mean([clf.predict(df_m[X_col]) for clf in best_clf_down], axis = 0)
df_m.to_csv("../04_output/予測結果_down.csv", encoding = "Cp932")
threshold = df_m.dropna(subset = ["target"]).query("年度 < '20181231'")["predict"].mean()

df_auc = df_m.dropna(subset = ["target"]).query("年度 >= '20191231'")

df_auc["decision"] = np.sign(df_auc["predict"] - threshold)
df_auc.loc[df_auc["decision"] == -1, "decision"] = 0

print("正解率 : " + str(accuracy_score(df_auc["target"], df_auc["decision"])))
print("Presicion : " + str(precision_score(df_auc["target"], df_auc["decision"])))
print("Recall : " + str(recall_score(df_auc["target"], df_auc["decision"])))

df_auc["decision"] = np.sign(df_auc["predict"] - 0.70)
df_auc.loc[df_auc["decision"] == -1, "decision"] = 0
print("正解率 : " + str(accuracy_score(df_auc["target"], df_auc["decision"])))
print("Presicion : " + str(precision_score(df_auc["target"], df_auc["decision"])))
print("Recall : " + str(recall_score(df_auc["target"], df_auc["decision"])))

#df_auc["predict"].hist(bins = 100, color = "r", edgecolor = "k", alpha = 0.4, density = True)
#df_all_test["predict"].hist(bins = 100, color = "c", edgecolor = "k", alpha = 0.4, density = True)
#%%
# =============================================================================
# 決算月ごとに、上位3銘柄、全購入、下位3銘柄で比べる
# =============================================================================
df_test["predict"] = np.mean([clf.predict(df_test[X_col]) for clf in best_clf_down], axis = 0)
df_test = df_test.sort_values(["年度"])
df_test["pred_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = False)
df_test["pred_worst_rank_each_month"] = df_test.groupby("年度")["predict"].rank(ascending = True)
print("======各月の上位1銘柄======\n", df_test.query("pred_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
print("======各月の下位1銘柄======\n", df_test.query("pred_worst_rank_each_month  <= 1")[["年度", "predict", "future_ret", "銘柄名", "33業種区分"]])
df_top5 = df_test.query("pred_rank_each_month  <= 3").groupby("年度")["future_ret"].mean()
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
df_test["predict"].mean()

#%%
# =============================================================================
# 直近の有力候補を確認(上昇モデル)
# =============================================================================
df_all_test["predict"] = np.mean([clf.predict(df_all_test[X_col]) for clf in best_clf_down], axis = 0)

df_all_test = df_all_test.sort_values(["年度" ,"predict"], ascending = False)
df_all_test["rank_each_month"] = df_all_test.groupby("年度")["predict"].rank(ascending = False)

explainers = [shap.TreeExplainer(ml) for ml in best_clf_down]
shap_values = [exp.shap_values(df_all_test[X_col]) for exp in explainers]
shap_values = np.mean(shap_values, axis = 0)

shap_values = shap_values[1]
base_value = np.mean([exp(df_all_test.iloc[0:1,:][X_col]).base_values[0][1] for exp in explainers])

print("======最新年月のデータ数====== \n ", df_all_test["年度"].value_counts().sort_index()[-1:])

df_best = df_all_test.iloc[:10, :]
print("======有力候補の銘柄======\n", df_best[["年度", "predict", "銘柄名", "33業種区分"]])


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
i = df_all_test.sort_values("年度").query("コード == 3655").index[-1]
df_all_test.loc[i, "stock_price"] = 913
df_all_test.loc[i, "PER"] = 913/df_all_test.loc[i, "EPS"]
df_all_test.loc[i, "PBR"] = 913/df_all_test.loc[i, "BPS"]
df_all_test.loc[i, "配当利回り"] = df_all_test.loc[i, "一株配当"] / 913
#%%
np.mean([clf.predict(df_all_test.loc[[i], X_col]) for clf in best_clf_down], axis = 0)
np.mean([clf.predict(df_all_test.loc[[i], X_col]) for clf in best_clf_up], axis = 0)
#%%
