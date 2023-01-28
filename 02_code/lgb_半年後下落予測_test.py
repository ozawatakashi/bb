# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 03:43:24 2022

@author: ozawa
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

# =============================================================================
# col_calc_ts = ["PBR", "PER", "自己資本比率", "配当利回り"]
# col_pct = [x + "前年差" for x in col_calc_ts]
# df_m[col_pct] = df_m.groupby("コード")[col_calc_ts].diff()
# df_m = df_m.dropna(subset = ["PBR"])
# =============================================================================


#%%
# =============================================================================
# 中身確認
# =============================================================================
print(df_m.info())
df_desc = df_m.drop(["コード", "銘柄名", "年度", "33業種区分", "17業種区分"], axis = 1).astype(np.float32).describe(percentiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999], include = "all")
print(df_desc)
df_corr = df_m.drop(["コード", "銘柄名", "年度", "33業種区分", "17業種区分"], axis = 1).corr(method = "spearman")
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
# =============================================================================
# lightgbmtunercvの構築
# =============================================================================
np.random.seed(0)
rn.seed(0)
lgb_train = lgb.Dataset(X_train, y_train)
folds = KFold(n_splits = 10, shuffle = True, random_state = 0)

lgb_params = {
    "objective":"binary", #regression
    "metric":"auc",
#     "learning_rate":0.05,
    "seed":0,
    "n_jobs":-1,
    "force_col_wise":True,
    "deterministic":True
}

tuner_cv = lgb.LightGBMTunerCV(
      lgb_params, lgb_train,
      num_boost_round=10000,
      early_stopping_rounds=100,
      verbose_eval=100,
      folds=folds,
      return_cvbooster = True,
      optuna_seed = 0,
      categorical_feature = list(col_cat)
  )
#%%
# =============================================================================
# 学習実行
# =============================================================================
tuner_cv.run()

#%%
# =============================================================================
# テストデータ検証
# =============================================================================
best_model = tuner_cv.get_best_booster()
with open("../05_model/下落モデル/best_boosters.pkl", "wb") as fout:
    pickle.dump(best_model.boosters, fout)

df_test["predict"] = np.mean(best_model.predict(X_test), axis = 0)
df_all_test["predict"] = np.mean(best_model.predict(df_all_test[X_col]), axis = 0)

if tuner_cv.best_params["objective"] == "binary":
    fpr, tpr, thresholds = roc_curve(df_test["target"], df_test["predict"])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color = "r", linestyle = "dashed")
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.title("AUC : " + str(roc_auc_score(df_test["target"], df_test["predict"])))
    plt.savefig("../04_output/図示_下落/AUC.jpg")    
    plt.show()
else:
    test_corr = df_test[["target", "predict"]].corr(method = "spearman")
    print("テストデータ相関 :", test_corr.iloc[0, 1])
    print("テストデータ決定係数 :", r2_score(df_test["target"], df_test["predict"]))
    print("符号の正答率 : ",np.round( (df_test["target"] * df_test["predict"] > 0).mean()*100, 1),"%", sep = "" )
    plt.scatter(df_test["target"], df_test["predict"])
    plt.title("テストデータ散布図")
    plt.xlabel("target")
    plt.ylabel("予測値")
    plt.savefig("../04_output/図示_下落/散布図.jpg")
    plt.show()

pred_mean = df_test["predict"].mean()
pred_median = df_test["predict"].median()
pred_sigma = df_test["predict"].std()*3

df_test["predict"].hist(bins = 100, color = "c", edgecolor = "k", alpha = 0.4)
plt.axvline(pred_mean, color = "k", linestyle = "dashed", linewidth = 1, label = "Mean:" + str(np.round(pred_mean, 3)))
plt.axvline(pred_median, color = "r", linestyle = "dashed", linewidth = 1, label = "Median:" + str(np.round(pred_median, 3)))
plt.axvline(pred_sigma + pred_mean, color = "blue", linestyle = "dashed", linewidth = 1, label = "Std * 3:" + str(np.round(pred_sigma, 3)))
plt.axvline(-pred_sigma + pred_mean, color = "blue", linestyle = "dashed", linewidth = 1)
plt.title("テストデータの予測値ヒストグラム")
plt.legend()
plt.savefig("../04_output/図示_下落/ヒストグラム.jpg")
plt.show()

#%%
# =============================================================================
# バックテストもどき検証
# =============================================================================
df_test = df_test.sort_values(by = "predict", ascending = False)
print("=======ベンチマーク1(全銘柄)のリターン======\n", df_test["future_ret"].describe(), sep = "")
print("=======ベンチマーク2(低PBR20銘柄)のリターン======\n", df_test.sort_values("PBR")["future_ret"].iloc[:20].describe(), sep = "")
print("=======ベンチマーク3(高PBR20銘柄)のリターン======\n", df_test.sort_values("PBR")["future_ret"].iloc[-20:].describe(), sep = "")
print("=======上位10銘柄のリターン======\n", df_test["future_ret"].iloc[:10].describe(), sep = "")
print("=======下位10銘柄のリターン======\n", df_test["future_ret"].iloc[-10:].describe(), sep = "")
print("=======各33業種の上位1銘柄を保有した場合======")
print(df_test.drop_duplicates("33業種区分", keep = "first")["future_ret"].describe(), sep = "")
print("=======各17業種の上位1銘柄を保有した場合======")
print(df_test.drop_duplicates("17業種区分", keep = "first")["future_ret"].describe(), sep = "")
print("=======上位10銘柄の業種=======\n", df_test[["future_ret", "銘柄名", "17業種区分"]].iloc[:10, :], sep = "")
print("=======下位10銘柄の業種=======\n", df_test[["future_ret", "銘柄名", "17業種区分"]].iloc[-10:, :], sep = "")


for code_type in df_test["17業種区分"].dropna().unique():
    df_plt = df_test.loc[df_test["17業種区分"] == code_type, :]
    plt.scatter(df_plt["future_ret"], df_plt["predict"])
    plt.title(code_type)
    plt.savefig("../04_output/図示_下落/業界別散布図/" + "散布図_" + str(code_type) + ".jpg")
    plt.show()
#%%
# =============================================================================
# 変数重要度など
# =============================================================================
list_imp = [best_model.boosters[i].feature_importance(importance_type = "gain") for i in range(10)]
arr_imp = np.mean(list_imp, axis = 0)
sr_imp = pd.Series(arr_imp, index = best_model.boosters[0].feature_name())
sr_imp.sort_values().plot(kind = "barh")
plt.title("gini importance")
plt.savefig("../04_output/図示_下落/gini_importance.jpg")
plt.show()

explainers = [shap.TreeExplainer(ml) for ml in best_model.boosters]
shap_values = [exp.shap_values(X_test) for exp in explainers]
shap_values = np.mean(shap_values, axis = 0)
if  tuner_cv.best_params["objective"] == "binary":
    shap_values = shap_values[1]
shap.summary_plot(shap_values, X_test, plot_type = "bar", show = False)
plt.savefig("../04_output/図示_下落/shap_summary.jpg")
plt.show()

X_test_cat = X_test.copy()
for i, col in enumerate(col_cat):
    col_map = {v-1:k for k, v in ce_oe.category_mapping[i]["mapping"].items()}
    X_test_cat[col] = X_test[col].map(col_map)

for col in X_col:
    shap.dependence_plot(ind = col, shap_values = shap_values,
                         xmin = "percentile(1)", xmax = "percentile(99)",
                         features = X_test_cat, show = False)
    if col in col_cat:plt.grid()
    plt.savefig("../04_output/図示_下落/shap/shap_" + col + ".jpg")
    plt.show()

#%%
# =============================================================================
# 決算月ごとに、上位3銘柄、全購入、下位3銘柄で比べる
# =============================================================================
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
# =============================================================================
# 決算月ごとに、予測値の高さに合わせて購入していく
# =============================================================================
df_test = df_test.sort_values(["年度"])
df_buy = df_test.query("(predict >= @pred_mean + @pred_sigma/3) & (pred_rank_each_month <= 5)")
df_buy.index = pd.to_datetime(df_buy.index)
count = df_buy.groupby("年度")["future_ret"].count()
df_buy = df_buy.groupby("年度")["future_ret"].mean()
print("平均リターン：", df_buy.mean())

fig = plt.figure()
ax1 = fig.subplots()
ax2 = ax1.twinx()
ax1.plot(df_buy, color = "r")
ax2.bar(count.index, count, alpha = 0.4)
fig.autofmt_xdate(rotation=45)
plt.title("決算月ごと 予測3σ以上5銘柄以下 \n 平均リターン : " + str(np.round(df_buy.mean()*100, 1)) + "%")
plt.savefig("../04_output/図示_下落/検証プロット.jpg")
plt.show()

#%%
# =============================================================================
# 予測結果保存
# =============================================================================
df_m["predict"] = np.mean(best_model.predict(df_m[X_col]), axis = 0)
df_m.to_csv("../04_output/予測結果_下落.csv", encoding = "Cp932")
#%%
# =============================================================================
# ブレインパッドの予測値確認
# =============================================================================
df_brain_pad = df_m.query("コード == 3655")
df_brain_pad["predict"] = np.mean(best_model.predict(df_brain_pad[X_col]), axis = 0)
df_brain_pad[["年度", "future_ret", "predict", "銘柄名", "17業種区分"]]
#%%
# =============================================================================
# 直近の有力候補とワーストを確認
# =============================================================================
df_all_test = df_all_test.sort_values(["年度" ,"predict"], ascending = False)
df_all_test["rank_each_month"] = df_all_test.groupby("年度")["predict"].rank(ascending = False)

explainers = [shap.TreeExplainer(ml) for ml in best_model.boosters]
shap_values = [exp.shap_values(df_all_test[X_col]) for exp in explainers]
shap_values = np.mean(shap_values, axis = 0)

if  tuner_cv.best_params["objective"] == "binary":
    shap_values = shap_values[1]
    base_value = np.mean([exp(df_all_test.iloc[0:1,:][X_col]).base_values[0][1] for exp in explainers])
else:
    base_value = np.mean([exp(df_all_test.iloc[0:1,:][X_col]).base_values[0] for exp in explainers])

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
    plt.title(df_all_test.iloc[i, :].loc["銘柄名"] + "\n " + df_all_test.iloc[i, :].loc["33業種区分"])
    plt.savefig("../04_output/図示_下落/最新の有力候補/" + str(i) + "_"  + df_all_test.iloc[i, :].loc["銘柄名"] + "_shap.jpg")
    plt.show()

# worst10
for i in range(10):
    shap_plt = df_shap_values.iloc[-i, :].sort_values(key = abs)
    shap_plt["BASE"] = base_value
    shap_plt["TOTAL"] = shap_plt.sum() + base_value
    plt.barh(shap_plt.index, shap_plt)
    plt.title(df_all_test.iloc[-i, :].loc["銘柄名"] + "\n " + df_all_test.iloc[-i, :].loc["33業種区分"])
    plt.savefig("../04_output/図示_下落/最新のWorst/" + str(i) + "_"  + df_all_test.iloc[-i, :].loc["銘柄名"] + "_shap.jpg")
    plt.show()

#%%
print(df_all_test.query("コード == 3929")[["年度", "future_ret", "predict", "銘柄名", "33業種区分"]])
df_one = df_all_test.query("コード == 3929").iloc[0, :]

i = df_all_test.reset_index().query("コード == 3929").index[0]
shap_plt = df_shap_values.iloc[i, :].sort_values(key = abs)
shap_plt["TOTAL"] = shap_plt.sum()
plt.barh(shap_plt.index, shap_plt, left = base_value)
plt.title(df_all_test.iloc[i, :].loc["銘柄名"] + "\n " + df_all_test.iloc[i, :].loc["33業種区分"])
plt.show()
#%%

"""
df_plt = df_test.query("future_ret < -0.5")
plt.scatter(df_plt["future_ret"], df_plt["predict"])
plt.title("テストデータ散布図")
plt.xlabel("1年後リターン")
plt.ylabel("予測値")
plt.show()
"""
print(df_all_test.query("銘柄名 == 'マネーフォワード'")[["コード", "年度", "future_ret", "predict", "銘柄名", "33業種区分"]])
