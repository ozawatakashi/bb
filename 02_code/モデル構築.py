# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:24:45 2022

@author: takashi.ozawa
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
import japanize_matplotlib
import optuna
import warnings
from optuna.integration import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import shap
import random as rn
import pickle
import category_encoders as ce
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score
from eli5.permutation_importance import get_score_importances

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.disable_default_handler()

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

# 値を1の始まりから0の始まりにする
for i in col_cat:
    df_m[i] = df_m[i] - 1
    df_m[i] = df_m[i].astype("category")

col_calc_diff = ["PBR", "PER","ROE", "ROA","ミックス係数",
                 "自社株買い", "自己資本比率",
                 "配当利回り", "ROS", "OIM",
                 "総資産経常利益率", "総資産回転率","理論株価倍率",
                 "営業CFマージン", "アクルーアル比率"]
col_diff = [x + "_前年差" for x in col_calc_diff]
df_m[col_diff] = df_m.groupby("コード")[col_calc_diff].diff()

col_calc_pct = ["EPS", "売上高", "総資産", "営業利益","理論株価",
                "経常利益", "純資産", "株主資本",
                "利益剰余金", "短期借入金",
                "長期借入金", "BPS", "営業CF",
                "投資CF", "財務CF", "設備投資",
                "現金同等物", "純利益", "アクルーアル"]
col_pct = [x + "_変化率" for x in col_calc_pct]
df_m[col_pct] = df_m.groupby("コード")[col_calc_pct].pct_change()

df_m["OIM_p_売上高_変化率"] = df_m["OIM"] + df_m["売上高_変化率"]

#%%
# =============================================================================
# 目的変数処理
# =============================================================================
df_use = df_m.dropna(subset = ["PBR_前年差"])
df_use["target"] = np.sign(df_use["future_ret"])
df_use.loc[df_use["target"] == -1, "target"] = 0

df_use["month"] = [x.split("-")[1] for x in df_use["年度"]]
#%%
# =============================================================================
# 分析用データの作成
# =============================================================================
TRAIN_END_DATE = "2017-12-31"
TEST_START_DATE = "2018-12-31"
SIMU_START_DATE = "2019-12-31"

X_col = ['総資産', '配当利回り',"ROS", "OIM",
         '自己資本比率', '営業CFマージン',
       '純資産配当率', 'before_ret',
       'PER', 'PBR', "ROA", "ROE","理論株価倍率",
       '総資産経常利益率', "総資産営業利益率",
       '総資産回転率', "設備投資率_総資産",
       "設備投資率_純資産", "設備投資率_売上高",
       "設備投資率_純利益","ミックス係数",
       "アクルーアル比率","売上高純利益率",
       "キャッシュ利益比率", "FCF",
       '33業種区分_cat'
       ] + col_diff + col_pct

df_use = df_use.dropna(subset = X_col)

X_train = df_use.query("年度 <= @TRAIN_END_DATE")[X_col]
X_test = df_use.dropna(subset = "target").query("(年度 > @TEST_START_DATE) & (年度 < @SIMU_START_DATE)")[X_col]
X_simu = df_use.dropna(subset = "target").query("年度 > @SIMU_START_DATE")[X_col]
y_train = df_use.query("年度 <= @TRAIN_END_DATE")["target"]
y_test = df_use.dropna(subset = "target").query("(年度 > @TEST_START_DATE) & (年度 < @SIMU_START_DATE)")["target"]
y_simu = df_use.dropna(subset = "target").query("年度 > @SIMU_START_DATE")["target"]


df_test = df_use.query("年度 > @SIMU_START_DATE").dropna(subset = ["target"])
df_all_test = df_use.query("年度 > @SIMU_START_DATE")

#%%
# =============================================================================
# モデル定義
# =============================================================================

def get_model(use_col):
    def score(X, y):
        y_pred = np.mean(best_model.predict(X), axis = 0)
        return roc_auc_score(y, y_pred)

    # 設定・実行
    np.random.seed(0)
    rn.seed(0)
    lgb_train = lgb.Dataset(X_train[use_col], y_train)
    folds = KFold(n_splits = 10, shuffle = True, random_state = 0)    
    lgb_params = {
        "objective":"binary", #regression
        "metric":"auc",
    #     "learning_rate":0.05,
        "seed":0,
        "n_jobs":-1,
        "force_col_wise":True,
        "deterministic":True,
        "verbosity":-1,
        # "is_unbalance":True
    }    
    
    tuner_cv = lgb.LightGBMTunerCV(
          lgb_params, lgb_train,
          num_boost_round=10000,
          early_stopping_rounds=100,
          verbose_eval = None,
          folds=folds,
          return_cvbooster = True,
          optuna_seed = 0,
          verbosity = None,
          show_progress_bar = False
          # , time_budget = 10
      )
    
    tuner_cv.run()
    best_model = tuner_cv.get_best_booster()
    
    # AUC精度確認
    pred_test = np.mean(best_model.predict(X_test[use_col]), axis = 0)
    pred_simu = np.mean(best_model.predict(X_simu[use_col]), axis = 0)
    auc_res = roc_auc_score(y_test, pred_test)
    auc_simu = roc_auc_score(y_simu, pred_simu)
    print("AUC TEST:", auc_res)
    print("AUC SIMU:", auc_simu)
    
    # Permutation Importanceで変数選択
    base_score, score_decreases = get_score_importances(score, X_test[use_col].values, y_test.values, random_state = 0, n_iter = 10)
    df_p_imp = pd.DataFrame({
        "p_imp":np.mean(score_decreases, axis=0)
        }, index = X_test[use_col].columns)
    list_drop = list(df_p_imp.query("p_imp <= 0").index)
    print("==========削除==========\n", list_drop, sep = "")
    
    return auc_res, auc_simu, list_drop, best_model
#%%
list_drop = [1]
list_auc = []
use_col = X_col
while len(list_drop) > 0:
    auc_res, auc_simu, list_drop, best_model = get_model(use_col)
    list_auc.append([auc_res, auc_simu])
    use_col = [col for col in use_col if not (col in list_drop)]
#%%
# =============================================================================
# モデル保存
# =============================================================================
with open("../05_model/best_boosters.pkl", "wb") as fout:
    pickle.dump(best_model.boosters, fout)
#%%
# =============================================================================
# モデルロード
# =============================================================================
with open("../05_model/best_boosters.pkl", "rb") as fin:
    best_model = pickle.load(fin)
#%%
use_col = best_model[0].feature_name()
#%%
# =============================================================================
# テストデータの精度確認と閾値設定
# =============================================================================
pred_test = np.mean([ml.predict(X_test[use_col]) for ml in best_model], axis = 0)
pred_simu = np.mean([ml.predict(X_simu[use_col]) for ml in best_model], axis = 0)

threshold = pred_test.mean()
pred_test_decision = np.sign(pred_test - threshold)
pred_test_decision[pred_test_decision == -1] = 0

print("正解率 : " + str(accuracy_score(y_test, pred_test_decision)))
print("Presicion : " + str(precision_score(y_test, pred_test_decision)))
print("Recall : " + str(recall_score(y_test, pred_test_decision)))
print("positive(%):" + str(np.mean(pred_test_decision)))
print("positive(num):" + str((pred_test_decision > 0).sum()))
#%%
# =============================================================================
# AUC確認
# =============================================================================
df_test["pred"] = np.mean([ml.predict(df_test[use_col]) for ml in best_model], axis = 0)
df_all_test["predict"] = np.mean([ml.predict(df_all_test[use_col]) for ml in best_model], axis = 0)
pred_test = np.mean([ml.predict(X_test[use_col]) for ml in best_model], axis = 0)
pred_simu = np.mean([ml.predict(X_simu[use_col]) for ml in best_model], axis = 0)

fpr, tpr, thresholds = roc_curve(y_test, pred_test)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color = "r", linestyle = "dashed")
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.title("AUC test: " + str(roc_auc_score(y_test, pred_test)))
# plt.savefig("../04_output/図示/AUC.jpg")
plt.show()

fpr, tpr, thresholds = roc_curve(y_simu, pred_simu)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color = "r", linestyle = "dashed")
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.title("AUC SIMU: " + str(roc_auc_score(y_simu, pred_simu)))
# plt.savefig("../04_output/図示/AUC.jpg")
plt.show()


#%%
# =============================================================================
# 精度確認
# =============================================================================
pred_test = np.mean([ml.predict(X_test[use_col]) for ml in best_model], axis = 0)
pred_simu = np.mean([ml.predict(X_simu[use_col]) for ml in best_model], axis = 0)

threshold = 0.75 # pred_test.mean()
pred_test_decision = np.sign(pred_test - threshold)
pred_test_decision[pred_test_decision == -1] = 0

print("正解率 : " + str(accuracy_score(y_test, pred_test_decision)))
print("Presicion : " + str(precision_score(y_test, pred_test_decision)))
print("Recall : " + str(recall_score(y_test, pred_test_decision)))
print("positive(%):" + str(np.mean(pred_test_decision)))
print("positive(num):" + str((pred_test_decision > 0).sum()))
#%%
# =============================================================================
# シミュレーション
# =============================================================================
df_test = df_test.sort_values("年度")
df_test["年度"] = pd.to_datetime(df_test["年度"])
df_test["signal"] = np.nan
df_test["signal_minas"] = np.nan

df_test.loc[df_test["pred"] >= threshold, "signal"] = 1
df_test.loc[(df_test["pred"] <= 0.3) , "signal_minas"] = 1

df_test["rank1"] = df_test["signal"] * df_test["future_ret"]
df_test["rank2"] = df_test["signal_minas"] * df_test["future_ret"]
df_test["benchmark"] = df_test["future_ret"]
rank1_plt = df_test.groupby("年度")[["rank1"]].mean()
rank2_plt = df_test.groupby("年度")[["rank2"]].mean()
bench_plt = df_test.groupby("年度")[["benchmark"]].mean()

fig = plt.figure()
ax1 = fig.subplots()
ax1.bar(rank1_plt.index, rank1_plt.values.reshape(-1), width = 10, color = "r", align = "edge", alpha = 0.3, label = "rank1")
ax1.bar(rank2_plt.index, rank2_plt.values.reshape(-1), width = 10, color = "blue", alpha = 0.3, label = "rank2")
ax1.plot(bench_plt.index, bench_plt.values.reshape(-1), label = "benchmark")
ax1.axhline(0, color = "k", linestyle = "dashed", linewidth = 1)
# ax1.grid(which = "major", axis = "x", color = "green", alpha = 0.8, linestyle = "--", linewidth = 1)
# ax1.grid(which = "major", axis = "y", color = "blue", alpha = 0.8, linestyle = "--", linewidth = 1)
fig.autofmt_xdate(rotation=45)
plt.legend()
plt.show()

print(df_test.dropna(subset = ["rank1"])["年度"].value_counts().sort_index())
print(df_test.dropna(subset = ["rank2"])["年度"].value_counts().sort_index())
print(df_test["年度"].value_counts().sort_index())

#%%
# =============================================================================
# シミュレーション
# =============================================================================
df_test = df_test.sort_values("年度")
df_test["年度"] = pd.to_datetime(df_test["年度"])
df_test.dropna(subset = use_col)
df_test["signal"] = np.nan
df_test["signal_minas"] = np.nan

df_test["rank_pred"] = df_test.groupby("年度").rank(ascending = False)["pred"]
df_test["rank_pred_minas"] = df_test.groupby("年度").rank(ascending = True)["pred"]

df_test.loc[df_test["rank_pred"] <= 5, "signal"] = 1
df_test.loc[df_test["rank_pred_minas"] <= 1 , "signal_minas"] = 1

df_test["rank1"] = df_test["signal"] * df_test["future_ret"]
df_test["rank2"] = df_test["signal_minas"] * df_test["future_ret"]
df_test["benchmark"] = df_test["future_ret"]
rank1_plt = df_test.groupby("年度")[["rank1"]].mean()
rank2_plt = df_test.groupby("年度")[["rank2"]].mean()
bench_plt = df_test.groupby("年度")[["benchmark"]].mean()

fig = plt.figure()
ax1 = fig.subplots()
ax1.bar(rank1_plt.index, rank1_plt.values.reshape(-1), width = 10, color = "r", align = "edge", alpha = 0.3, label = "rank1")
ax1.bar(rank2_plt.index, rank2_plt.values.reshape(-1), width = 10, color = "blue", alpha = 0.3, label = "rank2")
ax1.plot(bench_plt.index, bench_plt.values.reshape(-1), label = "benchmark")
# ax1.plot(bench_plt.index, (np.mean([rank1_plt, rank2_plt], axis = 0)).reshape(-1), label = "model", color = "y")
ax1.axhline(0, color = "k", linestyle = "dashed", linewidth = 1)
# ax1.grid(which = "major", axis = "x", color = "green", alpha = 0.8, linestyle = "--", linewidth = 1)
# ax1.grid(which = "major", axis = "y", color = "blue", alpha = 0.8, linestyle = "--", linewidth = 1)
fig.autofmt_xdate(rotation=45)
plt.legend()
plt.show()

# print(df_test.dropna(subset = ["rank1"])["年度"].value_counts().sort_index())
# print(df_test.dropna(subset = ["rank2"])["年度"].value_counts().sort_index())
# print(df_test["年度"].value_counts().sort_index())
print("ベンチマークバックテスト　リターン平均:", bench_plt.mean().iloc[0])
print("上昇モデルバックテスト　リターン平均:", rank1_plt.mean().iloc[0])

#%%
