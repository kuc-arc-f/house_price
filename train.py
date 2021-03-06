# encoding: utf-8
# 予測問題。 通常データ train/ test.csv の使用して検証する。
# 評価
# score : XX


# 途中で使用するため、あらかじめ読み込んでおいてください。
# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
# 機械学習モジュール
import sklearn
from sklearn import linear_model
import pickle
import time 

#
def conv_dats(df):
    #欠損値の補完
    df["PoolQC"].fillna('NA', inplace=True)
    df["MiscFeature"].fillna('None', inplace=True)
    df["Alley"].fillna('NA', inplace=True)
    df["Fence"].fillna('NA', inplace=True)
    df["FireplaceQu"].fillna('NA', inplace=True)
    df["GarageQual"].fillna('NA', inplace=True)
    df["GarageFinish"].fillna('NA', inplace=True)
    df["GarageCond"].fillna('NA', inplace=True)
    df["GarageType"].fillna('NA', inplace=True)
    df["BsmtCond"].fillna('NA', inplace=True)
    df["BsmtExposure"].fillna('NA', inplace=True)
    df["BsmtQual"].fillna('NA', inplace=True)
    df["BsmtFinType2"].fillna('NA', inplace=True)
    df["BsmtFinType1"].fillna('NA', inplace=True)
    df["MasVnrType"].fillna('None', inplace=True)
    #
    df["GarageYrBlt"].fillna(0, inplace=True) 
    df["MasVnrArea"].fillna(0, inplace=True)
    df["BsmtHalfBath"].fillna(0, inplace=True)
    df["BsmtFullBath"].fillna(0, inplace=True)
    df["TotalBsmtSF"].fillna(0, inplace=True)
    df["BsmtUnfSF"].fillna(0, inplace=True)
    df["BsmtFinSF2"].fillna(0, inplace=True)
    df["BsmtFinSF1"].fillna(0, inplace=True)
    df["GarageArea"].fillna(0, inplace=True)
    df["GarageCars"].fillna(0, inplace=True)
    #
    df["MSZoning"].fillna('RL', inplace=True)
    df["Functional"].fillna('Typ', inplace=True)
    df["Utilities"].fillna("AllPub", inplace=True)
    df['SaleType']    = df['SaleType'].fillna(df['SaleType'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Electrical']  = df['Electrical'].fillna(df['Electrical'].mode()[0])
    #
    f = lambda x: x.fillna(x.mean())
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(f)
    return df

# 学習データ
global_start_time = time.time()
train_data = pd.read_csv("train.csv" )
test_data = pd.read_csv("test.csv" )
#print( train_data.shape )
#print( train_data.head() )

#print(train_data["Id"][: 10])
train_sub = train_data.drop("Id", axis=1)
test_sub  = test_data.drop("Id", axis=1)

# 目的変数
y_train = train_sub["SalePrice"]
# 説明変数に "xx" 以外を利用
train_sub = train_sub.drop("SalePrice", axis=1)

#学習用データとテストデータを一度統合する
df_all = pd.concat((train_sub , test_sub)).reset_index(drop=True)
print(df_all.shape )
df_all=conv_dats(df_all)
#quit()

tmp=df_all.isnull().sum()[ df_all.isnull().sum() != 0].sort_values(ascending=False)
#print(tmp)
#One Hot Encoding
df_all = pd.get_dummies(df_all)
print( df_all.shape )
ntrain = train_sub.shape[0]
x_train = df_all[:ntrain]
x_test  = df_all[ntrain:]

print( x_train.shape,  y_train.shape )
print( x_test.shape )
#quit()

# モデルのインスタンス
model = linear_model.LinearRegression()
# fit
clf = model.fit( x_train ,y_train)
print("train:",clf.__class__.__name__ ,clf.score(x_train,y_train))
#print("test:",clf.__class__.__name__ , clf.score(X_test,y_test))

# 偏回帰係数
print(pd.DataFrame({"Name": x_train.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )

# 切片 
#print(clf.intercept_)

# モデルを保存する
filename = 'model.pkl'
pickle.dump( model , open(filename, 'wb'))
print("model save, complete !!")
print ('time : ', time.time() - global_start_time)
quit()


#pred
#pred = model.predict(x_train )
pred = model.predict(x_test )
pred_int = np.array( pred , np.int32)
#pred = pred.astype(np.float16)
print( pred_int[: 10] )
#quit()
# 予測をしてCSVへ書き出す
Id = np.array( test_data["Id"]).astype(int)
df = pd.DataFrame(pred_int, Id, columns=["SalePrice"])
df.head()
#
df.to_csv("out.csv", index_label=["Id"])

#plt
a1=np.arange(len(x_test) )
plt.plot(a1 , pred , label = "predict")
plt.legend()
plt.grid(True)
plt.title("price pred")
plt.xlabel("x")
plt.ylabel("price")
plt.show()
