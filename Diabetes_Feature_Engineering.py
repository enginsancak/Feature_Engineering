##############################################################
# Özellik Mühendisliği (Feature Engineering)
##############################################################

###############################################################
# İş Problemi
###############################################################
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi
# istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmek gerekmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.
# Veri seti 9 değişken, 768 gözlemden oluşmakta olup 24 KB alan kaplamaktadır.

# Pregnancies             : Hamilelik sayısı
# Glucose                 : Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure          : Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness           : Cilt Kalınlığı
# Insulin                 : 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction: Soydaki kişilere göre diyabet olma ihtimalini hesaplayan bir fonksiyon
# BMI                     : Vücut kitle endeksi
# Age                     : Yaş (yıl)
# Outcome                 : Hastalığa sahip (1) ya da değil (0)

###############################################################
# Görev 1 : Keşifçi Veri Analizi
###############################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Adım 1: Genel Resim
df = pd.read_csv("datasets/diabetes.csv")
def check_df(dataframe, head=5):
    print("##############Shape#############")
    print(dataframe.shape)
    print("#############Types#############")
    print(dataframe.dtypes)
    print("#############NumberUnique######")
    print(dataframe.nunique())
    print("#############Head##############")
    print(dataframe.head(head))
    print("#########Tail##################")
    print(dataframe.tail(head))
    print("##############NA###############")
    print(dataframe.isnull().sum())
    print("#############Quantiles#########")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 1]).T)

check_df(df)

# Adım 2: Numerik ve Kategorik Değişkenleri Yakalayalım.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ----------
    dataframe: dataframe
            Değişken isimleri alınmak istenen dataframe'dir
    cat_th: int, optional
            Numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, optional
            Kategorik fakat kardinal olan değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat olanlar cat_cols'un içerisindedir.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve Kategorik Değişkenleri Analiz Edelim.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("###########################################")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "RATIO": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "Outcome", plot=True)

# Adım 4: Hedef Değişken Analizi Yapalım. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene
# göre numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, numerical_col):
    print(pd.DataFrame({numerical_col+"_MEAN": dataframe.groupby(target)[numerical_col].mean()}))

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Adım 5: Aykırı Değerleri Gözlemleyelim.
for col in num_cols:
    sns.boxplot(x=df[col], data=df)
    plt.show(block=True)

# Adım 6: Eksik Değerleri Gözlemleyelim
df.isnull().sum()

# Adım 7: Korelasyon Analizi Yapalım.
corr = df.corr()
corr

sns.set(rc={"figure.figsize":(12,12)})
sns.heatmap(corr, annot=True, cmap="RdBu")
plt.show()

# NOT: Çok yüksek korelasyonlu (>0.9) değerlere rastlanmamıştır.

###############################################################
# Görev 2 : Feature Engineering
###############################################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapalım. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabiliriz.

# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
zero_columns = [col for col in df.columns if any(df[col] == 0) and col not in ["Pregnancies", "Outcome"]]
zero_columns

# Gözlem birimlerinde 0 içeren değişkenlerin 0 olan gozlem degerlerini NaN ile değiştirelim.
for col in zero_columns:
   df.loc[df[col] == 0, col] = np.nan


# Eksik Gözlem Analizi
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() * 100 / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)

# Eksik Değerlerin Doldurulması
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
df.isnull().sum()


# Aykırı Değer Analizi ve Baskılama
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 2: Yeni Değişkenler Oluşturalım.
df.head()

# Age değişkeni 21-50 arası mature, 50'den büyükler senior
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI 18.5'ten küçük underweight, 18.5 ile 24.9 arası healty, 25 ile 29.9 arası overweight, 30 ve üzeri obese
df.loc[(df["BMI"] < 18.5), "NEW_BMI_CAT"] = "underweight"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NEW_BMI_CAT"] = "healty"
df.loc[(df["BMI"] >= 25) & (df["BMI"] <= 29.9), "NEW_BMI_CAT"] = "overweight"
df.loc[(df["BMI"] >= 30), "NEW_BMI_CAT"] = "obese"

# Glucose 140'tan küçük normal, 140 ile 199 arası prediabetes, 200 ve üstü diabetes
df.loc[(df["Glucose"] < 140), "NEW_GLUCOSE_CAT"] = "normal"
df.loc[(df["Glucose"] >= 140) & (df["Glucose"] <= 199), "NEW_GLUCOSE_CAT"] = "prediabetes"
df.loc[(df["Glucose"] >= 200), "NEW_GLUCOSE_CAT"] = "diabetes"

# Age ile vücut kitle endeksini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_CAT"] = "underweightmature"
df.loc[(df["Age"] >= 50) & (df["BMI"] < 18.5), "NEW_AGE_BMI_CAT"] = "underweightsenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NEW_AGE_BMI_CAT"] = "healtymature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NEW_AGE_BMI_CAT"] = "healtysenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 25) & (df["BMI"] <= 29.9), "NEW_AGE_BMI_CAT"] = "overweightmature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 25) & (df["BMI"] <= 29.9), "NEW_AGE_BMI_CAT"] = "overweightsenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["BMI"] >= 30), "NEW_AGE_BMI_CAT"] = "obesemature"
df.loc[(df["Age"] >= 50) & (df["BMI"] >= 30), "NEW_AGE_BMI_CAT"] = "obesesenior"


# Age ile glikoz değişkenlerini bir arada düşünerek kategorik değişken oluşturma
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["Glucose"] < 140), "NEW_AGE_CLUCOSE_CAT"] = "normalmature"
df.loc[(df["Age"] >= 50) & (df["Glucose"] < 140), "NEW_AGE_CLUCOSE_CAT"] = "normalsenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["Glucose"] >= 140) & (df["Glucose"] <= 199), "NEW_AGE_CLUCOSE_CAT"] = "prediabetesmature"
df.loc[(df["Age"] >= 50) & (df["Glucose"] >= 140) & (df["Glucose"] <= 199), "NEW_AGE_CLUCOSE_CAT"] = "prediabetessenior"
df.loc[(df["Age"] >= 21) & (df["Age"] < 50) & (df["Glucose"] >= 200), "NEW_AGE_CLUCOSE_CAT"] = "diabetesmature"
df.loc[(df["Age"] >= 50) & (df["Glucose"] >= 200), "NEW_AGE_CLUCOSE_CAT"] = "diabetessenior"

# insulin değeri ile kategorik değişken oluşturma
df.loc[(df["Insulin"] <= 166) & (df["Insulin"] >= 16), "NEW_INSULIN_CAT"] = "Normal"
df.loc[((df["Insulin"] > 166) | (df["Insulin"] < 16)), "NEW_INSULIN_CAT"] = "Abnormal"

# Çarpımlardan yeni değişkenler türetmek
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
# df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]


df.columns = [col.upper() for col in df.columns]
df.head()


# proportion z-test
df.groupby("NEW_AGE_CAT").agg({"OUTCOME": "mean"})


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_AGE_CAT"] == "mature", "OUTCOME"].sum(),
                                             df.loc[df["NEW_AGE_CAT"] == "senior", "OUTCOME"].sum()],

                                      nobs=[df.loc[df["NEW_AGE_CAT"] == "mature", "OUTCOME"].shape[0],
                                            df.loc[df["NEW_AGE_CAT"] == "senior", "OUTCOME"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p=0.0047 < 0.05, H0 (P1=P2) RED, iki oran arasında anlamlı bir farklılık vardır

df.groupby("NEW_GLUCOSE_CAT").agg({"OUTCOME": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_GLUCOSE_CAT"] == "normal", "OUTCOME"].sum(),
                                             df.loc[df["NEW_GLUCOSE_CAT"] == "prediabetes", "OUTCOME"].sum()],

                                      nobs=[df.loc[df["NEW_GLUCOSE_CAT"] == "normal", "OUTCOME"].shape[0],
                                            df.loc[df["NEW_GLUCOSE_CAT"] == "prediabetes", "OUTCOME"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p < 0.05, H0 RED, iki oran arasında anlamlı bir farklılık vardır



# Adım 3: Encoding İşlemlerini Gerçekleştirelim.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding Yapalım
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    label_encoder(df, col)


# One-Hot Encoding Yapalım
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in cat_cols if col not in binary_cols and df[col].nunique() != 2]
ohe_cols

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

# Adım 4: Numerik Değişkenler için Standartlaştırma Yapalım.
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()


# Adım 5: Model Kurma

# Hiç Bir İşlem Yapılmadan Önce Model Oluşturma (Base Model)
dff = pd.read_csv("datasets/diabetes.csv")
dff.head()
y = dff["Outcome"]
X = dff.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

# Feature Engineering işlemlerinden sonra model oluşturma
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.8
# Recall: 0.73
# Precision: 0.67
# F1: 0.7
# Auc: 0.78

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

