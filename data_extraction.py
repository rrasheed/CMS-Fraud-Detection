import numpy as np
import pandas as pd
import os
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def feature_expand(df):
    # %%%%% Part 2: Process and Clean Data %%%%%%
    # The processing and cleaning techniques are derived from Herland et al. J Big Data (2018)
    g_mean = df.groupby('npi')['average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
                               'average_Medicare_standard_amt', 'average_submitted_chrg_amt',
                               'bene_day_srvc_cnt', 'bene_unique_cnt', 'line_srvc_cnt'].mean()
    g_mean = pd.DataFrame.from_dict(g_mean)
    g_mean.reset_index(inplace=True)
    g_mean.columns = ['npi', 'mean1', 'mean2', 'mean3', 'mean4', 'mean5', 'mean6', 'mean7']
    df = pd.merge(df, g_mean, on='npi', how='outer')
    del g_mean
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%
    g_sum = df.groupby(['npi'])['average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
                                'average_Medicare_standard_amt', 'average_submitted_chrg_amt',
                                'bene_day_srvc_cnt', 'bene_unique_cnt', 'line_srvc_cnt'].sum()
    g_sum = pd.DataFrame.from_dict(g_sum)
    g_sum.reset_index(inplace=True)
    g_sum.columns = ['npi', 'sum1', 'sum2', 'sum3', 'sum4', 'sum5', 'sum6', 'sum7']
    df = pd.merge(df, g_sum, on='npi', how='outer')
    del g_sum
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%
    g_std = df.groupby(['npi'])['average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
                                'average_Medicare_standard_amt', 'average_submitted_chrg_amt',
                                'bene_day_srvc_cnt', 'bene_unique_cnt', 'line_srvc_cnt'].std()
    g_std = pd.DataFrame.from_dict(g_std)
    g_std.reset_index(inplace=True)
    g_std.columns = ['npi', 'std1', 'std2', 'std3', 'std4', 'std5', 'std6', 'std7']
    g_std.fillna(0, inplace=True)
    df = pd.merge(df, g_std, on='npi', how='outer')
    del g_std

    return df


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    ACCESS ZIP FILE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# specifying the zip file name

file_name = "Medicare.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    print('Extracting all the txt files now...')
    for file in zip.namelist():
        if zip.getinfo(file).filename.endswith('.txt'):
            zip.extract(file)
            CMS = pd.read_csv(file, delimiter="\t")
            CMS = CMS.iloc[1:]
            CMS.dropna(subset=['npi'], inplace=True)
            CMS = CMS.loc[(CMS.npi != 0000000000) & (CMS['hcpcs_drug_indicator'] == 'N')]
            CMS = CMS[['npi', 'nppes_provider_gender', 'provider_type', 'medicare_participation_indicator',
                       'place_of_service', 'average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
                       'average_Medicare_standard_amt', 'average_submitted_chrg_amt', 'bene_day_srvc_cnt',
                       'bene_unique_cnt', 'line_srvc_cnt']]
    print('Done!')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                           DATA CLEANING / FEATURE EXPAND
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_cols = ['npi', 'average_Medicare_allowed_amt', 'average_Medicare_payment_amt', 'average_Medicare_standard_amt',
            'average_submitted_chrg_amt', 'bene_day_srvc_cnt', 'bene_unique_cnt', 'line_srvc_cnt']

CMS = feature_expand(CMS)

for i in range(1, len(num_cols)):
    CMS[num_cols[i]].fillna(CMS['mean'+str(i)], inplace=True)

print("NAN values have been filled with mean of respective physician...")

obj_cols = ['nppes_provider_gender', 'provider_type', 'medicare_participation_indicator', 'place_of_service']

CMS.dropna(subset=obj_cols, inplace=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    FRAUD MAPPING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%% Map LEIE Data %%%%%
LEIE = pd.read_csv("LEIE.csv")
NPI = list(LEIE.NPI)
del LEIE
CMS["fraud"] = 0
CMS.fraud.loc[CMS['npi'].isin(NPI)] = 1
CMS.sort_values(by=['fraud'], inplace=True)
print(CMS.fraud.value_counts(normalize=True))
del NPI


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    DATA ENCODING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# instantiate labelencoder object
le = LabelEncoder()
CMS[obj_cols] = CMS[obj_cols].apply(lambda col: le.fit_transform(col))

X = CMS[['nppes_provider_gender', 'provider_type', 'medicare_participation_indicator',
         'place_of_service', 'average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
         'average_Medicare_standard_amt', 'average_submitted_chrg_amt', 'bene_day_srvc_cnt',
         'bene_unique_cnt', 'line_srvc_cnt']].loc[CMS.fraud == 0].values
Xf = CMS[['nppes_provider_gender', 'provider_type', 'medicare_participation_indicator',
         'place_of_service', 'average_Medicare_allowed_amt', 'average_Medicare_payment_amt',
          'average_Medicare_standard_amt', 'average_submitted_chrg_amt', 'bene_day_srvc_cnt',
          'bene_unique_cnt', 'line_srvc_cnt']].loc[CMS.fraud == 1].values

y = CMS['fraud'].values
del CMS

# instantiate OHE object
ohe = OneHotEncoder(categorical_features=[1], sparse=False)
X = ohe.fit_transform(X)
Xf = ohe.transform(Xf)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                  TRAIN/TEST SPLIT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_train, X_test_norm = train_test_split(X, test_size=0.3)
print(np.shape(X_train))
print(np.shape(X_test_norm))
y_train = np.zeros(len(X_train))


X_test = np.concatenate((X_test_norm, Xf), axis=0)
y_test_norm = list(np.zeros((len(X_test_norm))))
yf_test = list(np.ones((len(Xf))))
y_test = np.array(y_test_norm + yf_test)
del X_test_norm, y_test_norm, yf_test

X_test, y_test = shuffle(X_test, y_test, random_state=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                  STANDARDIZATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

print(np.shape(X_train))
print(np.shape(X_test))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                  SAVE DATA TO ZIP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez_compressed('data_clean/train', X=X_train, y=y_train)
print(os.path.exists('data_clean/train.npz'))
np.savez_compressed('data_clean/test', X=X_test, y=y_test)
print(os.path.exists('data_clean/test.npz'))
