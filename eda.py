import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    ACCESS ZIP FILE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# specifying the zip file name
file_name = "Medicare_Provider_Util_Payment_PUF_CY2017.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zp:
    zp.printdir()
    print('Extracting all the txt files now...')
    for file in zp.namelist():
        if zp.getinfo(file).filename.endswith('.txt'):
            zp.extract(file)
            CMS = pd.read_csv(file, delimiter="\t")
    print('Done!')

CMS = CMS.iloc[1:]
CMS.dropna(subset=['npi'], inplace=True)
CMS = CMS.loc[(CMS.npi != 0000000000) & (CMS['hcpcs_drug_indicator'] == 'N')]

# %%%%% Map LEIE Data %%%%%
LEIE = pd.read_csv("2019_LEIE.csv")
CMS["fraud"] = 0
CMS.fraud.loc[CMS['npi'].isin(LEIE.NPI)] = 1
del LEIE
end = len(CMS.loc[CMS.fraud == 1])
ind = np.arange(0, end, 1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                               EXPLORATORY DATA ANALYSIS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('Plotting Figure 1')
plt.figure()
plt.scatter(ind, CMS["average_submitted_chrg_amt"].loc[CMS.fraud == 1], c='#FF3028', label="Fraudulent")
plt.scatter(ind, CMS["average_submitted_chrg_amt"].loc[CMS.fraud == 0].iloc[ind], c='#2EE85B', label="Normal")
plt.title("Charge Amount Across Fraud and Non-Fraud Physicians")
plt.xlabel("Indices")
plt.ylabel("Average Charge Amount ($)")
plt.xlim(0, end)
plt.ylim(-10, 12000)
plt.legend()


print('Plotting Figure 2')
plt.figure()
plt.scatter(ind, CMS['average_Medicare_payment_amt'].loc[CMS.fraud == 1], c='#FF3028', label="Fraudulent")
plt.scatter(ind, CMS['average_Medicare_payment_amt'].loc[CMS.fraud == 0].iloc[ind], c='#2EE85B', label="Normal")
plt.title("Medicare Payment Amount Across Fraud and Non-Fraud Physicians")
plt.xlabel("Indices")
plt.ylabel("Average Medicare Payment After Deductible ($)")
plt.xlim(0, end)
plt.ylim(-10, 2000)
plt.legend()


print('Plotting Figure 3')
plt.figure()
ax = CMS["provider_type"].loc[CMS.fraud == 1].value_counts(normalize=True).nlargest(5).plot(kind='bar', rot=45)
plt.xlabel("Physician Type")
plt.ylabel("Frequency")
plt.title("Frequency of Top 5 Fraudulent Physician Types")


print('Plotting Figure 4')
plt.figure(4)
CMS["medicare_participation_indicator"].loc[CMS.fraud == 1].value_counts(normalize=True).plot(kind='bar')
plt.xlabel("Medicare Participation")
plt.ylabel("Frequency")
plt.title("Frequency of Medicare Participation for Fraudulent Physicians")


print('Plotting Figure 5')
plt.figure(5)
CMS["place_of_service"].loc[CMS.fraud == 1].value_counts(normalize=True).plot(kind='bar')
plt.xlabel("Place of Service")
plt.ylabel("Frequency")
plt.title("Place of Service for Fraudulent Physicians")
plt.show()