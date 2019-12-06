import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import Autoencoder as Ae

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device: ", device)

print("Test Data Exists: ", os.path.exists('data_clean/test.npz'))
test = np.load('data_clean/test.npz')
X_test = torch.from_numpy(test["X"]).float().to(device)
y_test = test["y"]
del test

# PATH = 'medfraud_shallow_ae.pth'
# model = Ae.ShallowAE(98, 12)

# PATH = 'medfraud_ae_2hl.pth'
# model = Ae.AE_2HL(98, 35, 12)

PATH = 'medfraud_ae_3hl.pth'
model = Ae.AE_3HL(98, 48, 24, 12)

model.load_state_dict(torch.load(PATH))
model.to(device)
criterion = nn.MSELoss()

print("Testing in Progress....")
pred_loss = []
start_time = time.time()
with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
    for original in X_test:
        output = model(original)  # computations and makes sure the model can't use the test data to learn
        test_loss = criterion(output, original).data.item()
        pred_loss.append(test_loss)
print('Total Test Time: %.2f min' % ((time.time() - start_time) / 60))

pred_loss = np.array(pred_loss)
recon = pd.DataFrame(pred_loss)
recon['fraud'] = y_test
recon.columns = ['error', 'fraud']

print("Average Non Fraudulent Reconstruction Error: {}".format(recon.error.loc[recon.fraud == 0].mean()))
print("Average Fraudulent Reconstruction Error: {}".format(recon.error.loc[recon.fraud == 1].mean()))
del recon

# np.savez_compressed('data_clean/shallow_loss', loss=pred_loss, y=y_test)
# print(os.path.exists('data_clean/shallow_loss.npz'))

# np.savez_compressed('data_clean/2hl_loss', loss=pred_loss, y=y_test)
# print(os.path.exists('data_clean/2hl_loss.npz'))

np.savez_compressed('data_clean/3hl_loss', loss=pred_loss, y=y_test)
print(os.path.exists('data_clean/3hl_loss.npz'))
