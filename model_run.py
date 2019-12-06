import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import Autoencoder as Ae


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                        LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device: ", device)

print(os.path.exists('data_clean/train.npz'))
train = np.load('data_clean/train.npz')
X_train = torch.from_numpy(train["X"]).float().to(device)
del train

print(os.path.exists('data_clean/test.npz'))
test = np.load('data_clean/test.npz')
X_test = torch.from_numpy(test["X"]).float().to(device)
y_test = test["y"]
del test

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    DEEP AUTOENCODER - 3HL
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialize Model
torch.manual_seed(123)
hl1 = 48
hl2 = 24
hl3 = 12
model = Ae.AE_3HL(98, hl1, hl2, hl3)
model = model.to(device)

# Initialize Hyperparameters
batch_size = 1000
learning_rate = 0.01
num_epochs = 10
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Store loss for plotting
history = dict()
history['train_loss'] = []
history['test_loss'] = []

# Training
print("Starting training loop...")
# start_time = time.time()
for epoch in range(num_epochs):
    h = np.array([])
    loss_train = 0  # Initializes train loss which will be added up after going forward on each batch
    model.train()  # Activates Dropout and makes BatchNorm use the actual training data to compute the mean and std
    # (this is the default behaviour but will be changed later on the evaluation phase)
    for batch in range(len(X_train)//batch_size + 1):  # Loops over the number of batches (n_examples//batch_size)
        inds = slice(batch*batch_size, (batch+1)*batch_size)  # Gets a slice to index the data
        optimizer.zero_grad()
        decoded = model(X_train[inds])
        loss = criterion(decoded, X_train[inds])
        loss.backward()
        optimizer.step()
        h = np.append(h, loss.item())
    model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
    with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
        predicted = model(X_test)  # computations and makes sure the model can't use the test data to learn
        test_loss = criterion(predicted, X_test).item()

    mean_loss = np.mean(h)
    print('epoch [{}/{}], Train Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch + 1, num_epochs, mean_loss, test_loss))
    history['train_loss'].append(mean_loss)
    history['test_loss'].append(test_loss)


# print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
torch.save(model.state_dict(), 'medfraud_ae_3hl.pth')


plt.figure()
plt.plot(history['train_loss'], label="Train Loss")
plt.plot(history['test_loss'], color='orange', label="Test Loss")
plt.title('Training & Test Loss for Deep AE - (98-{}-{}-{}-{}-{}-98)'.format(hl1, hl2, hl3, hl2, hl1))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                    DEEP AUTOENCODER - 2HL
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Initialize Model
# hl1 = 35
# hl2 = 12
# model = Ae.AE_2HL(98, hl1, hl2)
# model = model.to(device)
#
# # Initialize Hyperparameters
# batch_size = 1000
# learning_rate = 0.01
# num_epochs = 10
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Store loss for plotting
# history = dict()
# history['train_loss'] = []
# history['test_loss'] = []
#
# # Training
# print("Starting training loop...")
# # start_time = time.time()
# for epoch in range(num_epochs):
#     h = np.array([])
#     loss_train = 0  # Initializes train loss which will be added up after going forward on each batch
#     model.train()  # Activates Dropout and makes BatchNorm use the actual training data to compute the mean and std
#     # (this is the default behaviour but will be changed later on the evaluation phase)
#     for batch in range(len(X_train)//batch_size + 1):  # Loops over the number of batches (n_examples//batch_size)
#         inds = slice(batch*batch_size, (batch+1)*batch_size)  # Gets a slice to index the data
#         optimizer.zero_grad()
#         decoded = model(X_train[inds])
#         loss = criterion(decoded, X_train[inds])
#         loss.backward()
#         optimizer.step()
#         h = np.append(h, loss.item())
#     model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
#     with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
#         predicted = model(X_test)  # computations and makes sure the model can't use the test data to learn
#         test_loss = criterion(predicted, X_test).item()
#
#     mean_loss = np.mean(h)
#     print('epoch [{}/{}], Train Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch + 1, num_epochs, mean_loss, test_loss))
#     history['train_loss'].append(mean_loss)
#     history['test_loss'].append(test_loss)
#
#
# # print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
# torch.save(model.state_dict(), 'medfraud_ae_2hl.pth')
#
#
# plt.figure()
# plt.plot(history['train_loss'], label="Train Loss")
# plt.plot(history['test_loss'], color='orange', label="Test Loss")
# plt.title('Training & Test Loss for Deep AE - (98-{}-{}-{}-98)'.format(hl1, hl2, hl1))
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                   SHALLOW AUTOENCODER
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Initialize Model
# hl1 = 12
# model = Ae.ShallowAE(98, hl1)
# model = model.to(device)
#
# # Initialize Hyperparameters
# batch_size = 1000
# learning_rate = 0.01
# num_epochs = 10
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Store loss for plotting
# history = dict()
# history['train_loss'] = []
# history['test_loss'] = []
#
# # Training
# print("Starting training loop...")
# for epoch in range(num_epochs):
#     h = np.array([])
#     loss_train = 0  # Initializes train loss which will be added up after going forward on each batch
#     model.train()  # Activates Dropout and makes BatchNorm use the actual training data to compute the mean and std
#     # (this is the default behaviour but will be changed later on the evaluation phase)
#     for batch in range(len(X_train)//batch_size + 1):  # Loops over the number of batches (n_examples//batch_size)
#         inds = slice(batch*batch_size, (batch+1)*batch_size)  # Gets a slice to index the data
#         optimizer.zero_grad()
#         decoded = model(X_train[inds])
#         loss = criterion(decoded, X_train[inds])
#         loss.backward()
#         optimizer.step()
#         h = np.append(h, loss.item())
#     model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
#     with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
#         predicted = model(X_test)  # computations and makes sure the model can't use the test data to learn
#         test_loss = criterion(predicted, X_test).item()
#
#     mean_loss = np.mean(h)
#     print('epoch [{}/{}], Train Loss:{:.4f}, Test Loss:{:.4f}'.format(epoch + 1, num_epochs, mean_loss, test_loss))
#     history['train_loss'].append(mean_loss)
#     history['test_loss'].append(test_loss)
#
# torch.save(model.state_dict(), 'medfraud_shallow_ae.pth')
#
# plt.figure()
# plt.plot(history['train_loss'], label="Train Loss")
# plt.plot(history['test_loss'], color='orange', label="Test Loss")
# plt.title('Training & Test Loss for Shallow AE (98-{}-98)'.format(hl1))
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
