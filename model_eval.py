import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, recall_score


def conf_plot(confmat):
    fig, ax = plt.subplots()
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.2)
    for h in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=h, s=confmat[h, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device: ", device)

# loss = 'data_clean/shallow_loss.npz'
loss = 'data_clean/2hl_loss.npz'
# loss = 'data_clean/3hl_loss.npz'

print("Loss Data Exists: ", os.path.exists(loss))
loss = np.load(loss)
pred_loss = loss["loss"]
y_test = loss["y"]
del loss


threshold_list = np.arange(0.01, 0.5, 0.01)
recall = []
precision = []
for i in threshold_list:
    y_pred = [1 if e > i else 0 for e in pred_loss]
    test_recall = recall_score(y_test, y_pred, average="weighted")
    recall.append(test_recall)

plt.figure()
plt.plot(threshold_list, recall, color="red")
plt.title("Effect of Threshold on Recall")
plt.ylabel("Recall")
plt.xlabel("Threshold")
plt.show()

thresh = [i for i, value in enumerate(recall) if value > 0.5]
threshold = threshold_list[thresh[0]]
print("Recall of 50% at Threshold of {}".format(threshold))
y_pred = [1 if e > threshold else 0 for e in pred_loss]


fpr, tpr, thresholds = roc_curve(y_test, pred_loss)
roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
print('AUC = %0.4f' % roc_auc)
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
conf_plot(conf_matrix)
