import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


train_date = "96TF_test0824"
df = pd.read_csv(f'/media/lu/lussd/kp_tf/record/test_matrix_{train_date}.csv')



# calculate f1 score
accuracy = accuracy_score(df['Labels'], np.round(df['Predictions']))
precision, recall, thresholds = precision_recall_curve(df['Labels'], df['Predictions'])
thresholds = np.append(thresholds, 1)
f1_scores = 2*(precision * recall) / (precision + recall)

# calculate roc,
fpr, tpr, _ = roc_curve(df['Labels'], df['Predictions'])
roc_auc = auc(fpr, tpr)

# calculae auprc
auprc = average_precision_score(df['Labels'], df['Predictions'])


# predictions
predictions_0 = df[df['Labels'] == 0]['Predictions']
predictions_1 = df[df['Labels'] == 1]['Predictions']

# plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(recall, precision, lw=2, label='Precision-Recall curve (area = %0.2f)' % auprc)
# plt.plot(thresholds, f1_scores, lw=2, label='F1 score')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate/Recall/Predictions')
plt.ylabel('True Positive Rate/Precision/F1 Score')
plt.title('Evaluation matrix')
plt.legend(loc="lower right")


# plot density
plt.subplot(1, 3, 2)
sns.kdeplot(predictions_0, label='Label 0', fill=True)
sns.kdeplot(predictions_1, label='Label 1', fill=True)
plt.title('Density Plot of Predictions for Labels 0 and 1')
plt.xlabel('Predictions')
plt.ylabel('Density')
plt.legend()


# plot f1
plt.subplot(1, 3, 3)
plt.plot(thresholds, f1_scores, lw=2, label='F1 score')
plt.title('F1 score vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()

plt.savefig(f'/media/lu/lussd/kp_tf/history_data/test_metrics_{train_date}.png')
plt.show()

print(f'Accuracy: {accuracy:.2f}')
print(f'AUC: {roc_auc:.2f}')
print(f'AUPRC: {auprc:.2f}')