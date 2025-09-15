import matplotlib.pyplot as plt
import pandas as pd

# load history data
train_date = "96TF_test0824"

df_history = pd.read_csv(f'/media/lu/lussd/kp_tf/record/training_history_{train_date}.csv')

# plot Train/Val Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss')
plt.plot(df_history['epoch'], df_history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# plot Val Acc 和 Val AUC
plt.subplot(1, 2, 2)
plt.plot(df_history['epoch'], df_history['val_acc'], label='Val Accuracy')
plt.plot(df_history['epoch'], df_history['val_auc'], label='Val AUC')
plt.plot(df_history['epoch'], df_history['val_auprc'], label='Val AUPRC')
# plt.plot(df_history['epoch'], df_history['val_cor'], label='Val correlation')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.savefig(f'/media/lu/lussd/kp_tf/history_data/training_metrics_{train_date}.png')
plt.show()