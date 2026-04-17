import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, f1_score
from tensorflow.keras import layers, models, optimizers, regularizers
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from prepare_data import prepare_dataset
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import concatenate
from keras.utils import plot_model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Attention, Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten, MultiHeadAttention, LayerNormalization, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
from tensorflow.keras.callbacks import LearningRateScheduler
import random
from keras.saving import register_keras_serializable
from trans_former import TransformerBlock

def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=5, initial_lr=1e-3, final_lr=1e-8):
    if epoch < warmup_epochs:
        # ウォームアップ期間
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # 余弦アニーリング
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * progress))

total_epochs = 3000
lr_scheduler = LearningRateScheduler(lambda epoch: cosine_decay_with_warmup(epoch, total_epochs))
n_splits = 20
person = "s01"

# データセットの読み込みと準備
file_path = f'/Users/sudaxin/Desktop/data_preprocessed_python/{person}.dat'
#selected_channels = ['Fp1','Fp2']  # Adjust based on your selected channels
selected_channels = ['Fp1','Fp2']  # Adjust based on your selected channels
X, y = prepare_dataset(file_path, selected_channels)
print(X.shape)

# Ensure no values exceed the range of 1 to 9
y = np.clip(y, 1, 9)

# Normalize y to [-1, 1]
y = (y - 5) / 4  # Transform [1, 9] to [-1, 1]

# Print the distribution of y values
print("Valence range:", np.min(y[:, 0]), np.max(y[:, 0]))
print("Arousal range:", np.min(y[:, 1]), np.max(y[:, 1]))

# Ensure the data has enough samples
print(f"Total samples: {X.shape[0]}")

# Cross-Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
valence_reports = []
arousal_reports = []
valence_auc_scores = []
arousal_auc_scores = []
y_test_all_arousal = []
prediction_all_arousal = []
y_test_all_valence = []
prediction_all_valence = []


"""
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(key_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-9)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-9)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding
    x = layers.Conv1D(64, 2, activation='relu',padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 2, activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Transformer Encoder
    x = layers.Reshape((-1, 128))(x)  # Reshape for Transformer
    transformer_block = TransformerBlock(num_heads=6, key_dim=128, ff_dim=128)
    x = transformer_block(x, training=True)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2)(x)  # 2 outputs: valence and arousal
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate = 1e-5),loss ='mse' ,metrics=['mae'],)
    return model
"""
def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding
    #x = layers.Conv1D(64, 2, activation='relu',padding='same')(inputs)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D(2)(x)
    #x = layers.Conv1D(128, 2, activation='relu',padding='same')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D(2)(x)
    
    # Transformer Encoder
    #x = layers.Reshape((-1, 128))(x)  # Reshape for Transformer
    transformer_block = TransformerBlock(num_heads=6, key_dim=128, ff_dim=128)
    x = transformer_block(inputs, training=True)
    
    # Flatten and Dense layers
    x = layers.Flatten()(x)
    #x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.004))(x)
    x = layers.Dropout(0.1)(x)
    #x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.004))(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2)(x)  # 2 outputs: valence and arousal
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(),loss ='mse' ,metrics=['mae'],)
    return model

best_model = None
best_total_mse = float('inf') 
best_fold = 0  
best_valence_mse = float('inf')  
best_arousal_mse = float('inf')  

fold = 1
for train_index, test_index in kf.split(X):
    
    print(f'fold{fold} started')
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #print(X_train.shape)
    # Ensure the data is correctly reshaped
    """X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))"""
    X_train = X_train.reshape((X_train.shape[0], 5, -1))
    X_test = X_test.reshape((X_test.shape[0], 5, -1))
    
    #print(X_train.shape)
    # Recreate the model for each fold
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_transformer_model(input_shape)
    
    history = model.fit(X_train, y_train, epochs=total_epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping,lr_scheduler])

    model.save(f'/Users/sudaxin/Desktop/KOBA/models/{person}/model_{fold}.keras', save_format='keras')

    # 損失のグラフをプロット
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Loss Curve Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/loss/loss_curve_fold_{fold}.png')
    plt.close()

    predictions = model.predict(X_test)

    # モデルの評価
    predictions = model.predict(X_test)
    valence_mse = mean_squared_error(y_test[:, 0], predictions[:, 0])
    arousal_mse = mean_squared_error(y_test[:, 1], predictions[:, 1])
    total_mse = valence_mse + arousal_mse  # ValenceとArousalのMSEの合計

    

    # 最良のモデルを更新
    if total_mse < best_total_mse:
        best_total_mse = total_mse
        best_model = model
        best_fold = fold
        best_valence_mse = valence_mse
        best_arousal_mse = arousal_mse

    print(f"Fold {fold} - Valence MSE: {valence_mse:.4f}, Arousal MSE: {arousal_mse:.4f}, Total MSE: {total_mse:.4f}")

    valence_true = y_test[:, 0] > 0  # Threshold is 0 to match [-1, 1] range
    valence_pred = predictions[:, 0] > 0
    valence_report = classification_report(valence_true, valence_pred, output_dict=True, zero_division=1)
    valence_reports.append(valence_report)
    valence_report_df = pd.DataFrame(valence_report).transpose()
    valence_report_df.to_csv(f'/Users/sudaxin/Desktop/KOBA/classification_report/valence_classification_report_fold_{fold}.csv')

    arousal_true = y_test[:, 1] > 0
    arousal_pred = predictions[:, 1] > 0
    arousal_report = classification_report(arousal_true, arousal_pred, output_dict=True, zero_division=1)
    arousal_reports.append(arousal_report)
    arousal_report_df = pd.DataFrame(arousal_report).transpose()
    arousal_report_df.to_csv(f'/Users/sudaxin/Desktop/KOBA/classification_report/arousal_classification_report_fold_{fold}.csv')

    # Save confusion matrix for valence
    valence_conf_matrix = confusion_matrix(valence_true, valence_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(valence_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Valence Confusion Matrix Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/classification_report/valence_confusion_matrix_fold_{fold}.png')
    plt.close()

    # Save confusion matrix for arousal
    arousal_conf_matrix = confusion_matrix(arousal_true, arousal_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(arousal_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Arousal Confusion Matrix Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/classification_report/arousal_confusion_matrix_fold_{fold}.png')
    plt.close()

    # Calculate and save ROC and AUC for valence
    valence_fpr, valence_tpr, _ = roc_curve(valence_true, predictions[:, 0])
    valence_roc_auc = auc(valence_fpr, valence_tpr)
    valence_auc_scores.append(valence_roc_auc)
    plt.figure()
    plt.plot(valence_fpr, valence_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % valence_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Valence Fold {fold}')
    plt.legend(loc="lower right")
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/classification_report/valence_roc_curve_fold_{fold}.png')
    plt.close()

    # Calculate and save ROC and AUC for arousal
    arousal_fpr, arousal_tpr, _ = roc_curve(arousal_true, predictions[:, 1])
    arousal_roc_auc = auc(arousal_fpr, arousal_tpr)
    arousal_auc_scores.append(arousal_roc_auc)
    plt.figure()
    plt.plot(arousal_fpr, arousal_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % arousal_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Arousal Fold {fold}')
    plt.legend(loc="lower right")
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/classification_report/arousal_roc_curve_fold_{fold}.png')
    plt.close()

    # 残差プロットの作成
    plt.figure(figsize=(10, 6))
    # 第2象限と第4象限のポイントを赤く、それ以外を青くする
    colors = np.where((y_test[:, 0] * predictions[:, 0]) < 0, 'red', 'blue')
    plt.scatter(y_test[:, 0], predictions[:, 0], color=colors, edgecolor='k', alpha=0.7)
    plt.plot([-1,1], [-1,1], linestyle='-')
    plt.axhline(0, color='black', linewidth=1)  # x軸を追加
    plt.axvline(0, color='black', linewidth=1)  # y軸を追加
    plt.xlabel('Actual Valence')
    plt.ylabel('Predicted Valence')

    plt.fill([0, 1, 1, 0], [0, 0, 1, 1], 'skyblue', alpha=0.5)
    plt.fill([0, -1, -1, 0], [0, 0, -1, -1], 'skyblue', alpha=0.5)

    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(f'Residual for Valence Fold {fold}')
    #現在の日時を取得

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # テキストを配置
    plt.text(0.95, 0.05, f'Created: {date_string}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/residuals/valence_residual_plot_fold_{fold}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    colors = np.where((y_test[:, 1] * predictions[:, 1]) < 0, 'red', 'blue')
    plt.scatter(y_test[:, 1], predictions[:, 1], color=colors, edgecolor='k', alpha=0.7)
    plt.plot([-1,1], [-1,1], linestyle='-')
    plt.axhline(0, color='black', linewidth=1)  # x軸を追加
    plt.axvline(0, color='black', linewidth=1)  # y軸を追加
    plt.xlabel('Actual Arousal')
    plt.ylabel('Predicted Arousal')

    plt.fill([0, 1, 1, 0], [0, 0, 1, 1], 'skyblue', alpha=0.5)
    plt.fill([0, -1, -1, 0], [0, 0, -1, -1], 'skyblue', alpha=0.5)

    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(f'Residual for Arousal Fold {fold}')
            # 現在の日時を取得
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # テキストを配置
    plt.text(0.95, 0.05, f'Created: {date_string}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
    plt.savefig(f'/Users/sudaxin/Desktop/KOBA/residuals/arousal_residual_plot_fold_{fold}.png')
    plt.close()

    y_test_all_valence.append(y_test[:,0])
    prediction_all_valence.append(predictions[:,0])
    y_test_all_arousal.append(y_test[:,1])
    prediction_all_arousal.append(predictions[:,1])

    print(f'fold{fold} finished')
    

    fold += 1
    

y_test_all_valence = np.concatenate(y_test_all_valence)
prediction_all_valence = np.concatenate(prediction_all_valence)
y_test_all_arousal = np.concatenate(y_test_all_arousal)
prediction_all_arousal = np.concatenate(prediction_all_arousal)

# データフレームの作成
df_valence = pd.DataFrame({
    '実際の快不快度': y_test_all_valence,
    '予測した値': prediction_all_valence
})

df_arousal = pd.DataFrame({
    '実際の覚醒度': y_test_all_arousal,
    '予測した値': prediction_all_arousal
})

# CSVファイルとして保存
df_valence.to_csv(f'/Users/sudaxin/Desktop/KOBA/prediction_results/{person}_n_splits_{n_splits}_valence_results.csv', index=False)
df_arousal.to_csv(f'/Users/sudaxin/Desktop/KOBA/prediction_results/{person}_n_splits_{n_splits}_arousal_results.csv', index=False)


plt.figure(figsize=(10, 6))
colors = np.where((y_test_all_valence * prediction_all_valence) < 0, 'red', 'blue')
plt.scatter(y_test_all_valence, prediction_all_valence, color=colors, edgecolor='k', alpha=0.7)
plt.plot([-1,1], [-1,1], linestyle='-')
plt.axhline(0, color='black', linewidth=1)  # x軸を追加
plt.axvline(0, color='black', linewidth=1)  # y軸を追加
plt.xlabel('Actual Valence')
plt.ylabel('Predicted Valence')
plt.fill([0, 1, 1, 0], [0, 0, 1, 1], 'skyblue', alpha=0.5)
plt.fill([0, -1, -1, 0], [0, 0, -1, -1], 'skyblue', alpha=0.5)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.title(f'Residual for Valence')

# 現在の日時を取得
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d %H:%M:%S")

# テキストを配置
plt.text(0.95, 0.05, f'Created: {date_string}',horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
plt.savefig(f'/Users/sudaxin/Desktop/KOBA/residuals/valence_residual_plot_fold_all.png')
plt.close()

plt.figure(figsize=(10, 6))
colors = np.where((y_test_all_arousal * prediction_all_arousal) < 0, 'red', 'blue')
plt.scatter(y_test_all_arousal, prediction_all_arousal, color=colors, edgecolor='k', alpha=0.7)
plt.plot([-1,1], [-1,1], linestyle='-')
plt.axhline(0, color='black', linewidth=1)  # x軸を追加
plt.axvline(0, color='black', linewidth=1)  # y軸を追加
plt.xlabel('Actual Arousal')
plt.ylabel('Predicted Arousal')
plt.fill([0, 1, 1, 0], [0, 0, 1, 1], 'skyblue', alpha=0.5)
plt.fill([0, -1, -1, 0], [0, 0, -1, -1], 'skyblue', alpha=0.5)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.title(f'Residual for Arousal')
# 現在の日時を取得
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d %H:%M:%S")

# テキストを配置
plt.text(0.95, 0.05, f'Created: {date_string}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=8, alpha=0.7)
plt.savefig(f'/Users/sudaxin/Desktop/KOBA/residuals/arousal_residual_plot_fold_all.png')
plt.close()

# Calculate average classification reports
def average_classification_reports(reports):
    avg_report = {}
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):
            avg_report[key] = average_classification_reports([report[key] for report in reports])
        else:
            avg_report[key] = np.mean([report[key] for report in reports])
    return avg_report

def evaluate_regression(y_true, y_pred):
    # 評価指標の計算
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2
    }

# Valenceの評価
print("Valenceの評価:")
valence_results = evaluate_regression(y_test_all_valence, prediction_all_valence)
for metric, value in valence_results.items():
    print(f"{metric}: {value:.4f}")

# Arousalの評価
print("\nArousalの評価:")
arousal_results = evaluate_regression(y_test_all_arousal, prediction_all_arousal)
for metric, value in arousal_results.items():
    print(f"{metric}: {value:.4f}")

# 結果をデータフレームに変換
results_df = pd.DataFrame({
    "Metric": valence_results.keys(),
    "Valence": valence_results.values(),
    "Arousal": arousal_results.values()
})

# CSVファイルとして保存
evaluation_filename = f'/Users/sudaxin/Desktop/KOBA/evaluation/{person}_n_splits_{n_splits}_evaluation.csv'
results_df.to_csv(evaluation_filename, index=False)

"""
valence_report_avg = average_classification_reports(valence_reports)
arousal_report_avg = average_classification_reports(arousal_reports)

valence_report_avg_df = pd.DataFrame(valence_report_avg).transpose()
arousal_report_avg_df = pd.DataFrame(arousal_report_avg).transpose()

valence_report_avg_df.to_csv('./report_data/valence_classification_report_avg.csv')
arousal_report_avg_df.to_csv('./report_data/arousal_classification_report_avg.csv')
"""
# Save the model
best_model.save(f'/Users/sudaxin/Desktop/KOBA/models/best_model_{person}.keras', save_format='keras')
print(f"最良のモデルが保存されました: /Users/sudaxin/Desktop/KOBA/best_model_{person}.keras")
print(f"Best model from fold {best_fold}")
print(f"Best Valence MSE: {best_valence_mse:.4f}")
print(f"Best Arousal MSE: {best_arousal_mse:.4f}")
print(f"Best Total MSE: {best_total_mse:.4f}")

"""
# Calculate and print average AUC scores
average_valence_auc = np.mean(valence_auc_scores)
average_arousal_auc = np.mean(arousal_auc_scores)
print(f'Average AUC for Valence: {average_valence_auc}')
print(f'Average AUC for Arousal: {average_arousal_auc}')
"""



print("Cross-validation completed and results saved.")