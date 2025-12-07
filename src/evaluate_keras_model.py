# import os
# import numpy as np
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     roc_curve
# )

# from matplotlib.backends.backend_pdf import PdfPages

# # ================== PATHS ==================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
# MODEL_PATH = os.path.join(BASE_DIR, "../models/liveness_model.keras")
# REPORT_PDF_PATH = os.path.join(BASE_DIR, "../reports/liveness_report.pdf")
# os.makedirs(os.path.join(BASE_DIR, "../reports"), exist_ok=True)

# # ================== LOAD MODEL ==================
# model = load_model(MODEL_PATH)
# print("✅ Loaded Keras model from:", MODEL_PATH)

# # ================== DATA GENERATORS ==================
# img_size = (64, 64)
# batch_size = 32

# test_datagen = ImageDataGenerator(rescale=1./255)

# # If you don't have a 'test' folder, use 'val' here:
# test_dir = os.path.join(DATASET_PATH, "test")   # or "val"
# test_gen = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False   # VERY IMPORTANT
# )

# # ================== PREDICTIONS ==================
# # y_true from generator
# y_true = test_gen.classes   # 0/1 labels

# # raw probabilities from model
# y_prob = model.predict(test_gen).ravel()

# # binary predictions using threshold 0.5
# y_pred = (y_prob >= 0.5).astype(int)

# # ================== METRICS ==================
# print("\n📌 Classification Report (Precision, Recall, F1):")
# print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# print("\n📌 Confusion Matrix:\n", cm)

# # ROC-AUC
# try:
#     roc_auc = roc_auc_score(y_true, y_prob)
#     print(f"\n📌 ROC-AUC: {roc_auc:.4f}")
# except ValueError:
#     print("\n⚠️ ROC-AUC could not be computed (need both classes present).")

# # ================== PLOTS & PDF REPORT ==================
# with PdfPages(REPORT_PDF_PATH) as pdf:
#     # ---------- 1. Confusion Matrix ----------
#     fig_cm, ax_cm = plt.subplots()
#     im = ax_cm.imshow(cm, interpolation='nearest')
#     ax_cm.figure.colorbar(im, ax=ax_cm)
#     classes = list(test_gen.class_indices.keys())
#     ax_cm.set(
#         xticks=np.arange(len(classes)),
#         yticks=np.arange(len(classes)),
#         xticklabels=classes,
#         yticklabels=classes,
#         ylabel='True label',
#         xlabel='Predicted label',
#         title='Confusion Matrix'
#     )
#     # label each cell
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax_cm.text(j, i, format(cm[i, j], 'd'),
#                        ha="center", va="center",
#                        color="white" if cm[i, j] > thresh else "black")
#     fig_cm.tight_layout()
#     pdf.savefig(fig_cm)
#     plt.close(fig_cm)

#     # ---------- 2. ROC Curve ----------
#     try:
#         fpr, tpr, thresholds = roc_curve(y_true, y_prob)
#         fig_roc, ax_roc = plt.subplots()
#         ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
#         ax_roc.plot([0, 1], [0, 1], linestyle='--')
#         ax_roc.set_xlim([0.0, 1.0])
#         ax_roc.set_ylim([0.0, 1.05])
#         ax_roc.set_xlabel('False Positive Rate')
#         ax_roc.set_ylabel('True Positive Rate')
#         ax_roc.set_title('Receiver Operating Characteristic')
#         ax_roc.legend(loc="lower right")
#         pdf.savefig(fig_roc)
#         plt.close(fig_roc)
#     except Exception as e:
#         print("⚠️ Could not plot ROC curve:", e)

#     # ---------- 3. Training Curves (if you saved history) ----------
#     # If you saved history using `history.history` to a file, load it here.
#     # Example if you saved as npy:
#     history_path = os.path.join(BASE_DIR, "../models/train_history.npy")
#     if os.path.exists(history_path):
#         history = np.load(history_path, allow_pickle=True).item()
#         acc = history.get('accuracy', [])
#         val_acc = history.get('val_accuracy', [])
#         loss = history.get('loss', [])
#         val_loss = history.get('val_loss', [])
#         epochs = range(1, len(acc) + 1)

#         # Accuracy plot
#         fig_acc, ax_acc = plt.subplots()
#         ax_acc.plot(epochs, acc, label='Train Accuracy')
#         ax_acc.plot(epochs, val_acc, label='Val Accuracy')
#         ax_acc.set_title('Training & Validation Accuracy')
#         ax_acc.set_xlabel('Epoch')
#         ax_acc.set_ylabel('Accuracy')
#         ax_acc.legend()
#         pdf.savefig(fig_acc)
#         plt.close(fig_acc)

#         # Loss plot
#         fig_loss, ax_loss = plt.subplots()
#         ax_loss.plot(epochs, loss, label='Train Loss')
#         ax_loss.plot(epochs, val_loss, label='Val Loss')
#         ax_loss.set_title('Training & Validation Loss')
#         ax_loss.set_xlabel('Epoch')
#         ax_loss.set_ylabel('Loss')
#         ax_loss.legend()
#         pdf.savefig(fig_loss)
#         plt.close(fig_loss)

#     # ---------- 4. Summary Page ----------
#     fig_text = plt.figure()
#     summary = (
#         f"Liveness Model Evaluation Report\n\n"
#         f"Samples: {len(y_true)}\n"
#         f"ROC-AUC: {roc_auc:.4f}\n\n"
#         f"Confusion Matrix:\n{cm}\n"
#     )
#     plt.text(0.01, 0.99, summary, va='top', wrap=True)
#     plt.axis('off')
#     pdf.savefig(fig_text)
#     plt.close(fig_text)

# print(f"\n✅ PDF report saved at: {REPORT_PDF_PATH}")


import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    precision_recall_curve,
    accuracy_score
)

from matplotlib.backends.backend_pdf import PdfPages

# ================== PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
MODEL_PATH = os.path.join(BASE_DIR, "../models/liveness_model.keras")
REPORT_PDF_PATH = os.path.join(BASE_DIR, "../reports/liveness_report.pdf")
os.makedirs(os.path.join(BASE_DIR, "../reports"), exist_ok=True)

# ================== LOAD MODEL ==================
model = load_model(MODEL_PATH)
print("✅ Loaded Keras model from:", MODEL_PATH)

# ================== DATA GENERATORS ==================
img_size = (64, 64)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

# If you don't have a 'test' folder, use 'val' here:
test_dir = os.path.join(DATASET_PATH, "test")   # or "val"
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False   # VERY IMPORTANT
)

# ================== PREDICTIONS ==================
# y_true from generator
y_true = test_gen.classes   # 0/1 labels
class_names = list(test_gen.class_indices.keys())

# raw probabilities from model
y_prob = model.predict(test_gen).ravel()

# binary predictions using threshold 0.5
y_pred = (y_prob >= 0.5).astype(int)

# ================== METRICS ==================
print("\n📌 Classification Report (Precision, Recall, F1):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\n📌 Confusion Matrix:\n", cm)

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\n📌 ROC-AUC: {roc_auc:.4f}")
except ValueError:
    roc_auc = float("nan")
    print("\n⚠️ ROC-AUC could not be computed (need both classes present).")

# Precision, Recall, F1 per class
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1], zero_division=0
)
accuracy = accuracy_score(y_true, y_pred)

print(f"\n📌 Overall Accuracy: {accuracy * 100:.2f}%")

# ================== PLOTS & PDF REPORT ==================
with PdfPages(REPORT_PDF_PATH) as pdf:
    # ---------- 1. Confusion Matrix ----------
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation='nearest')
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    fig_cm.tight_layout()
    pdf.savefig(fig_cm)
    plt.close(fig_cm)

    # ---------- 2. ROC Curve ----------
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        pdf.savefig(fig_roc)
        plt.close(fig_roc)
    except Exception as e:
        print("⚠️ Could not plot ROC curve:", e)

    # ---------- 3. Precision-Recall Curve ----------
    try:
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall_curve, precision_curve)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        pdf.savefig(fig_pr)
        plt.close(fig_pr)
    except Exception as e:
        print("⚠️ Could not plot Precision-Recall curve:", e)

    # ---------- 4. Metrics Bar Chart (Precision, Recall, F1 per class) ----------
    fig_metrics, ax_metrics = plt.subplots()
    x = np.arange(len(class_names))
    width = 0.25

    ax_metrics.bar(x - width, precisions, width, label='Precision')
    ax_metrics.bar(x, recalls, width, label='Recall')
    ax_metrics.bar(x + width, f1s, width, label='F1-score')

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(class_names)
    ax_metrics.set_ylim([0.0, 1.05])
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Per-Class Metrics')
    ax_metrics.legend()
    pdf.savefig(fig_metrics)
    plt.close(fig_metrics)

    # ---------- 5. Probability Histogram ----------
    fig_hist, ax_hist = plt.subplots()
    # separate probabilities by true class
    probs_class0 = y_prob[y_true == 0]
    probs_class1 = y_prob[y_true == 1]

    ax_hist.hist(probs_class0, bins=20, alpha=0.7, label=f"{class_names[0]} (true)")
    ax_hist.hist(probs_class1, bins=20, alpha=0.7, label=f"{class_names[1]} (true)")
    ax_hist.set_xlabel('Predicted Probability of Class 1 (e.g. Live)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Predicted Probability Distribution by True Class')
    ax_hist.legend()
    pdf.savefig(fig_hist)
    plt.close(fig_hist)

    # ---------- 6. True vs Predicted Class Counts ----------
    fig_counts, ax_counts = plt.subplots()
    true_counts = np.bincount(y_true, minlength=2)
    pred_counts = np.bincount(y_pred, minlength=2)

    x = np.arange(len(class_names))
    width = 0.35

    ax_counts.bar(x - width/2, true_counts, width, label='True Count')
    ax_counts.bar(x + width/2, pred_counts, width, label='Predicted Count')

    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(class_names)
    ax_counts.set_ylabel('Number of Samples')
    ax_counts.set_title('True vs Predicted Sample Counts')
    ax_counts.legend()
    pdf.savefig(fig_counts)
    plt.close(fig_counts)

    # ---------- 7. Training Curves (if you saved history) ----------
    history_path = os.path.join(BASE_DIR, "../models/train_history.npy")
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        acc = history.get('accuracy', [])
        val_acc = history.get('val_accuracy', [])
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(acc) + 1)

        # Accuracy plot
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(epochs, acc, label='Train Accuracy')
        ax_acc.plot(epochs, val_acc, label='Val Accuracy')
        ax_acc.set_title('Training & Validation Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        pdf.savefig(fig_acc)
        plt.close(fig_acc)

        # Loss plot
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(epochs, loss, label='Train Loss')
        ax_loss.plot(epochs, val_loss, label='Val Loss')
        ax_loss.set_title('Training & Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        pdf.savefig(fig_loss)
        plt.close(fig_loss)

    # ---------- 8. Summary Page ----------
    fig_text = plt.figure()
    summary_lines = [
        "Liveness Model Evaluation Report",
        "",
        f"Samples: {len(y_true)}",
        f"Overall Accuracy: {accuracy * 100:.2f}%",
        f"ROC-AUC: {roc_auc:.4f}",
        "",
        "Per-Class Metrics:",
    ]
    for i, cls in enumerate(class_names):
        summary_lines.append(
            f"  {cls}: "
            f"Precision={precisions[i]:.3f}, "
            f"Recall={recalls[i]:.3f}, "
            f"F1={f1s[i]:.3f}, "
            f"Support={supports[i]}"
        )

    summary_lines.append("")
    summary_lines.append("Confusion Matrix:")
    summary_lines.append(str(cm))

    summary = "\n".join(summary_lines)
    plt.text(0.01, 0.99, summary, va='top', wrap=True)
    plt.axis('off')
    pdf.savefig(fig_text)
    plt.close(fig_text)

print(f"\n✅ PDF report saved at: {REPORT_PDF_PATH}")
