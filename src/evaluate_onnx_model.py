import os
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from matplotlib.backends.backend_pdf import PdfPages

# ============== PATHS ==============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "../models/liveness_model.onnx")
REPORT_PDF_PATH = os.path.join(BASE_DIR, "../reports/liveness_onnx_report.pdf")
os.makedirs(os.path.join(BASE_DIR, "../reports"), exist_ok=True)

# ============== DATA GENERATOR ==============
img_size = (64, 64)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = os.path.join(DATASET_PATH, "test")  # or "val"
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

y_true = test_gen.classes
n_samples = len(y_true)

# ============== ONNX RUNTIME SESSION ==============
sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

print("✅ Loaded ONNX model:", ONNX_MODEL_PATH)
print("Input name:", input_name, "Output name:", output_name)

# ============== PREDICTIONS USING ONNX ==============
y_prob_list = []

# Loop over generator without infinite loop
steps = int(np.ceil(n_samples / batch_size))
i = 0
for batch_x, _ in test_gen:
    # batch_x already rescaled
    preds = sess.run([output_name], {input_name: batch_x.astype(np.float32)})[0].ravel()
    y_prob_list.append(preds)
    i += 1
    if i >= steps:
        break

y_prob = np.concatenate(y_prob_list)[:n_samples]
y_pred = (y_prob >= 0.5).astype(int)

# ============== METRICS ==============
print("\n📌 ONNX Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

cm = confusion_matrix(y_true, y_pred)
print("\n📌 ONNX Confusion Matrix:\n", cm)

try:
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\n📌 ONNX ROC-AUC: {roc_auc:.4f}")
except ValueError:
    print("\n⚠️ ONNX ROC-AUC could not be computed (need both classes).")

# ============== PLOTS & PDF ==============
with PdfPages(REPORT_PDF_PATH) as pdf:
    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation='nearest')
    ax_cm.figure.colorbar(im, ax=ax_cm)
    classes = list(test_gen.class_indices.keys())
    ax_cm.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='ONNX Confusion Matrix'
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

    # ROC Curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ONNX ROC Curve')
        ax_roc.legend(loc="lower right")
        pdf.savefig(fig_roc)
        plt.close(fig_roc)
    except Exception as e:
        print("⚠️ Could not plot ONNX ROC curve:", e)

    # Summary page
    fig_text = plt.figure()
    summary = (
        f"ONNX Liveness Model Evaluation\n\n"
        f"Samples: {len(y_true)}\n"
        f"ROC-AUC: {roc_auc:.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    plt.text(0.01, 0.99, summary, va='top', wrap=True)
    plt.axis('off')
    pdf.savefig(fig_text)
    plt.close(fig_text)

print(f"\n✅ ONNX PDF report saved at: {REPORT_PDF_PATH}")
