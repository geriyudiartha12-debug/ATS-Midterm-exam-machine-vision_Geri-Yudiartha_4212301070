# ==============================================
# main_optimized_13k.py
# Versi Cepat + Akurat (13K Data, HOG + PCA + SVM RBF + GridSearch + 5-Fold)
# ==============================================
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import string
import warnings
from tqdm import tqdm

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning

# ==============================
# 1️⃣ Setup Awal
# ==============================
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data_path = r"D:\Midterm\emnist-letters-train.csv"
output_dir = r"D:\Midterm\output"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 2️⃣ Load Dataset
# ==============================
print("📂 Memuat dataset...")
data = pd.read_csv(data_path, header=None)
print(f"Total data awal: {len(data)}")

n_classes = 26
samples_per_class = 500  # 26 huruf × 500 = 13.000 sampel total
sampled_data = []

for label in range(1, n_classes + 1):
    subset = data[data[0] == label].sample(samples_per_class, random_state=42)
    sampled_data.append(subset)

sampled_data = pd.concat(sampled_data)
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Total data digunakan: {len(sampled_data)} (26 huruf × 500)")

# ==============================
# 3️⃣ Pisahkan Fitur & Label
# ==============================
X = sampled_data.iloc[:, 1:].values / 255.0
y = sampled_data.iloc[:, 0].values

# ==============================
# 4️⃣ Ekstraksi Fitur HOG
# ==============================
print("⚙ Ekstraksi fitur HOG...")

hog_features = []
for img in tqdm(X, desc="HOG Processing"):
    img = np.reshape(img, (28, 28))
    img = np.transpose(img)
    feature = hog(img,
                  orientations=12,
                  pixels_per_cell=(6, 6),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys',
                  transform_sqrt=True)
    hog_features.append(feature)

hog_features = np.array(hog_features)
print(f"✅ Total fitur HOG: {hog_features.shape[1]}")

# ==============================
# 5️⃣ PCA (Reduksi Dimensi)
# ==============================
print("🔍 Melakukan reduksi dimensi dengan PCA...")

pca = PCA(n_components=0.95, random_state=42)  # pertahankan 95% variansi
X_pca = pca.fit_transform(hog_features)

print(f"✅ Dimensi setelah PCA: {X_pca.shape[1]} (dari {hog_features.shape[1]})")

# ==============================
# 6️⃣ SVM + GridSearchCV (Optimasi Parameter)
# ==============================
print("🚀 Optimasi parameter SVM (Grid Search)...")

param_grid = {
    'C': [1, 5, 10, 20],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'kernel': ['rbf']
}

svm = SVC(random_state=42)

grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
grid_search.fit(X_pca, y)
best_svm = grid_search.best_estimator_
end_time = time.time()

print(f"\n🏆 Parameter terbaik: {grid_search.best_params_}")
print(f"🕐 Waktu training GridSearch: {(end_time - start_time)/60:.2f} menit")

# ==============================
# 7️⃣ Evaluasi dengan Stratified K-Fold (5-Fold)
# ==============================
print("\n🧠 Evaluasi model terbaik (5-Fold Cross Validation)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(best_svm, X_pca, y, cv=cv, n_jobs=-1)

acc = accuracy_score(y, y_pred)
print(f"\n🎯 Akurasi akhir (5-Fold): {acc * 100:.2f}%")

print("\n🧪 Menjalankan LOOCV FULL 13.000 data...")
print("⚠ Proses ini sangat berat. Bisa memakan waktu berjam-jam hingga berhari-hari.")
print("   Pastikan komputer memiliki RAM besar dan tidak sedang dipakai pekerjaan lain.")

loo = LeaveOneOut()

y_pred_loo = []
start_loo = time.time()

total_samples = len(X_pca)
progress = tqdm(total=total_samples, desc="LOOCV Full 13K")

for train_idx, test_idx in loo.split(X_pca):
    # Train ulang SVM dengan 12.999 data
    best_svm.fit(X_pca[train_idx], y[train_idx])

    # Prediksi 1 data
    pred = best_svm.predict(X_pca[test_idx])
    y_pred_loo.append(pred[0])

    progress.update(1)

progress.close()
end_loo = time.time()

# Hitung akurasi LOOCV
acc_loocv = accuracy_score(y, y_pred_loo)
print(f"\n📈 Akurasi LOOCV Full: {acc_loocv * 100:.2f}%")
print(f"⏱ Total Waktu LOOCV: {(end_loo - start_loo)/3600:.2f} jam")

# Simpan hasil LOOCV
loocv_full_path = os.path.join(output_dir, "loocv_full_13k_result.txt")
with open(loocv_full_path, "w") as f:
    f.write("=== LOOCV Full 13.000 Sampel ===\n")
    f.write(f"Akurasi LOOCV Full: {acc_loocv * 100:.2f}%\n")
    f.write(f"Total Waktu: {(end_loo - start_loo)/3600:.2f} jam\n")

print(f"📁 Hasil LOOCV Full disimpan di: {loocv_full_path}")
# ==============================
# 8️⃣ Output Huruf A–Z + Report
# ==============================
labels = list(string.ascii_uppercase)
y_true_letters = [labels[i - 1] for i in y]
y_pred_letters = [labels[i - 1] for i in y_pred]

report = classification_report(
    y_true_letters,
    y_pred_letters,
    labels=labels,
    target_names=labels,
    digits=3
)

print("\n📊 Classification Report:\n", report)

report_path = os.path.join(output_dir, "classification_report_13k.txt")
with open(report_path, "w") as f:
    f.write("=== Hasil Evaluasi EMNIST (HOG + PCA + SVM RBF + GridSearch + 5-Fold) ===\n")
    f.write(f"Akurasi: {acc * 100:.2f}%\n\n")
    f.write(report)

# ==============================
# 9️⃣ Confusion Matrix
# ==============================
cm = confusion_matrix(y_true_letters, y_pred_letters, labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r',
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 8})
plt.title(f'Confusion Matrix (Akurasi: {acc * 100:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('True')

cm_path = os.path.join(output_dir, "confusion_matrix_13k.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📁 Hasil disimpan di: {output_dir}")
print(f"- Confusion Matrix: {cm_path}")
print(f"- Classification Report: {report_path}")
print("\n✅ Program selesai sukses!")