import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Excel'den veriyi yükle
veri_seti = pd.read_excel("verieskisehir.xlsx")

# Bağımsız ve bağımlı değişkenleri ayır
X = veri_seti.iloc[:, 0].values.reshape(-1, 1)  # Yıl
y = veri_seti.iloc[:, 1].values.reshape(-1, 1)  # Yolcu sayısı

# SVR modelini oluştur
svr_modeli = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10000, gamma=0.1))
svr_modeli.fit(X, y.ravel())

# Gelecek yıllar için tahmin yapmak için bir yıl dizisi oluştur
gelecek_gunler = np.arange(1, 366).reshape(-1, 1)

# Gelecek yıllar için yolcu sayısı tahminleri yap
gelecek_tahminler = svr_modeli.predict(gelecek_gunler)

# Gelecek yıllar ve tahmin edilen yolcu sayılarını ekrana yazdır
print("Gelecek Gunler ve Tahmin Edilen Yolcu Sayıları:")
for gunler, tahmin in zip(gelecek_gunler, gelecek_tahminler):
    print(f"{int(gunler)}: {tahmin:.2f} yolcu")

# Tahminleri ve gerçek değerleri görselleştir
import matplotlib.pyplot as plt

plt.scatter(X, y, color='red', label='Gerçek Veriler')
plt.plot(X, svr_modeli.predict(X), color='blue', label='SVR Tahmin')
plt.plot(gelecek_gunler, gelecek_tahminler, color='green', linestyle='--', label='Gelecek Tahmin')
plt.xlabel('Gunler')
plt.ylabel('Yolcu Sayısı')
plt.title('SVR ile Yolcu Sayısı Tahmini')
plt.legend()
plt.show()
