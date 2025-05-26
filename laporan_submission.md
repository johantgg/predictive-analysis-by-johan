# Laporan Proyek Machine Learning - [Johan Adrian Sitanggang]

## Domain Proyek

Kartu skor kredit adalah sistem yang digunakan industri keuangan untuk menilai risiko kredit dari seorang calon pemohon. Model tradisional seperti regresi logistik umum digunakan karena transparansinya, namun metode pembelajaran mesin yang lebih kompleks seperti Random Forest dan XGBoost kini mulai diadopsi karena potensi prediktif yang lebih tinggi.

Masalah utama yang diangkat adalah membangun model untuk mengklasifikasikan apakah pelamar termasuk klien 'baik' atau 'buruk'. Definisi eksplisit dari label ini tidak tersedia, sehingga harus diturunkan dari data historis (`credit_record.csv`). Tantangan tambahan mencakup ketidakseimbangan kelas dan data yang tidak lengkap.

## Business Understanding

### Problem Statement

1. Bagaimana membangun model ML yang efektif untuk memprediksi risiko kredit?
2. Teknik preprocessing apa yang paling sesuai?
3. Model mana yang memberikan hasil terbaik?

### Goals

* Mengembangkan sistem klasifikasi berbasis machine learning yang dapat mengidentifikasi pemohon dengan risiko gagal bayar tinggi.
* Memberikan rekomendasi keputusan kredit yang lebih akurat dan cepat untuk mendukung proses bisnis lembaga keuangan.
* Mengurangi tingkat kerugian akibat pemberian kredit bermasalah dengan memanfaatkan data historis dan riwayat kredit pemohon.

## Data Understanding

![app_missing_heatmap](https://github.com/user-attachments/assets/9fbe4e8b-1444-4091-94f5-a9dfc4e6e8e9)

### Jumlah Data
- **application_record.csv**: 438.557 baris, 18 kolom. Berisi informasi pribadi dan demografi pemohon.
- **credit_record.csv**: 1.048.575 baris, 3 kolom. Berisi histori pembayaran kredit bulanan.

### Kondisi Data
- **application_record.csv**:
  - Missing value: Kolom `OCCUPATION_TYPE` memiliki 134.203 nilai kosong.
  - Duplikat: Tidak ditemukan data duplikat.
- **credit_record.csv**:
  - Missing value: Tidak ada nilai kosong.
  - Duplikat: Tidak ditemukan data duplikat.

### Tautan Sumber Data
Dataset dapat diakses di: [Kaggle Credit Card Approval Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

application_record.csv (Total 18 fitur):
| No | Nama Fitur            | Tipe    | Digunakan? | Alasan                                                              |
| -- | --------------------- | ------- | ---------- | ------------------------------------------------------------------- |
| 1  | ID                    | int64   | ✅          | Untuk join dengan `credit_record.csv`                               |
| 2  | CODE\_GENDER          | object  | ✅          | Informasi demografi yang relevan                                    |
| 3  | FLAG\_OWN\_CAR        | object  | ✅          | Menunjukkan kestabilan finansial                                    |
| 4  | FLAG\_OWN\_REALTY     | object  | ✅          | Menunjukkan kepemilikan aset                                        |
| 5  | CNT\_CHILDREN         | int64   | ✅          | Indikasi beban tanggungan finansial                                 |
| 6  | AMT\_INCOME\_TOTAL    | float64 | ✅          | Fitur utama: pendapatan                                             |
| 7  | NAME\_INCOME\_TYPE    | object  | ✅          | Mengklasifikasi sumber penghasilan                                  |
| 8  | NAME\_EDUCATION\_TYPE | object  | ✅          | Indikasi tingkat pendidikan (berkorelasi dengan kemampuan membayar) |
| 9  | NAME\_FAMILY\_STATUS  | object  | ✅          | Status pernikahan bisa berpengaruh pada stabilitas ekonomi          |
| 10 | NAME\_HOUSING\_TYPE   | object  | ✅          | Menunjukkan pola tempat tinggal                                     |
| 11 | DAYS\_BIRTH           | int64   | ✅          | Untuk menentukan usia                                               |
| 12 | DAYS\_EMPLOYED        | int64   | ✅          | Menunjukkan stabilitas pekerjaan                                    |
| 13 | FLAG\_MOBIL           | int64   | ❌          | Semua nilainya 1, tidak informatif                                  |
| 14 | FLAG\_WORK\_PHONE     | int64   | ✅          | Kemungkinan menunjukkan status pekerjaan                            |
| 15 | FLAG\_PHONE           | int64   | ✅          | Informasi tambahan komunikasi                                       |
| 16 | FLAG\_EMAIL           | int64   | ✅          | Sarana komunikasi, digunakan                                        |
| 17 | OCCUPATION\_TYPE      | object  | ❌          | Terlalu banyak missing value (>30%), dihapus                        |
| 18 | CNT\_FAM\_MEMBERS     | float64 | ✅          | Beban keluarga, digunakan                                           |

credit_record.csv (Total 3 fitur):
| No | Nama Fitur      | Tipe   | Digunakan?  | Alasan                                                                        |
| -- | --------------- | ------ | ----------- | ----------------------------------------------------------------------------  |
| 1  | ID              | int64  | ✅           | Untuk join ke `application_record.csv`                                       |
| 2  | MONTHS\_BALANCE | int64  | ✅ (turunan) | Digunakan untuk membuat urutan waktu riwayat kredit                          |
| 3  | STATUS          | object | ✅ (diolah)  | Diubah menjadi label target (0 = lancar, 1 = gagal bayar) berdasarkan aturan |

🧩 Penjelasan Transformasi Penting
STATUS diolah dengan aturan:

'C', 'X', 0, 1 → 0 (tidak gagal bayar)

2 ke atas → 1 (gagal bayar)

Label akhir per pemohon ditentukan dari nilai maksimum STATUS per ID.



## Data Preparation
![data_preparation](https://github.com/user-attachments/assets/f062272f-e10f-48ac-96c3-075e4b007709)

Langkah-langkah persiapan data dilakukan secara terstruktur untuk memastikan data siap digunakan dalam proses pelatihan model machine learning. Berikut tahapan yang dilakukan:

1. **Menghapus Duplikat**  
   Tidak ditemukan data duplikat dalam kolom `ID`.

2. **Menghapus Kolom dengan Missing Value Besar**  
   Kolom `OCCUPATION_TYPE` dihapus karena memiliki >30% nilai kosong.

3. **Encoding Fitur Kategorikal**  
   Dilakukan menggunakan `LabelEncoder`.
   
4. **Mengatasi Outlier dengan IQR**  
   Diterapkan pada `CNT_CHILDREN`, `AMT_INCOME_TOTAL`, dan `CNT_FAM_MEMBERS`.

5. **Pengolahan Kolom `STATUS` pada `credit_record.csv`**  
   Diubah menjadi label biner (0: tidak gagal bayar, 1: gagal bayar) berdasarkan nilai maksimum per ID.

6. **Join Dataset**  
   Digabung berdasarkan kolom `ID`.

7. **Menangani Ketidakseimbangan Kelas**  
   Diterapkan SMOTE untuk menyeimbangkan kelas target.

8. **Split Data**  
   Data dibagi dengan rasio 70% training dan 30% testing.

9. **Normalisasi Fitur**  
   Menggunakan MinMaxScaler.



## Modeling

## Model Development

Dalam tahap ini, dilakukan pelatihan dan evaluasi beberapa model machine learning. Semua model dilatih menggunakan data hasil oversampling dari SMOTE.

### Model yang Digunakan:

1. **Logistic Regression**
   Logistic Regression adalah model linier untuk klasifikasi biner yang menggunakan fungsi sigmoid.
   code : 
   LogisticRegression(random_state=42)

   Model ini digunakan tanpa tuning tambahan karena menjadi baseline awal.


2. **K-Nearest Neighbors (KNN)**
   KNN mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekatnya. Dalam proyek ini, digunakan nilai k=5 (default). Algoritma ini non-parametrik dan cocok untuk dataset kecil atau yang telah dinormalisasi.
   Parameter yang digunakan:
   KNeighborsClassifier(n_neighbors=5)

   Parameter n_neighbors mengatur jumlah tetangga terdekat. Model ini digunakan dengan nilai default.

3. **Support Vector Machine (SVM)**
   SVM bekerja dengan mencari hyperplane terbaik yang memisahkan kelas dalam data. Kernel yang digunakan adalah RBF (default). SVM cocok digunakan pada data dengan dimensi tinggi.
   Parameter yang digunakan:
   SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

   * kernel='rbf': Kernel non-linear (default).
   * C: Kontrol regularisasi.
   * gamma: Skala pengaruh satu titik data.

4. **Decision Tree**
   Model pohon keputusan memisahkan data berdasarkan aturan tertentu untuk membentuk struktur pohon. Setiap node memisahkan data berdasarkan fitur yang paling informatif. Model ini digunakan dengan parameter default.
   Parameter yang digunakan:
   DecisionTreeClassifier(random_state=42, max_depth=None)

   Model ini digunakan dengan parameter default. Dapat dikembangkan dengan tuning max_depth dan min_samples_split

5. **Random Forest**
   Random Forest adalah ensemble dari banyak pohon keputusan yang dibangun dari subset acak data. Hasil klasifikasi ditentukan melalui voting mayoritas. Model ini tahan terhadap overfitting dan memberikan hasil yang stabil.
   Parameter:
   RandomForestClassifier(n_estimators=100, random_state=42)
   * n_estimators: Jumlah pohon dalam hutan.

6. **XGBoost (Extreme Gradient Boosting)**
   XGBoost adalah metode boosting berbasis pohon yang cepat dan efisien. Model dibangun secara bertahap, di mana setiap iterasi bertujuan untuk mengoreksi kesalahan dari model sebelumnya. XGBoost menggunakan regularisasi untuk menghindari overfitting dan dirancang untuk performa tinggi.
   Parameter default:
   XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

   Model digunakan dengan parameter default untuk baseline awal. Tuning akan menjadi langkah pengembangan selanjutnya jika diperlukan peningkatan performa.

## Evaluation



### Metrik Evaluasi

Model dievaluasi berdasarkan:
* **Accuracy**: Total prediksi benar.
* **Precision**: Ketepatan dalam memprediksi kelas positif.
* **Recall**: Kemampuan model menangkap kelas positif.
* **F1-score**: Harmonis antara precision dan recall.

### Hasil Evaluasi Model

## Kesimpulan Evaluasi Model

Setelah melakukan evaluasi terhadap enam model machine learning, berikut ringkasan performa utama:

| Model               | Accuracy | Precision | Recall | F1-Score | Catatan Singkat                                  |
|---------------------|----------|-----------|--------|----------|--------------------------------------------------|
| Logistic Regression | 0.51     | 0.51      | 0.51   | 0.50     | Baseline kurang baik, kurang menangani minoritas |
| KNN                 | 0.78     | 0.84      | 0.78   | 0.77     | Performa rendah, recall kelas 1 buruk            |
| SVM                 | 0.86     | 0.87      | 0.86   | 0.86     | Cukup baik, relatif seimbang                     |
| Decision Tree       | 0.86     | 0.89      | 0.86   | 0.86     | Akurat, rawan overfitting                        |
| Random Forest       | 0.85     | 0.88      | 0.85   | 0.85     | Stabil, kuat untuk generalisasi                  |
| **XGBoost**         | **0.96** | **0.96**  | **0.96** | **0.96** | **Terbaik secara keseluruhan**                 |




## 🎯 Model Terbaik: XGBoost

Model **XGBoost** dipilih sebagai model terbaik karena menunjukkan performa tinggi pada seluruh metrik penting, khususnya:

- **Precision tinggi** pada kelas gagal bayar (penting untuk menghindari risiko).
- **Recall memadai**, artinya model tidak melewatkan terlalu banyak nasabah bermasalah.
- **Macro F1-score sebesar 0.96**, paling seimbang dari seluruh model.
- Model robust, scalable, dan mendukung interpretabilitas (dapat melihat feature importance).






## 💼 Rekomendasi Implementasi

- **Gunakan XGBoost** untuk membangun sistem pendukung keputusan dalam proses evaluasi aplikasi kredit.
- Integrasikan model dengan sistem backend lembaga keuangan agar bisa mengklasifikasikan pengajuan kredit secara otomatis.
- Lakukan monitoring berkala dan **tuning ulang** model berdasarkan data baru untuk menjaga performa.

---



### Hubungan dengan Business Understanding

* Model berhasil menjawab semua *problem statement*.
* Model mencapai goal utama: klasifikasi yang akurat terhadap risiko kredit.
* Solusi yang dirancang berdampak karena bisa digunakan untuk menghindari pemberian kredit ke pemohon dengan risiko tinggi secara efisien.

## Kesimpulan



![eval metrics](https://github.com/user-attachments/assets/46cbb9f2-ffad-4f3c-8ce5-c788a5e39e1a)
![conclusion](https://github.com/user-attachments/assets/b3225ec9-331f-4537-81d4-22846baff071)


Model XGBoost menunjukkan performa terbaik dengan akurasi dan f1-score tertinggi dibandingkan model lain. Meskipun recall untuk kelas buruk masih bisa ditingkatkan, model ini sudah cukup baik untuk diterapkan dalam sistem penilaian risiko kredit oleh lembaga keuangan. Diperlukan iterasi dan tuning lebih lanjut untuk mencapai performa maksimal.

Model ini mampu menjawab problem statement dan mencapai goals utama proyek dengan baik.

