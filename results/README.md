# Hasil Evaluasi

Folder ini berisi hasil pengujian dan evaluasi model YOLOv11 untuk deteksi alfabet Bahasa Isyarat Indonesia (BISINDO).

## Model yang Dievaluasi

Penelitian membandingkan dua varian model:

- YOLOv11 Nano
- YOLOv11 Small

## Metrik Evaluasi

Evaluasi model dilakukan menggunakan beberapa metrik, yaitu:

- Precision
- Recall
- F1-Score
- mAP@50
- mAP@50-95
- FPS
- Waktu inferensi

## Hasil Evaluasi

Hasil evaluasi mencakup pengujian pada:

- Validation Set
- Test Set
- Pengujian real-time menggunakan aplikasi Streamlit

## File dalam Folder

| File | Keterangan |
|---|---|
| `metrics.xlsx` | Ringkasan metrik evaluasi model |
| `precision_recall_f1.xlsx` | Perhitungan Precision, Recall, dan F1-Score |
| `fps_results.xlsx` | Hasil pengujian FPS dan waktu inferensi |
| `confusion_matrix_nano.png` | Confusion matrix YOLOv11 Nano |
| `confusion_matrix_small.png` | Confusion matrix YOLOv11 Small |
| `results_nano.png` | Grafik hasil pelatihan YOLOv11 Nano |
| `results_small.png` | Grafik hasil pelatihan YOLOv11 Small |

## Catatan

File pada folder ini merupakan hasil pengolahan dan evaluasi penelitian skripsi. Dataset dan bobot model (`.pt`) tidak disertakan dalam repository karena ukuran file yang besar.
