# VS Code

Folder ini berisi kode sumber aplikasi deteksi alfabet Bahasa Isyarat Indonesia (BISINDO) yang dikembangkan menggunakan Visual Studio Code.

## Isi Folder

- `bisindo_suara.py` — aplikasi Streamlit untuk deteksi alfabet BISINDO secara real-time.
- `requirements.txt` — daftar library Python yang diperlukan untuk menjalankan aplikasi.

## Fitur Aplikasi

Aplikasi memiliki beberapa fitur utama:

- Deteksi alfabet BISINDO secara real-time menggunakan YOLOv11.
- Menampilkan hasil deteksi dan confidence.
- Word Builder untuk merangkai hasil deteksi menjadi kata.
- Text-to-Speech (TTS) Bahasa Indonesia.
- Pengukuran FPS secara real-time.

## Cara Menjalankan

Install library yang diperlukan:

```bash
pip install -r requirements.txt
```
## Cara Menjalankan aplikasi Streamlit:
```bash
streamlit run bisindo_suara.py
```
