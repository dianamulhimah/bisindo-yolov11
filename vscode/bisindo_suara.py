"""
==============================================================
  APLIKASI DETEKSI ALFABET BISINDO
  Skripsi S1 — Universitas Trilogi FSTD 2026
  
  Judul  : Analisis Performa Model YOLOv11 untuk Deteksi
           Bahasa Isyarat Indonesia (BISINDO) Berbasis Real-Time
  Model  : YOLOv11s (best_YOLOv11_Small.pt)
  Stack  : Streamlit · OpenCV · gTTS · pygame · ultralytics

  Fitur:
  - Deteksi huruf BISINDO A–Z secara real-time via kamera
  - Word Builder otomatis (tahan isyarat 1.5 detik per huruf), disertai
    suara per huruf saat berhasil tercatat
  - Text-to-Speech Bahasa Indonesia (gTTS + pygame): suara per huruf saat
    dicatat, plus tombol "🔊 Ucapkan Kata" untuk membaca ulang SELURUH
    kata di Word Builder sebagai satu kata utuh (bukan dieja per huruf)
  - Dashboard statistik & riwayat sesi
  - Tab Panduan & Tentang Aplikasi

CARA MENJALANKAN:
    pip install -r requirements_suara.txt
    streamlit run bisindo_suara.py
==============================================================
"""

# ============================================================
# BAGIAN 1 — IMPORT LIBRARY
# ============================================================

import cv2
import os
import sys
import time
import json
import numpy as np
import threading
import tempfile
import datetime
from collections import deque

from gtts import gTTS
import pygame
from ultralytics import YOLO
import streamlit as st
import torch


# ============================================================
# BAGIAN 2 — KONFIGURASI GLOBAL
# ============================================================

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# Warna bounding box per huruf (BGR)
COLORS = [
    (255,80,80),   (80,255,80),   (80,80,255),   (255,255,80),
    (255,80,255),  (80,255,255),  (200,120,80),  (80,200,120),
    (120,80,200),  (200,200,80),  (200,80,200),  (80,200,200),
    (255,160,80),  (255,80,160),  (160,255,80),  (80,255,160),
    (160,80,255),  (80,160,255),  (255,160,160), (160,255,160),
    (160,160,255), (255,210,100), (210,255,100), (100,210,255),
    (210,100,255), (255,100,100)
]

MODEL_PATH    = "best_YOLOv11_Small.pt"
DEFAULT_CONF  = 0.90
DEFAULT_IOU   = 0.5
HOLD_DURATION = 1.5   # detik tahan isyarat sebelum dicatat
SPEAK_DELAY   = 1.0   # jeda minimum antar ucapan (detik)
MAX_RIWAYAT   = 100   # maksimum entri riwayat sesi


# ============================================================
# BAGIAN 3 — MODUL TEXT-TO-SPEECH (gTTS + pygame)
# ============================================================

class SuaraBISINDO:
    """
    Modul TTS untuk mengucapkan huruf dan kata BISINDO.
    Menggunakan gTTS (Bahasa Indonesia) + pygame untuk playback.
    Dijalankan pada daemon thread agar tidak memblokir UI.
    """

    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._ok = True
        except Exception:
            self._ok = False
        self.last_speak_time = 0
        self.is_speaking     = False
        self._lock           = threading.Lock()

    def _generate_and_play(self, teks: str):
        try:
            with self._lock:
                self.is_speaking = True
            tts = gTTS(text=teks, lang='id', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tmp = f.name
                tts.save(tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            os.unlink(tmp)
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            with self._lock:
                self.is_speaking = False

    def ucapkan(self, teks: str, force: bool = False):
        if not self._ok or not teks:
            return
        now = time.time()
        if not force and (now - self.last_speak_time) < SPEAK_DELAY:
            return
        if self.is_speaking and not force:
            return
        self.last_speak_time = now
        threading.Thread(
            target=self._generate_and_play,
            args=(teks,), daemon=True
        ).start()

    def ucapkan_huruf(self, h: str):
        self.ucapkan(h)

    def ucapkan_kata(self, k: str):
        self.ucapkan(k, force=True)

    def stop(self):
        if self._ok:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass


# ============================================================
# BAGIAN 4 — HELPER UI
# ============================================================

def _init_state():
    """Inisialisasi session state Streamlit."""
    defaults = {
        'kata'        : '',
        'riwayat'     : [],          # list of {kata, waktu, panjang}
        'total'       : 0,
        'running'     : False,
        'stats'       : {h: 0 for h in CLASS_NAMES},
        'suara_aktif' : True,
        'fps_log'     : deque(maxlen=60),
        'conf_log'    : deque(maxlen=60),
        'last_huruf'  : '—',
        'last_conf'   : 0.0,
        'sesi_mulai'  : None,
        'hold_pct'    : 0.0,
        'tab_aktif'   : 'Kamera',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _css():
    """Inject CSS kustom."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Header ── */
    .app-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d4aa 0%, #0099ff 60%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .app-sub {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
    }
    .app-badge {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .badge {
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.03em;
    }
    .badge-teal  { background:#0d3330; color:#00d4aa; border:1px solid #00d4aa44; }
    .badge-blue  { background:#0a1f3a; color:#60a5fa; border:1px solid #60a5fa44; }
    .badge-purple{ background:#1e1040; color:#a855f7; border:1px solid #a855f744; }
    .badge-gold  { background:#2a1f00; color:#fbbf24; border:1px solid #fbbf2444; }

    /* ── Box Huruf ── */
    .box-huruf {
        font-size: 5.5rem;
        font-weight: 700;
        text-align: center;
        color: #00d4aa;
        background: linear-gradient(135deg, #0a1a18 0%, #0d2420 100%);
        border: 2px solid #00d4aa55;
        border-radius: 20px;
        padding: 18px 10px;
        margin: 6px 0;
        letter-spacing: 0.05rem;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 30px #00d4aa15, inset 0 1px 0 #00d4aa22;
        transition: all .3s;
    }
    .box-huruf-aktif {
        font-size: 5.5rem;
        font-weight: 700;
        text-align: center;
        color: #00ffcc;
        background: linear-gradient(135deg, #0a2e28 0%, #0d3830 100%);
        border: 2px solid #00d4aa;
        border-radius: 20px;
        padding: 18px 10px;
        margin: 6px 0;
        letter-spacing: 0.05rem;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 0 40px #00d4aa30, inset 0 1px 0 #00d4aa44;
    }

    /* ── Box Kata ── */
    .box-kata {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        color: #fbbf24;
        background: linear-gradient(135deg, #1c1500 0%, #251c00 100%);
        border: 2px solid #fbbf2455;
        border-radius: 14px;
        padding: 12px 16px;
        margin: 6px 0;
        letter-spacing: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        min-height: 66px;
        box-shadow: 0 0 20px #fbbf2412;
    }

    /* ── Confidence Bar ── */
    .conf-wrap { margin: 6px 0 10px; }
    .conf-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 3px;
        display: flex;
        justify-content: space-between;
    }
    .conf-bg {
        background: #1f2937;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 8px;
        transition: width .3s;
    }

    /* ── Progress Hold ── */
    .hold-wrap { margin: 4px 0 12px; }
    .hold-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 3px;
        display: flex;
        justify-content: space-between;
    }
    .hold-bg {
        background: #1f2937;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    .hold-fill {
        height: 100%;
        border-radius: 8px;
        transition: width .2s;
    }

    /* ── Riwayat ── */
    .riw-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        background: #111827;
        border-radius: 10px;
        margin-bottom: 5px;
        border-left: 3px solid #00d4aa;
        font-family: 'JetBrains Mono', monospace;
    }
    .riw-kata { font-size: 1.1rem; color: #fbbf24; font-weight: 600; }
    .riw-meta { font-size: 0.7rem; color: #6b7280; }
                

    /* ── Stat Card ── */
    .stat-card {
        background: #111827;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #1f2937;
        text-align: center;
    }
    .stat-val {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-lbl { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; }

    /* ── Huruf chart bar ── */
    .bar-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 3px;
    }
    .bar-lbl { width: 20px; font-size: 0.75rem; color: #9ca3af;
               font-family: 'JetBrains Mono', monospace; text-align: right; }
    .bar-bg2  { flex:1; background:#1f2937; border-radius:4px; height:10px; overflow:hidden; }
    .bar-fill2 { height:100%; background:linear-gradient(90deg,#00d4aa,#0099ff);
                 border-radius:4px; transition:width .4s; }
    .bar-cnt  { width:32px; font-size:0.7rem; color:#6b7280;
                font-family:'JetBrains Mono',monospace; text-align:right; }

    /* ── Panduan Alfabet ── */
    .alfa-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    .alfa-card {
        background: #0d1117;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 14px 8px;
        text-align: center;
        transition: border-color .2s;
    }
    .alfa-card:hover { border-color: #00d4aa55; }
    .alfa-huruf {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
        font-family: 'JetBrains Mono', monospace;
    }
    .alfa-name { font-size: 0.65rem; color: #6b7280; margin-top: 2px; }

    /* ── Tombol Aksi ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all .2s !important;
    }

    /* ── Info box ── */
    .info-box {
        background: #0a1628;
        border-left: 3px solid #60a5fa;
        border-radius: 0 10px 10px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #93c5fd;
    }

    /* ── Status Dot ── */
    .status-dot-on  { color:#00d4aa; font-size:0.8rem; }
    .status-dot-off { color:#6b7280; font-size:0.8rem; }

    /* Sembunyikan footer Streamlit */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def _header():
    """Render header aplikasi."""
    st.markdown('<div class="app-title">🤟 BISINDO Detector</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Deteksi Alfabet Bahasa Isyarat Indonesia · '
        'YOLOv11 · Text-to-Speech · Real-Time</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="app-badge">
      <span class="badge badge-teal">YOLOv11</span>
      <span class="badge badge-blue">OpenCV</span>
      <span class="badge badge-purple">Streamlit</span>
      <span class="badge badge-gold">gTTS · pygame</span>
    </div>
    """, unsafe_allow_html=True)


def _sidebar():
    """Render sidebar pengaturan dan kontrol."""
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")

        model_p  = st.text_input("Path Model", "best_YOLOv11_Small.pt",
                                  help="Lokasi file best.pt")
        conf_val = st.slider("Confidence Threshold",
                              0.10, 1.0, DEFAULT_CONF, 0.90,
                              help="Nilai minimum kepercayaan deteksi")
        cam_idx  = st.selectbox("Pilih Kamera",
                                 options=[0, 1, 2],
                                 format_func=lambda x: f"Kamera {x}",
                                 help="Indeks perangkat kamera")

        st.divider()

        
        # Toggle suara
        st.markdown("Suara")
        suara_on = st.toggle("Aktifkan Suara",
                              value=st.session_state.suara_aktif,
                              help="TTS otomatis saat huruf dicatat")
        st.session_state.suara_aktif = suara_on
        if suara_on:
            st.markdown('<p class="status-dot-on">● Suara Aktif</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-dot-off">● Suara Mati</p>',
                        unsafe_allow_html=True)

        st.divider()

        # Kontrol kamera
        st.markdown("Kontrol")
        c1, c2 = st.columns(2)
        with c1:
            mulai = st.button("▶️ Mulai", use_container_width=True,
                               type="primary")
        with c2:
            stop  = st.button("⏹️ Stop",  use_container_width=True)

        if mulai:
            st.session_state.running   = True
            st.session_state.sesi_mulai = datetime.datetime.now()
        if stop:
            st.session_state.running   = False

        st.divider()

        # Word Builder
        st.markdown("Word Builder")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("⬅️ Hapus", use_container_width=True,
                          help="Hapus huruf terakhir"):
                st.session_state.kata = st.session_state.kata[:-1]
        with c4:
            if st.button("🗑️ Reset", use_container_width=True,
                          help="Hapus seluruh kata"):
                st.session_state.kata = ""

        if st.button("␣ Spasi", use_container_width=True,
                      help="Sisipkan spasi untuk memisahkan kata "
                           "(gabungkan beberapa kata jadi satu frasa)"):
            kata = st.session_state.kata
            # Hindari spasi di awal kalimat atau spasi ganda berturut-turut
            if kata and not kata.endswith(" "):
                st.session_state.kata += " "

        if st.button("✅ Simpan Kata", use_container_width=True,
                      help="Simpan kata ke riwayat"):
            if st.session_state.kata:
                st.session_state.riwayat.append({
                    'kata'   : st.session_state.kata,
                    'waktu'  : datetime.datetime.now().strftime("%H:%M:%S"),
                    'panjang': len(st.session_state.kata)
                })
                st.session_state.kata = ""
                st.toast("✅ Kata disimpan ke riwayat!", icon="✅")

        if st.button("🔊 Ucapkan Kata", use_container_width=True,
                      help="Ucapkan kata saat ini"):
            st.session_state['force_speak'] = True

        if st.button("🔄 Reset Sesi", use_container_width=True):
            st.session_state.kata    = ""
            st.session_state.riwayat = []
            st.session_state.total   = 0
            st.session_state.stats   = {h: 0 for h in CLASS_NAMES}
            st.session_state.fps_log = deque(maxlen=60)
            st.toast("🔄 Sesi direset!", icon="🔄")

        st.divider()
        st.markdown("""
        <div class="info-box">
        💡 <b>Tips:</b><br>
        · Tahan isyarat ±1,5 detik → huruf otomatis dicatat ke Word
          Builder dan diucapkan (suara per huruf)<br>
        · Klik <b>␣ Spasi</b> untuk memisahkan kata, lalu lanjutkan
          isyarat huruf untuk kata berikutnya (bisa jadi frasa/kalimat)<br>
        · Klik <b>🔊 Ucapkan Kata</b> untuk membaca seluruh isi Word
          Builder sekaligus (bukan dieja ulang per huruf)<br>
        · Pastikan pencahayaan cukup<br>
        · Posisikan tangan di tengah kamera
        </div>
        """, unsafe_allow_html=True)

    return conf_val, cam_idx, model_p


def _render_huruf(huruf: str, aktif: bool = False):
    cls = "box-huruf-aktif" if aktif else "box-huruf"
    return f'<div class="{cls}">{huruf if huruf else "—"}</div>'


def _render_kata(kata: str):
    return (
        f'<div class="box-kata">'
        f'{kata if kata else "· · ·"}'
        f'</div>'
    )


def _render_conf_bar(conf: float):
    pct = int(conf * 100)
    if conf >= 0.8:
        col = "#00d4aa"
    elif conf >= 0.5:
        col = "#fbbf24"
    else:
        col = "#f87171"
    return f"""
    <div class="conf-wrap">
      <div class="conf-label">
        <span>Confidence</span><span>{pct}%</span>
      </div>
      <div class="conf-bg">
        <div class="conf-fill" style="width:{pct}%;background:{col};"></div>
      </div>
    </div>
    """


def _render_hold_bar(pct: float):
    p = int(pct * 100)
    col = "#00d4aa" if pct < 1.0 else "#a855f7"
    return f"""
    <div class="hold-wrap">
      <div class="hold-label">
        <span>Tahan Isyarat</span><span>{p}%</span>
      </div>
      <div class="hold-bg">
        <div class="hold-fill" style="width:{p}%;background:{col};"></div>
      </div>
    </div>
    """


def _render_stat_cards():
    total   = st.session_state.total
    n_riw   = len(st.session_state.riwayat)
    n_kelas = sum(1 for v in st.session_state.stats.values() if v > 0)

    sesi = ""
    if st.session_state.sesi_mulai:
        delta = datetime.datetime.now() - st.session_state.sesi_mulai
        menit, detik = divmod(int(delta.total_seconds()), 60)
        sesi = f"{menit:02d}:{detik:02d}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-val" style="color:#00d4aa">{total}</div>
          <div class="stat-lbl">Total Deteksi</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-val" style="color:#fbbf24">{n_riw}</div>
          <div class="stat-lbl">Kata Tersimpan</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-val" style="color:#a855f7">{n_kelas}</div>
          <div class="stat-lbl">Huruf Terdeteksi</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-val" style="color:#60a5fa">{sesi or "—"}</div>
          <div class="stat-lbl">Durasi Sesi</div>
        </div>""", unsafe_allow_html=True)


def _render_bar_chart():
    """Render bar chart horizontal deteksi per huruf."""
    stats  = st.session_state.stats
    maks   = max(stats.values()) if any(stats.values()) else 1
    baris  = ""
    for h in CLASS_NAMES:
        v    = stats[h]
        pct  = int((v / maks) * 100) if maks > 0 else 0
        baris += f"""
        <div class="bar-row">
          <div class="bar-lbl">{h}</div>
          <div class="bar-bg2">
            <div class="bar-fill2" style="width:{pct}%"></div>
          </div>
          <div class="bar-cnt">{v}</div>
        </div>"""
    st.markdown(baris, unsafe_allow_html=True)


def _render_riwayat():
    """Render daftar riwayat kata tersimpan."""
    if not st.session_state.riwayat:
        st.caption("Belum ada kata yang disimpan.")
        return

    for item in reversed(st.session_state.riwayat[-MAX_RIWAYAT:]):
        kata   = item['kata']
        waktu  = item['waktu']
        panjang = item['panjang']
        st.markdown(f"""
        <div class="riw-item">
          <span class="riw-kata">{kata}</span>
          <span class="riw-meta">{panjang} huruf · {waktu}</span>
        </div>""", unsafe_allow_html=True)


def _tab_panduan():
    """Konten tab Panduan Penggunaan."""
    st.markdown("### 📖 Panduan Penggunaan")
    st.markdown("""
    <div class="info-box">
    Aplikasi ini mendeteksi <b>26 huruf alfabet BISINDO (A–Z)</b> secara
    <i>real-time</i> menggunakan model YOLOv11n.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Langkah Penggunaan")
        st.markdown("""
        1. Pastikan **best_YOLOv11_Nano.pt** ada di folder yang sama
        2. Klik **▶️ Mulai** di sidebar
        3. Arahkan tangan ke kamera
        4. **Tahan isyarat ±1,5 detik** → huruf otomatis dicatat ke
           *Word Builder* dan diucapkan (suara per huruf)
        5. Ulangi untuk huruf berikutnya hingga satu kata terbentuk
        6. Klik **␣ Spasi** untuk mengakhiri kata dan mulai kata baru
           (bisa dirangkai jadi frasa/kalimat)
        7. Klik **🔊 Ucapkan Kata** → seluruh isi *Word Builder* dibaca
           ulang, termasuk jeda antar kata (bukan dieja per huruf)
        8. Klik **✅ Simpan Kata** untuk menyimpan ke riwayat
        """)

    with col_b:
        st.markdown("#### ⚙️ Pengaturan")
        st.markdown("""
        - **Confidence Threshold** — Semakin tinggi nilai ini, semakin
          ketat syarat deteksi.
        - **Pilih Kamera** — Pilih indeks kamera (0 = kamera utama)
        - **Aktifkan Suara** — Toggle untuk mengaktifkan/mematikan suara
          per huruf (saat tercatat) maupun suara dari tombol
          **🔊 Ucapkan Kata**
        - **␣ Spasi** — Menyisipkan spasi di *Word Builder* untuk
          memisahkan satu kata dari kata berikutnya
        - **Hold Duration** — Waktu tahan isyarat per huruf: **1,5 detik**
        """)

    st.divider()
    st.markdown("26 Huruf Alfabet BISINDO yang Dapat Dideteksi")

    nama_huruf = {
        'A':'Alpha','B':'Bravo','C':'Charlie','D':'Delta','E':'Echo',
        'F':'Foxtrot','G':'Golf','H':'Hotel','I':'India','J':'Juliet',
        'K':'Kilo','L':'Lima','M':'Mike','N':'November','O':'Oscar',
        'P':'Papa','Q':'Quebec','R':'Romeo','S':'Sierra','T':'Tango',
        'U':'Uniform','V':'Victor','W':'Whiskey','X':'X-ray','Y':'Yankee',
        'Z':'Zulu'
    }
    kartu = "".join([
        f'<div class="alfa-card">'
        f'<div class="alfa-huruf">{h}</div>'
        f'<div class="alfa-name">{nama_huruf[h]}</div>'
        f'</div>'
        for h in CLASS_NAMES
    ])
    st.markdown(f'<div class="alfa-grid">{kartu}</div>',
                unsafe_allow_html=True)


def _tab_tentang():
    """Konten tab Tentang Aplikasi."""
    st.markdown("### ℹ️ Tentang Aplikasi")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        Aplikasi ini merupakan implementasi dari skripsi S1 berjudul:

        > **"Analisis Performa Model YOLOv11 untuk Deteksi Bahasa Isyarat
        > Indonesia (BISINDO) Berbasis Real-Time"**

        **Program Studi** : Sistem Informasi  
        **Universitas**   : Universitas Trilogi  
        **Fakultas**      : Sains, Teknik, dan Desain (FSTD)  
        **Tahun**         : 2026  
        """)

        st.divider()
        st.markdown("#### 🔧 Spesifikasi Teknis")
        data_teknis = {
            "Model"              : "YOLOv11s (yolo11s.pt)",
            "Dataset"            : "Dataset citra bahasa isyarat alfabet BISINDO sebanyak 26 kelas",
            "Kelas"              : "26 huruf A–Z",
            "Training Epochs"    : "50",
            "Optimizer"          : "AdamW (lr=0.0003)",
            "Image Size"         : "640×640 px",
            "Hold Duration"      : "1,5 detik (per huruf)",
            "TTS Engine"         : "gTTS · Bahasa Indonesia (suara per huruf saat tercatat + tombol Ucapkan Kata untuk kata utuh)",
        }
        for k, v in data_teknis.items():
            st.markdown(f"- **{k}** : {v}")

    with col2:
        st.markdown("#### 📦 Library")
        libs = [
            ("ultralytics", "8.3.40", "#00d4aa"),
            ("opencv-python", "≥4.8.0", "#60a5fa"),
            ("streamlit", "≥1.28.0", "#a855f7"),
            ("gtts", "≥2.4.0", "#fbbf24"),
            ("pygame", "≥2.5.0", "#f87171"),
            ("numpy", "≥1.24.0", "#34d399"),
        ]
        for name, ver, col in libs:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid #1f2937;">'
                f'<span style="color:{col};font-family:monospace;">{name}</span>'
                f'<span style="color:#6b7280;font-size:0.8rem;">{ver}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📐 Metrik Evaluasi")
        st.markdown("""
        - mAP@50-95
        - mAP@50
        - Precision
        - Recall
        - F1-Score
        - FPS (real-time)
        """)


# ============================================================
# BAGIAN 5 — FUNGSI UTAMA STREAMLIT
# ============================================================

def run_streamlit():
    """Fungsi utama antarmuka web Streamlit."""

    st.set_page_config(
        page_title="BISINDO Detector · YOLOv11",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    _css()
    _init_state()
    _header()

    # Sidebar → dapatkan pengaturan
    conf_val, cam_idx, model_p = _sidebar()

    # ── Tab Navigasi ──
    tab_kamera, tab_panduan, tab_tentang = \
        st.tabs(["📹 Kamera", "📖 Panduan", "ℹ️ Tentang"])

    # ─────────────────────────────
    # TAB 1 — KAMERA
    # ─────────────────────────────
    with tab_kamera:
        _render_stat_cards()
        st.markdown("<br>", unsafe_allow_html=True)

        col_cam, col_info = st.columns([3, 2])

        with col_cam:
            st.markdown("#### 📹 Live Kamera")
            cam_ph = st.empty()
            if not st.session_state.running:
                cam_ph.info(
                    "Klik **▶️ Mulai** di sidebar untuk mengaktifkan kamera. "
                    "Pastikan **best_YOLOv11_Small.pt** tersedia."
                )

        with col_info:
            st.markdown("#### 🔤 Deteksi Huruf")
            huruf_ph = st.empty()
            huruf_ph.markdown(
                _render_huruf("—", aktif=False),
                unsafe_allow_html=True
            )

            conf_ph = st.empty()
            conf_ph.markdown(
                _render_conf_bar(0.0),
                unsafe_allow_html=True
            )

            hold_ph = st.empty()
            hold_ph.markdown(
                _render_hold_bar(0.0),
                unsafe_allow_html=True
            )

            st.markdown("#### 💬 Word Builder")
            kata_ph = st.empty()
            kata_ph.markdown(
                _render_kata(st.session_state.kata),
                unsafe_allow_html=True
            )

            st.markdown("#### 📚 Riwayat Cepat")
            riw_ph = st.empty()
            if st.session_state.riwayat:
                terakhir = [r['kata'] for r in st.session_state.riwayat[-5:]]
                riw_ph.markdown(
                    "  ·  ".join(f"**{k}**" for k in reversed(terakhir))
                )
            else:
                riw_ph.caption("Belum ada kata tersimpan.")

        # ── Loop Kamera ──
        if st.session_state.running:
            if not os.path.exists(model_p):
                st.error(
                    f"❌ Model tidak ditemukan: **{model_p}**. "
                    f"Pastikan best_YOLOv11_Small.pt ada di folder yang sama!"
                )
                st.stop()

            model  = YOLO(model_p)
            suara  = SuaraBISINDO()
            cap    = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            last_letter = ""
            last_time   = time.time()
            prev_time   = time.time()
            fps_buf     = deque(maxlen=30)

            if not cap.isOpened():
                st.error(
                    f"❌ Kamera {cam_idx} tidak dapat dibuka. "
                    f"Coba kamera lain di sidebar."
                )
                st.stop()

            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Gagal membaca frame kamera.")
                        break

                    frame = cv2.flip(frame, 1)

                    # Inferensi YOLOv11
                    results    = model(frame, conf=conf_val, iou=DEFAULT_IOU, verbose=False)
                    best_huruf = ""
                    best_conf  = 0.0

                    for result in results:
                        for box in result.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            cs           = float(box.conf[0])
                            cid          = int(box.cls[0])
                            h_det        = CLASS_NAMES[cid]
                            warna        = COLORS[cid]

                            cv2.rectangle(frame, (x1,y1), (x2,y2), warna, 3)
                            label       = f"{h_det}  {cs:.0%}"
                            (lw,lh), _  = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2
                            )
                            cv2.rectangle(frame,
                                          (x1, y1-lh-12),
                                          (x1+lw+10, y1),
                                          warna, -1)
                            cv2.putText(frame, label, (x1+5, y1-5),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.9, (255,255,255), 2)
                            st.session_state.stats[h_det] += 1
                            st.session_state.total        += 1

                            if cs > best_conf:
                                best_conf  = cs
                                best_huruf = h_det

                    # FPS
                    now     = time.time()
                    fps     = 1.0 / max(now - prev_time, 1e-8)
                    fps_buf.append(fps)
                    avg_fps = float(np.mean(fps_buf))
                    prev_time = now
                    st.session_state.fps_log.append(avg_fps)

                    # Overlay HUD pada frame
                    elapsed = 0.0
                    h_fr, w_fr = frame.shape[:2]

                    # Panel atas
                    ov = frame.copy()
                    cv2.rectangle(ov, (0,0), (w_fr,85), (10,10,10), -1)
                    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

                    cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                                (12,32), cv2.FONT_HERSHEY_DUPLEX,
                                0.85, (0,230,100), 2)

                    # Hold duration tracking
                    if best_huruf:
                        if best_huruf == last_letter:
                            elapsed = now - last_time
                            prog    = min(elapsed / HOLD_DURATION, 1.0)

                            # Progress bar pada frame
                            bx,by,bw2,bh2 = 10, h_fr-22, 300, 14
                            cv2.rectangle(frame,
                                          (bx,by), (bx+bw2,by+bh2),
                                          (50,50,50), -1)
                            cv2.rectangle(frame,
                                          (bx,by),
                                          (bx+int(bw2*prog), by+bh2),
                                          (0,230,120), -1)
                            cv2.putText(frame, f"Tahan {prog:.0%}",
                                        (bx+bw2+10, by+bh2-2),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0,230,120), 1)

                            # Update hold bar UI
                            hold_ph.markdown(
                                _render_hold_bar(prog),
                                unsafe_allow_html=True
                            )
                            st.session_state.hold_pct = prog

                            # Catat otomatis ke Word Builder + ucapkan
                            # huruf yang baru saja tercatat (suara per
                            # huruf). Tombol "🔊 Ucapkan Kata" terpisah
                            # tetap membaca SELURUH isi Word Builder
                            # sebagai satu kata utuh, bukan dieja ulang.
                            if elapsed >= HOLD_DURATION:
                                kata = st.session_state.kata
                                if not kata or kata[-1] != best_huruf:
                                    st.session_state.kata += best_huruf
                                    kata_ph.markdown(
                                        _render_kata(st.session_state.kata),
                                        unsafe_allow_html=True
                                    )
                                    if st.session_state.suara_aktif:
                                        suara.ucapkan_huruf(best_huruf)
                                last_time = now
                        else:
                            last_letter = best_huruf
                            last_time   = now
                            hold_ph.markdown(
                                _render_hold_bar(0.0),
                                unsafe_allow_html=True
                            )
                    else:
                        hold_ph.markdown(
                            _render_hold_bar(0.0),
                            unsafe_allow_html=True
                        )

                    # Update UI huruf & confidence
                    huruf_ph.markdown(
                        _render_huruf(best_huruf, aktif=bool(best_huruf)),
                        unsafe_allow_html=True
                    )
                    conf_ph.markdown(
                        _render_conf_bar(best_conf),
                        unsafe_allow_html=True
                    )
                    st.session_state.last_huruf = best_huruf or "—"
                    st.session_state.last_conf  = best_conf

                    # Tombol "🔊 Ucapkan Kata": baca SELURUH isi Word
                    # Builder sebagai satu kata utuh (bukan per huruf).
                    # Menghormati toggle "Aktifkan Suara" di sidebar.
                    if st.session_state.get('force_speak'):
                        if st.session_state.kata and st.session_state.suara_aktif:
                            suara.ucapkan_kata(st.session_state.kata)
                        st.session_state.force_speak = False

                    # Tampilkan frame RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_ph.image(frame_rgb, channels="RGB",
                                 use_container_width=True)

                    # Riwayat cepat
                    if st.session_state.riwayat:
                        terakhir = [r['kata'] for r in
                                    st.session_state.riwayat[-5:]]
                        riw_ph.markdown(
                            "  ·  ".join(f"**{k}**"
                                         for k in reversed(terakhir))
                        )

            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                cap.release()
                suara.stop()

    # ─────────────────────────────
    # TAB 6 — PANDUAN
    # ─────────────────────────────
    with tab_panduan:
        _tab_panduan()

    # ─────────────────────────────
    # TAB  — TENTANG
    # ─────────────────────────────
    with tab_tentang:
        _tab_tentang()


# ============================================================
# BAGIAN 8 — ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_streamlit()