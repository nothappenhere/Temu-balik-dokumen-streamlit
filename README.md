# **Sistem Temu Balik Dokumen**

Aplikasi **Sistem Temu Balik Dokumen** adalah sebuah aplikasi berbasis web yang dirancang untuk membantu pengguna mencari dan menemukan dokumen relevan berdasarkan query yang diberikan. Sistem ini menggunakan metode **TF-IDF** dan **Cosine Similarity** untuk menghitung tingkat kemiripan antara dokumen-dokumen yang ada dengan query. 

Aplikasi ini mendukung berbagai format file seperti PDF dan DOCX serta mampu melakukan preprocessing teks yang meliputi tokenisasi, penghapusan stopwords, dan stemming menggunakan **[PySastrawi](https://github.com/har07/PySastrawi)**. 

## Fitur Utama
- Mendukung format file PDF dan DOCX.
- Preprocessing teks (tokenisasi, penghapusan stopwords, stemming).
- Perhitungan kemiripan dokumen menggunakan TF-IDF dan Cosine Similarity.
- Tampilan hasil yang interaktif dengan **Streamlit**, termasuk:
  - Highlight query dalam dokumen.
  - Daftar dokumen yang relevan berdasarkan tingkat kemiripan.
- Desain modular untuk memudahkan pengembangan lebih lanjut.

## Cara Install
1. Pastikan Python 3.8 atau lebih baru sudah terinstall di komputer Anda.
2. Clone atau unduh repository ini ke komputer Anda.
3. Buka terminal atau command prompt, lalu pindah ke direktori proyek.
4. Install semua library yang dibutuhkan dengan menjalankan perintah berikut:
```bash
  pip install -r requirements.txt
```

## Penggunanaan
1. Jalankan aplikasi dengan perintah berikut di terminal:
```bash
  streamlit run main.py
```
2. Buka browser Anda dan akses aplikasi di http://localhost:8000.
3. Isi form yang disediakan dengan:
    - Direktori tempat dokumen-dokumen berada.
    - Path ke file kamus (*dictionary*).
    - Path ke file stopwords.
    - Query yang ingin dicari.
4. Klik tombol "Proses" untuk memulai proses. Hasil pencarian akan ditampilkan berupa daftar dokumen yang relevan beserta tingkat kemiripannya.

## Lisensi
Proyek ini menggunakan lisensi MIT License. Anda bebas untuk menggunakan, memodifikasi, dan mendistribusikan ulang proyek ini sesuai dengan ketentuan lisensi.