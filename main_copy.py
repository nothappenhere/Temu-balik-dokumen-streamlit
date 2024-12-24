import streamlit as st
import os
import csv
import string
import re

from collections import defaultdict
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter


from PyPDF2 import PdfReader
from docx import Document

from vsm import compute_idf, compute_tf, compute_tf_idf, cosine_similarity

from document_reader import (
    load_dict,
    load_stopwords,
    read_txt,
    read_docx,
    read_pdf,
    read_file,
)

st.set_page_config(
    page_title="Information Retrieval (IR)",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "## *Get this on [Github](https://github.com/nothappenhere/)!*",
    },
)


# Fungsi untuk melakukan preprocessing (case folding, tokenizing, filtering)
def preprocess(text, stopwords):
    # Tokenisasi
    tokens = word_tokenize(text.lower())
    # print(tokens)

    # Menghapus tanda baca
    tokens = [word for word in tokens if word not in string.punctuation]

    # Filter stopwords
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stopwords
    ]

    return filtered_tokens


# Fungsi untuk stemming menggunakan Sastrawi
def stemming(tokens, dictionary, stemmer):
    """
    Melakukan stemming pada token dan mencocokkan dengan kamus kata dasar.

    Parameters:
        tokens (list): Daftar token hasil preprocessing.
        dictionary (set): Kamus kata dasar.
        stemmer (SastrawiStemmer): Objek stemmer dari Sastrawi.

    Returns:
        dict: Dictionary berisi token asli, kata dasar, dan jumlahnya.
    """
    # Tokenisasi dan normalisasi teks (menghapus karakter non-huruf)
    # text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # tokens = re.findall(r'\b\w+\b', text)

    word_pairs = Counter()

    # Proses stemming dan bandingkan dengan kamus kata dasar
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if stemmed_token in dictionary:
            word_pairs[(token, stemmed_token)] += 1
    return word_pairs


# Streamlit UI
st.title("Aplikasi Temu Balik Dokumen")
st.write(
    """
Aplikasi ini dirancang untuk membantu pengguna dalam melakukan proses pengolahan teks pada dokumen. 
Dengan aplikasi ini, pengguna dapat:
- Membaca dan memproses dokumen dalam format *.txt*, *.docx*, dan *.pdf*.
- Melakukan *preprocessing* teks , termasuk *case folding*, *tokenizing*, *filtering*, dan *stemming*.
- Menampilkan hasil *stemming* dan jumlah kata dasar yang ditemukan pada setiap dokumen.
- Menganalisis kemiripan antara *query* yang dimasukkan dengan dokumen yang diproses.
"""
)

st.sidebar.header("KONFIGURASI", divider="orange")

# Input direktori
directory = st.sidebar.text_input(
    "Masukkan path direktori:",
    help="Gunakan **./** atau **../** untuk relative path.",
)
dictionary_path = (
    st.sidebar.text_input(
        "Masukkan path kamus kata dasar:",
        help="Default: dictionary.txt",
    )
    or "./preprocessing/dictionary.txt"
)
stopwords_path = (
    st.sidebar.text_input("Masukkan path file stopwords:", help="Default: stopword.csv")
    or "./preprocessing/stopword.csv"
)

query = st.sidebar.text_input("Masukan query yang ingin dicari:")

st.divider()

if st.sidebar.button("Proses", type="primary"):
    if not os.path.exists(directory):
        st.error(f"Direktori **{directory}** tidak ditemukan!", icon="‼️")
    elif not os.path.exists(dictionary_path):
        st.error(f"File kamus **{dictionary_path}** tidak ditemukan!", icon="‼️")
    elif not os.path.exists(stopwords_path):
        st.error(f"File stopwords **{stopwords_path}** tidak ditemukan!", icon="‼️")
    elif query == "":
        st.error("Query tidak boleh kosong!", icon="‼️")
    else:
        # Load dictionary dan stopwords
        dictionary = load_dict(dictionary_path)
        stopwords = load_stopwords(stopwords_path)

        # Membuat stemmer menggunakan Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # List file di direktori
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        if not files:
            st.warning(f"Tidak ada file dalam direktori **{directory}**.", icon="⚠️")
        else:
            st.write(f"#### File ditemukan di direktori `{directory}`:")
            for file_path in files:
                st.write(f"- {os.path.basename(file_path)}")

            st.divider()

            # Proses setiap file
            results = defaultdict(dict)
            for file_path in files:
                try:
                    # Baca isi file
                    content = read_file(file_path)
                    st.write(f"#### Isi file `{os.path.basename(file_path)}`:")
                    st.text(content[:500])  # Menampilkan 500 karakter pertama

                    # Preprocessing
                    tokens = preprocess(content, stopwords)
                    # Stemming
                    stemming_result = stemming(tokens, dictionary, stemmer)

                    # Simpan hasil
                    results[file_path]["original"] = content
                    results[file_path]["processed"] = " ".join(tokens)
                    results[file_path]["stemming"] = [
                        f"{original} -> {stemmed}"
                        for original, stemmed in stemming_result
                    ]
                    results[file_path]["stemming_count"] = sum(stemming_result.values())

                    # Tampilkan hasil
                    st.write("##### Hasil Preprocessing:")
                    st.write(tokens)
                    st.write("##### Hasil Stemming:")
                    for original, stemmed in stemming_result:
                        st.write(f"{original} -> {stemmed}")
                    st.write(
                        f"Jumlah kata dasar: {results[file_path]['stemming_count']}"
                    )
                except Exception as e:
                    st.error(
                        f"Kesalahan memproses file `{os.path.basename(file_path)}`: {e}"
                    )
