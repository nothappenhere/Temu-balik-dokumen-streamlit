import streamlit as st
from utils.text_utils import process_documents

st.set_page_config(
    page_title="Information Retrieval (IR)",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "## **Get this on [Github](https://github.com/nothappenhere/)!**",
    },
)


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

st.sidebar.header("Konfigurasi", divider="orange")

# Input direktori
directory = st.sidebar.text_input(
    "Masukkan path direktori:",
    "./documents",
    help="Gunakan **./** atau **../** untuk relative path.",
)
dictionary_path = (
    st.sidebar.text_input(
        "Masukkan path kamus kata dasar:",
        help="Default: *dictionary.txt*",
    )
    or "./helper/dictionary.txt"
)
stopwords_path = (
    st.sidebar.text_input(
        "Masukkan path file stopwords:", help="Default: *stopword.csv*"
    )
    or "./helper/stopword.csv"
)
query = st.sidebar.text_area("Masukan query yang ingin dicari:")

st.divider()

if st.sidebar.button("Proses", type="primary"):
    process_documents(directory, dictionary_path, stopwords_path, query)
