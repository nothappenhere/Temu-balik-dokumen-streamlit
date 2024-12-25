import os
import string
import streamlit as st
from collections import Counter
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from utils.vsm_utils import compute_tf, compute_idf, compute_tf_idf, cosine_similarity
from utils.document_reader_utils import (
    load_dict,
    load_stopwords,
    read_file,
)


def preprocess(text, stopwords):
    """
    Melakukan preprocessing teks, termasuk case folding, tokenizing, dan filtering.

    Args:
        text (str): Teks yang akan diproses.
        stopwords (set): Daftar stopwords untuk filtering.

    Returns:
        list: Token hasil preprocessing.
    """
    # Tokenisasi
    tokens = word_tokenize(text.lower())

    # Menghapus tanda baca
    tokens = [word for word in tokens if word not in string.punctuation]

    # Filter stopwords
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stopwords
    ]

    return filtered_tokens


def stemming(tokens, dictionary, stemmer):
    """
    Melakukan stemming pada token dan mencocokkan dengan kamus kata dasar.

    Parameters:
        tokens (list): Daftar token hasil preprocessing.
        dictionary (set): Kamus kata dasar.
        stemmer (SastrawiStemmer): Objek stemmer dari Sastrawi.

    Returns:
        dict: Dictionary berisi pasangan token asli, kata dasar, dan jumlahnya.
    """
    word_pairs = Counter()

    # Proses stemming dan bandingkan dengan kamus kata dasar
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if stemmed_token in dictionary:
            word_pairs[(token, stemmed_token)] += 1
    return word_pairs


def highlight_query(text, query_tokens, stemmer):
    """
    Menyoroti kata-kata dalam teks yang cocok dengan query tokens.

    Parameters:
        text (str): Teks dokumen asli.
        query_tokens (list): Daftar token hasil preprocessing query.
        stemmer (SastrawiStemmer): Objek stemmer dari Sastrawi.

    Returns:
        str: Teks dengan kata-kata yang cocok disorot menggunakan tag <mark>.
    """
    words = word_tokenize(text)
    highlighted_text = []

    for word in words:
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word in query_tokens:
            # Sorot kata yang cocok
            highlighted_text.append(f"<mark>{word}</mark>")
        else:
            highlighted_text.append(word)

    return " ".join(highlighted_text)


def validate_inputs(directory, dictionary_path, stopwords_path, query):
    """
    Memvalidasi input yang diberikan oleh pengguna, termasuk memeriksa keberadaan direktori, file kamus,
    file stopwords, dan memastikan bahwa query tidak kosong.

    Args:
        directory (str): Path ke direktori yang berisi dokumen.
        dictionary_path (str): Path ke file kamus.
        stopwords_path (str): Path ke file stopwords.
        query (str): Query yang dimasukkan oleh pengguna.

    Returns:
        bool: True jika semua input valid, False jika ada yang tidak valid.
    """
    if not os.path.exists(directory):
        st.error(f"Direktori **{directory}** tidak ditemukan!", icon="‼️")
        return False
    elif not os.path.exists(dictionary_path):
        st.error(f"File kamus **{dictionary_path}** tidak ditemukan!", icon="‼️")
        return False
    elif not os.path.exists(stopwords_path):
        st.error(f"File stopwords **{stopwords_path}** tidak ditemukan!", icon="‼️")
        return False
    elif query == "":
        st.error("Query tidak boleh kosong!", icon="‼️")
        return False
    return True


def load_resources(dictionary_path, stopwords_path):
    """
    Memuat kamus dan stopwords dari file yang diberikan.

    Args:
        dictionary_path (str): Path ke file kamus.
        stopwords_path (str): Path ke file stopwords.

    Returns:
        tuple: Sebuah tuple yang berisi kamus dan stopwords.
    """
    dictionary = load_dict(dictionary_path)
    stopwords = load_stopwords(stopwords_path)
    return dictionary, stopwords


def process_document(file_path, stopwords, dictionary, stemmer):
    """
    Memproses sebuah dokumen, mulai dari membaca konten file, melakukan preprocessing,
    dan mengembalikan token dan hasil stemming yang telah difilter berdasarkan kamus.

    Args:
        file_path (str): Path ke file dokumen yang akan diproses.
        stopwords (list): Daftar kata-kata stopwords.
        dictionary (list): Daftar kata-kata valid dalam kamus.
        stemmer (object): Objek stemmer untuk melakukan stemming pada kata.

    Returns:
        dict: Sebuah dictionary yang berisi konten asli, token, stemmed tokens, dan jumlah kata dasar.
    """
    try:
        content = read_file(file_path)
        tokens = preprocess(content, stopwords)
        stemmed_tokens = [
            stemmer.stem(token) for token in tokens if token in dictionary
        ]
        return {
            "original": content,
            "tokens": tokens,
            "stemmed": stemmed_tokens,
            "stemming_count": len(stemmed_tokens),
        }
    except Exception as e:
        st.error(f"Kesalahan memproses file `{os.path.basename(file_path)}`: {e}")
        return None


def process_query(query, stopwords, dictionary, stemmer):
    """
    Memproses query pengguna dengan cara yang sama seperti dokumen, termasuk preprocessing
    dan stemming terhadap token-token dalam query.

    Args:
        query (str): Query yang dimasukkan oleh pengguna.
        stopwords (list): Daftar kata-kata stopwords.
        dictionary (list): Daftar kata-kata valid dalam kamus.
        stemmer (object): Objek stemmer untuk melakukan stemming.

    Returns:
        list: Daftar kata dasar (stemmed tokens) dari query pengguna.
    """
    query_tokens = preprocess(query, stopwords)
    stemmed_query = [
        stemmer.stem(token) for token in query_tokens if token in dictionary
    ]
    return stemmed_query


def compute_similarity(documents, query, unique_terms):
    """
    Menghitung kemiripan antara dokumen-dokumen dan query menggunakan metode TF-IDF dan cosine similarity.

    Args:
        documents (list): Daftar dokumen yang sudah diproses.
        query (list): Query yang sudah diproses.
        unique_terms (list): Daftar istilah unik yang ada pada dokumen.

    Returns:
        list: Daftar nilai kemiripan antara setiap dokumen dengan query.
    """
    tf_idf_matrix, _ = compute_tf_idf(documents)
    query_tf_idf = [
        compute_tf(term, query) * compute_idf(term, documents) for term in unique_terms
    ]
    similarities = [
        cosine_similarity(doc_vector, query_tf_idf) for doc_vector in tf_idf_matrix
    ]
    return similarities


def display_results(directory, sorted_results, file_contents, stemmed_query, stemmer):
    """
    Menampilkan hasil kemiripan antara dokumen dan query beserta informasi terkait
    dokumen-dokumen yang relevan, termasuk konten yang disorot dan hasil preprocessing.

    Args:
        sorted_results (list): Daftar hasil yang sudah diurutkan berdasarkan kemiripan.
        file_contents (dict): Dictionary yang berisi konten dan hasil dari setiap dokumen.
        stemmed_query (list): Query yang sudah diproses dan di-stem.
        stemmer (object): Objek stemmer untuk menampilkan hasil stemming.
    """
    with st.container(height=393, border=True):
        cols_mr = st.columns([10.9, 0.2, 10.9])
        with cols_mr[0].container(height=350, border=False):
            st.write(f"#### File ditemukan dalam direktori `{directory}`:")
            for file_path, _ in sorted_results:
                st.write(f"- {os.path.basename(file_path)}")
        with cols_mr[1]:
            st.html(
                """
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 350px;
                        margin: auto;
                    }
                </style>
            """
            )
        with cols_mr[2].container(height=350, border=False):
            st.write("#### Hasil Similiarity:")
            for idx, (file_path, similarity) in enumerate(sorted_results, start=1):
                st.write(
                    f"{idx}. **{os.path.basename(file_path)}** - *Similarity*: {similarity:.4f}"
                )

    st.divider()

    for idx, (file_path, similarity) in enumerate(sorted_results, start=1):
        st.write(
            f"#### {idx}. **{os.path.basename(file_path)}** - *Similarity*: {similarity:.4f}"
        )
        if similarity > 0:
            st.write(f"###### Isi file `{os.path.basename(file_path)}`:")
            highlighted_content = highlight_query(
                file_contents[file_path]["original"], stemmed_query, stemmer
            )
            if len(highlighted_content) > 500:
                st.markdown(highlighted_content[:1000], unsafe_allow_html=True)
            else:
                st.markdown(highlighted_content, unsafe_allow_html=True)

            st.write("##### Hasil Preprocessing:")
            st.write(file_contents[file_path]["tokens"])
            st.write("##### Hasil Stemming:")
            for token in file_contents[file_path]["stemmed"]:
                st.write(f"- {token}")
            st.write(
                f"##### Jumlah kata dasar: {file_contents[file_path]['stemming_count']}"
            )
            st.divider()


def process_documents(directory, dictionary_path, stopwords_path, query):
    """
    Fungsi utama untuk memproses dokumen dalam sebuah direktori, melakukan preprocessing dan
    perhitungan kemiripan antara dokumen dengan query yang diberikan pengguna.

    Fungsi ini menggabungkan beberapa proses, termasuk validasi input, pemuatan sumber daya,
    pemrosesan dokumen dan query, perhitungan TF-IDF, dan perhitungan kemiripan menggunakan
    cosine similarity.

    Args:
        directory (str): Direktori yang berisi file-file yang akan diproses.
        dictionary_path (str): Path ke file kamus yang digunakan untuk validasi token.
        stopwords_path (str): Path ke file stopwords yang digunakan untuk menghapus kata-kata tidak penting.
        query (str): Query dari pengguna untuk dihitung kemiripannya dengan dokumen.

    Returns:
        None: Fungsi ini tidak mengembalikan nilai, melainkan menampilkan hasil ke dalam antarmuka pengguna (UI).
    """
    if not validate_inputs(directory, dictionary_path, stopwords_path, query):
        return

    dictionary, stopwords = load_resources(dictionary_path, stopwords_path)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]

    if not files:
        st.warning(f"Tidak ada file dalam direktori **{directory}**.", icon="⚠️")
        return

    documents = []
    file_contents = {}
    for file_path in files:
        result = process_document(file_path, stopwords, dictionary, stemmer)
        if result:
            documents.append(result["stemmed"])
            file_contents[file_path] = result

    stemmed_query = process_query(query, stopwords, dictionary, stemmer)

    tf_idf_matrix, unique_terms = compute_tf_idf(documents)
    similarities = compute_similarity(documents, stemmed_query, unique_terms)

    sorted_results = sorted(zip(files, similarities), key=lambda x: x[1], reverse=True)

    display_results(directory, sorted_results, file_contents, stemmed_query, stemmer)
