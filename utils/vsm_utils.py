import math


def compute_tf(term, document):
    """
    Hitung Term Frequency (TF) suatu term dalam dokumen.

    Parameters:
        term (str): Term yang dihitung frekuensinya.
        document (list): Daftar token dalam dokumen.

    Returns:
        float: Nilai TF untuk term.
    """
    term_count = document.count(term)
    total_terms = len(document)
    return term_count / total_terms if total_terms != 0 else 0


def compute_idf(term, documents):
    """
    Hitung Inverse Document Frequency (IDF) suatu term dalam kumpulan dokumen.

    Parameters:
        term (str): Term yang dihitung IDF-nya.
        documents (list of list): Daftar dokumen, masing-masing berupa daftar token.

    Returns:
        float: Nilai IDF untuk term.
    """
    doc_count = sum(1 for doc in documents if term in doc)
    total_docs = len(documents)
    return math.log10(total_docs / (1 + doc_count)) if doc_count != 0 else 0


def compute_tf_idf(documents):
    """
    Bangun matriks TF-IDF untuk kumpulan dokumen.

    Parameters:
        documents (list of list): Daftar dokumen, masing-masing berupa daftar token.

    Returns:
        tuple: Matriks TF-IDF (list of list) dan daftar unik term (list).
    """
    unique_terms = set(term for doc in documents for term in doc)
    tf_idf_matrix = []

    # Menghitung TF-IDF untuk setiap dokumen
    for doc in documents:
        tf_idf_vector = []
        for term in unique_terms:
            tf = compute_tf(term, doc)
            idf = compute_idf(term, documents)
            tf_idf_vector.append(tf * idf)
        tf_idf_matrix.append(tf_idf_vector)

    return tf_idf_matrix, list(unique_terms)


def cosine_similarity(vector1, vector2):
    """
    Hitung cosine similarity antara dua vektor.

    Parameters:
        vector1 (list): Vektor pertama.
        vector2 (list): Vektor kedua.

    Returns:
        float: Nilai cosine similarity antara dua vektor.
    """
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(b**2 for b in vector2))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
