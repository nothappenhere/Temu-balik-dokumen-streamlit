import csv
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader


def load_dict(file_name):
    """
    Membaca file kamus dan mengembalikan set kata dasar.

    Parameters:
        file_name (str): Path ke file kamus kata dasar.

    Returns:
        set: Kumpulan kata dasar dari file kamus.
    """
    dictionary = set()
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            for row in file:
                word = row.strip()
                if word:
                    dictionary.add(word)
        if not dictionary:
            st.error(ValueError("File kamus tidak boleh kosong!"), icon="‼️")
    except FileNotFoundError:
        st.warning(f"File *{file_name}* tidak ditemukan.", icon="⚠️")
    except Exception as e:
        st.info(f"Kesalahan saat membaca file kamus: {e}", icon="ℹ️")
    return dictionary


def load_stopwords(file_name):
    """
    Membaca file stopwords dari CSV dan mengembalikan set stopwords.

    Parameters:
        file_name (str): Path ke file CSV stopwords.

    Returns:
        set: Kumpulan stopwords.
    """
    stopwords = set()
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row:  # Abaikan baris kosong
                    stopwords.add(row[0].strip())
        if not stopwords:
            st.error(ValueError("File stopwords tidak boleh kosong!"), icon="‼️")
    except FileNotFoundError:
        st.warning(f"File *{file_name}* tidak ditemukan.", icon="⚠️")
    except Exception as e:
        st.info(f"Kesalahan saat membaca file stopwords: {e}", icon="ℹ️")
    return stopwords


def read_txt(file_name):
    """
    Membaca teks dari file txt dan mengembalikan string.

    Parameters:
        file_name (str): Path ke file txt.

    Returns:
        str: Konten teks dari file.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            text = file.read()
        if not text:
            st.error(ValueError("File txt tidak boleh kosong!"), icon="‼️")
        return text
    except FileNotFoundError:
        st.warning(f"File *{file_name}* tidak ditemukan.", icon="⚠️")
    except Exception as e:
        st.info(f"Kesalahan saat membaca file txt: {e}", icon="ℹ️")
        return ""


def read_docx(file_name):
    """
    Membaca teks dari file docx dan mengembalikan string.

    Parameters:
        file_name (str): Path ke file docx.

    Returns:
        str: Konten teks dari file.
    """
    try:
        docx = Document(file_name)
        full_text = []
        for paragraph in docx.paragraphs:
            if paragraph.text.strip():  # Abaikan paragraf kosong
                full_text.append(paragraph.text)
        text = "\n".join(full_text)
        if not text:
            st.error(ValueError("File docx tidak boleh kosong!"), icon="‼️")
        return text
    except FileNotFoundError:
        st.warning(f"File *{file_name}* tidak ditemukan.", icon="⚠️")
    except Exception as e:
        st.info(f"Kesalahan saat membaca file docx: {e}", icon="ℹ️")
        return ""


def read_pdf(file_name):
    """
    Membaca teks dari file PDF dan mengembalikannya sebagai string.

    Parameters:
        file_name (str): Path ke file PDF.

    Returns:
        str: Konten teks dari file PDF.
    """
    try:
        with open(file_name, "rb") as file:
            pdf = PdfReader(file)
            if not pdf.pages:
                st.error(ValueError("PDF tidak memiliki halaman!"), icon="‼️")
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Abaikan halaman kosong
                    text += page_text
            if not text:
                st.error(ValueError("File PDF tidak boleh kosong!"), icon="‼️")
            return text
    except FileNotFoundError:
        st.warning(f"File *{file_name}* tidak ditemukan.", icon="⚠️")
    except Exception as e:
        st.info(f"Kesalahan saat membaca file PDF: {e}", icon="ℹ️")
        return ""


def read_file(file_name):
    """
    Membaca file berdasarkan ekstensinya (.txt, .docx, .pdf).

    Parameters:
        file_name (str): Path ke file input.

    Returns:
        str: Konten teks dari file.
    """
    if file_name.endswith(".txt"):
        return read_txt(file_name)
    elif file_name.endswith(".docx"):
        return read_docx(file_name)
    elif file_name.endswith(".pdf"):
        return read_pdf(file_name)
    else:
        st.error(
            ValueError(
                "Format file tidak didukung, hanya file dengan ekstensi **.txt**, **.docx**, atau **.pdf** yang diperbolehkan."
            ),
            icon="‼️",
        )
