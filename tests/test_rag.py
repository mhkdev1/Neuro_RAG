import fitz

from config import CHUNK_SIZE
from ingest import split_text, extract_chunks_from_pdf


def test_texto_vazio_nao_gera_chunks():
    """Um texto vazio não deve gerar trechos."""

    resultado = split_text("   \n\t  ")

    assert resultado == []


def test_texto_curto_permanece_inteiro():
    """Um texto menor que o limite deve permanecer em um chunk."""

    texto = "A EEG é usada para estudar a atividade cerebral."

    resultado = split_text(texto)

    assert resultado == [texto]


def test_texto_longo_e_dividido():
    """Um texto longo deve ser dividido em múltiplos chunks."""

    texto = "neurociencia " * 500

    resultado = split_text(texto)

    assert len(resultado) > 1
    assert all(len(chunk) <= CHUNK_SIZE for chunk in resultado)


def test_divisao_e_deterministica():
    """A mesma entrada deve gerar os mesmos chunks."""

    texto = "EEG e imagetica motora " * 100

    primeira_execucao = split_text(texto)
    segunda_execucao = split_text(texto)

    assert primeira_execucao == segunda_execucao


def test_extracao_de_pdf_preserva_a_pagina(tmp_path):
    """A extração deve preservar o texto e a página de origem."""

    caminho_pdf = tmp_path / "artigo_teste.pdf"

    # Cria um PDF temporário com texto selecionável.
    with fitz.open() as documento:
        pagina = documento.new_page()
        pagina.insert_text(
            (72, 72),
            "EEG e imagetica motora",
        )
        documento.save(caminho_pdf)

    registros = extract_chunks_from_pdf(caminho_pdf)

    assert len(registros) > 0
    assert any("EEG" in registro["text"] for registro in registros)
    assert all(
        registro["metadata"]["source"] == "artigo_teste.pdf"
        for registro in registros
    )
    assert all(
        registro["metadata"]["page"] == 1
        for registro in registros
    )