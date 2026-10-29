import streamlit as st

from config import DOCUMENTS_DIR, CHROMA_DIR
from rag import answer_question


st.set_page_config(
    page_title="NeuroRAG",
    page_icon="🧠",
    layout="wide",
)

st.title("NeuroRAG")
st.caption(
    "Assistente de pesquisa científica baseado em documentos "
    "sobre EEG e interfaces cérebro-computador."
)


def display_sources(sources: list[dict]) -> None:
    """Exibe os trechos recuperados e seus dados de origem."""

    if not sources:
        return

    with st.expander(
        f"Consultar fontes recuperadas ({len(sources)})"
    ):
        for source in sources:
            page = source.get("page", "?")
            distance = source.get("distance")

            st.markdown(
                f"**{source['citation']}** "
                f"— `{source['source']}`, página {page}"
            )

            if distance is not None:
                st.caption(
                    f"Distância vetorial: {distance:.4f} "
                    "(menor geralmente indica maior proximidade)"
                )

            st.write(source["text"])
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Base de conhecimento")

    st.write("Pasta dos PDFs:")
    st.code(str(DOCUMENTS_DIR))

    st.write("Armazenamento vetorial:")
    st.code(str(CHROMA_DIR))

    st.markdown(
        "Para adicionar documentos, coloque PDFs na pasta "
        "indicada e execute `python ingest.py` no terminal."
    )

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Restaura as mensagens anteriores na interface.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# Campo de entrada de uma nova pergunta.
question = st.chat_input(
    "Pergunte algo sobre os artigos científicos..."
)


if question:
    # Guarda e mostra a pergunta do usuário.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Recupera os trechos e gera a resposta da IA.
    with st.chat_message("assistant"):
        with st.spinner(
            "Consultando os documentos e gerando a resposta..."
        ):
            try:
                result = answer_question(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                display_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                error_message = str(error)

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "is_error": True,
                    }
                )