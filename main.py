import streamlit as st
from gradio_client import Client
import time
import pyperclip  # Biblioteca para copiar para a área de transferência

client = Client("chuanli11/Chat-Llama-3.2-3B-Instruct-uncensored")

def get_model_response(word):
    user_message = f"Por favor, forneça a definição, sinônimos e um exemplo de uso da palavra: {word}"

    result = client.predict(
        message=user_message,
        system_prompt="Você é um dicionário completo. Forneça definição, sinônimos e um exemplo de uso para a palavra solicitada.",
        max_new_tokens=1024,
        temperature=0.6,
        api_name="/chat"
    )

    if isinstance(result, str):
        return result.replace("assistant", "").strip()
    else:
        return "Erro ao processar a resposta."

def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Texto copiado para a área de transferência!")

hide_header_style = """
    <script>
    window.addEventListener('load', function() {
        var header = document.querySelector('header');
        if (header) {
            header.style.display = 'none';  // Esconde o cabeçalho assim que a página carrega
        }
    });
    </script>
"""

st.markdown(hide_header_style, unsafe_allow_html=True)

st.title("Dicionário I.A para meu amor ❤️")

input_placeholder = st.empty()

if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

if 'word' not in st.session_state:
    st.session_state.word = ''

if not st.session_state.is_loading:
    with input_placeholder:
        word = st.text_input("Digite a palavra que deseja pesquisar:", key="input_word")
        if word:
            st.session_state.word = word

if st.session_state.word:
    st.session_state.is_loading = True  # Definir que está processando
    input_placeholder.empty()  # Esconder o campo de input enquanto processa

    with st.spinner("Processando sua consulta..."):
        response = get_model_response(st.session_state.word)
        time.sleep(1)  # Apenas para garantir que o spinner seja exibido por um tempo
    st.session_state.is_loading = False  # Processamento concluído

    st.write(f"{response}")

    with input_placeholder:
        word = st.text_input("Digite a palavra que deseja pesquisar:", key="input_word_2")