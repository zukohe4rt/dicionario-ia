import streamlit as st
from gradio_client import Client
from PIL import Image
import os

client = Client("mukaist/Midjourney")

# Função para enviar a mensagem ao modelo Gradio
def generate_image(user_prompt):
    result = client.predict(
        prompt=user_prompt,
        negative_prompt=(
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, "
            "sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, "
            "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, "
            "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, "
            "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, "
            "disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, "
            "extra legs, fused fingers, too many fingers, long neck"
        ),
        use_negative_prompt=True,
        style="2560 x 1440",
        seed=0,
        width=1024,
        height=1024,
        guidance_scale=6,
        randomize_seed=True,
        api_name="/run"
    )

    if result:
        return result
    else:
        return "Erro ao gerar a imagem."

st.title("Geração de Imagens com Midjourney")

user_prompt = st.text_input("Digite seu prompt:")

if user_prompt:
    response = generate_image(user_prompt)
    
    if isinstance(response, list) and len(response) > 0:
        for item in response:
            image_path = item['image']
            if os.path.exists(image_path):
                # Exibe a imagem usando PIL
                image = Image.open(image_path)
                st.image(image, caption=item['caption'], use_column_width=True)
            else:
                st.error(f"Imagem não encontrada: {image_path}")
    else:
        st.write(response)
