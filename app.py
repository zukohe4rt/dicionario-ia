import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login("")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando o dispositivo: {device}")

print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

print("Modelo carregado! Iniciando o loop...")

model.to(device)

while True:
    input_text = input("Digite uma pergunta ou frase (ou 'sair' para encerrar): ")

    if input_text.lower() == 'sair':
        print("Encerrando a aplicação.")
        break

    inputs = tokenizer(input_text, return_tensors="pt")  # return_tensors="pt" para PyTorch

    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Resposta do modelo:", generated_text)
