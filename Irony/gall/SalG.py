import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login
import gc

# Limpieza de memoria inicial
gc.collect()
torch.cuda.empty_cache()

# Inicia sesión en Hugging Face Hub (reemplaza 'TU_TOKEN' con tu token real)
login(token='TU_TOKEN')

# Carga el dataset
try:
    df = pd.read_csv("sample_df.csv")
except FileNotFoundError:
    print("Error: El archivo sample_df.csv no se encontró.")
    exit()

# Selecciona los primeros 10 mensajes
mensajes = df["message_gl"].tolist()

# Lista de modelos a iterar
model_names = [
    "BSC-LT/salamandra-7b",
]

for model_id in model_names:
    print(f"Procesando modelo: {model_id}")

    # Configuración del modelo
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # Función para obtener solo la respuesta generada por el modelo
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=dtype)

    def obtener_respuesta_modelo(texto):
        prompt = f"O seguinte texto é irónico? Responde con 'Si' ou 'Non'.\n\nTexto: {texto}\n\nRespuesta:"
        output = pipe(prompt, max_new_tokens=10)
        respuesta_completa = output[0]['generated_text']
        respuesta = respuesta_completa.split("Respuesta:")[-1].strip()
        return respuesta

    # Procesa cada mensaje y guarda los resultados
    resultados = []
    for mensaje in mensajes:
        respuesta_modelo = obtener_respuesta_modelo(mensaje)
        resultados.append({"Mensaje": mensaje, "Respuesta del Modelo": respuesta_modelo})

    # Convierte la lista de resultados en un DataFrame y guarda como CSV
    df_resultados = pd.DataFrame(resultados)
    csv_filename = f"respuestas_{model_id.split('/')[-1]}.csv"  # Nombre del archivo basado en el nombre del modelo
    df_resultados.to_csv(csv_filename, index=False)

    print(f"Respuestas guardadas en {csv_filename}")

    # Limpieza de memoria para el modelo actual
    del pipe
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("Procesamiento de todos los modelos completado.")