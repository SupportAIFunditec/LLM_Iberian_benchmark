import subprocess
import time
import os

# Leer modelos desde el archivo
with open("tareas_modelos_funditec/modelos.txt", "r") as f:
    modelos = [line.strip() for line in f if line.strip()]

# Leer tareas desde el archivo
with open("tareas_modelos_funditec/tareas.txt", "r") as f:
    tareas = [line.strip() for line in f if line.strip()]

# Asegurar que el directorio de resultados existe
os.makedirs("resultados_pruebas/tiempos", exist_ok=True)

# Iterar sobre modelos y tareas
for modelo in modelos:
    nombre_modelo = modelo.split("/")[-1]
    for tarea in tareas:
        output_filename = f"{tarea}_{nombre_modelo}.txt"
        output_directory = f"resultados_pruebas/tiempos/{output_filename}"
        time_log_filename = f"resultados_pruebas/tiempos/tiempo_{tarea}_{nombre_modelo}.txt"

        #Carga de modelo
        command_carga = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={modelo},load_in_4bit=True",
            "--tasks", "belebele_cat_Latn",
            "--device", "cuda:0",
            "--batch_size", "1",
            "--limit", "1"
        ]
        
        # Construir el comando
        command = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={modelo},load_in_4bit=True",
            "--tasks", tarea,
            "--device", "cuda:0",
            "--batch_size", "1"
            #,"--limit", "1000"
        ]
        
        # Ejecutar el comando y capturar la salida
        try:

            result_carga=subprocess.run(
                command_carga, 
                capture_output=True, 
                text=True, 
                check=True
            )

            print(f"Realizando tarea {tarea} para el modelo {modelo}")
            start_time = time.time()
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Guardar la salida en un archivo
            with open(output_directory, "w") as output_file:
                output_file.write(result.stdout)
            
            # Guardar el tiempo de ejecución
            with open(time_log_filename, "w") as time_file:
                time_file.write(f"Tiempo de ejecución: {execution_time:.4f} segundos\n")
            
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar el comando para {modelo} con {tarea}: {e}")

    # Limpiar la caché de Hugging Face después de procesar cada modelo
    try:
        subprocess.run(["rm", "-rf", "~/.cache/huggingface"], check=True)
        print(f"Caché de Hugging Face eliminada después de procesar el modelo {modelo}.")
    except subprocess.CalledProcessError as e:
        print(f"Error al intentar eliminar la caché de Hugging Face: {e}")

print("Ejecución finalizada.")
