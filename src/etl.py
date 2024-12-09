from PIL import Image
import os

# Función para cargar una imagen desde una ruta específica
def load_image(img_path):
    return Image.open(img_path)

# Función para obtener una lista de imágenes desde una carpeta específica
def get_image_files(folder, extensions=('.jpg', '.jpeg', '.png')):
    return [f for f in os.listdir(folder) if f.endswith(extensions)]