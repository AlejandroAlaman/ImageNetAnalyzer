import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import torch.nn.functional as F

# Cargar el modelo preentrenado ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Definir las transformaciones necesarias para las imágenes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar las etiquetas de ImageNet
json_path = os.path.join(os.path.dirname(__file__), 'imagenet_class_index.json')
with open(json_path, "r") as f:
    class_idx = json.load(f)

# Función para preprocesar y hacer predicción con el modelo
def predict_image(img):
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)

    probabilities = F.softmax(outputs, dim=1)
    top3_probs, top3_classes = torch.topk(probabilities, 3)
    top3_class_names = [class_idx[str(idx.item())][1] for idx in top3_classes[0]]
    top3_probs = top3_probs[0].numpy()

    return top3_class_names, top3_probs, probabilities

# Función para calcular la entropía de la distribución de probabilidades
def compute_entropy(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).item()