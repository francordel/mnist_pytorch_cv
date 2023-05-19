from torchvision.io import read_image
from torchvision import transforms
import torch
import cv2
imagen = read_image("docs/prueba.png").float()
imagen = transforms.Resize((28,28))(imagen)
imagen = transforms.Grayscale(num_output_channels=1)(imagen)
# Quito la dimensión canales
imagen = imagen.squeeze(1)


# Carga del modelo
modelo_produccion = torch.jit.load("docs/model.zip")
modelo_produccion.eval()
# Se desactiva el cálculo del gradiente de tensores porque sólo hay evaluación
with torch.no_grad():
  # Se pide al modelo la evaluación de la imagen (forward)
  salida = modelo_produccion(imagen)
  # Etiqueta inferida
  label_inferida = salida[1].item()
  # Imprimo etiqueta inferida
  print(label_inferida)
  # Tensor con las probabilidades de todas las clases
  probabilidades = salida[0].squeeze()
  print(probabilidades)
  # Tensor con la probabilidad de la clase inferida
  print(probabilidades[label_inferida])
  