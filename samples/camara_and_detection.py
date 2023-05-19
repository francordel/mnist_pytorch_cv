from torchvision import transforms
from torchvision.transforms import functional
import torch
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)
currentframe = 0
# Carga del modelo
mnist_model = torch.jit.load("./docs/model.zip")
mnist_model.eval()
# Se desactiva el cálculo del gradiente de tensores porque sólo hay evaluación
with torch.no_grad():

    while(True):

        # Capture the video frame by frame
        ret, frame = vid.read()

        # Transform to tensor
        imagen = transforms.ToTensor()(frame)
        # Obtener la dimensión menor de la imagen
        min_size = min(imagen.size(dim=1),imagen.size(dim=2))
        # Obtener la dimensión mayor de la imagen
        max_size = max(imagen.size(dim=1),imagen.size(dim=2))
        # Recortar el cuadrado central
        imagen = functional.crop(imagen,0,(max_size-min_size)//2,min_size,min_size)
        # Convertir en escala de grises
        imagen = transforms.Grayscale(num_output_channels=1)(imagen)
        # Invertir la imagen para fondo oscuro y trazos claros
        imagen = functional.invert(imagen)
        # Contraste
        imagen = functional.autocontrast(imagen)
        # Resize to 28x28
        imagen = transforms.Resize((28,28))(imagen)
        # Escalar valores a 0-255
        imagen = imagen*255.0

        # Texto que se sobreimpresiona en la imagen
        texto_imagen = ''

        # Se pide al modelo la evaluación de la imagen (forward)
        salida = mnist_model(imagen)
        # Obtener array con la imagen de salida
        tratada = imagen.squeeze().unsqueeze(dim=-1)
        # Concatenar tres veces la imagen para pasar a tres canales
        tratada = torch.cat([tratada, tratada, tratada], dim=2).numpy()
        # Superponer en el frame la imagen
        frame[0:28,0:28] = tratada
        # Etiqueta inferida
        label_inferida = salida[2].item()
        # Tensor con las probabilidades de todas las clases
        probabilidades = salida[1].squeeze()
        # Raw certainty
        certainty = salida[0].squeeze()

        texto_imagen = f'{label_inferida} - {certainty}'

        # Nombre de la ventana
        window_name = 'Reconocimiento'
        # font
        font = cv2.FONT_HERSHEY_PLAIN
        # org
        org = (40, 25)
        # fontScale
        fontScale = 1
        # White color in BGR
        color = (255, 255, 255)
        # Line thickness of 1 px
        thickness = 1
        # Using cv2.putText() method
        frame = cv2.putText(frame, texto_imagen, org, font, fontScale, color, thickness, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
 
  
  
