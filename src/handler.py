from torchvision import transforms
from torchvision.transforms import functional
import torch
import cv2
import logging

# Recieve the route of the model we want to load and the id of the camera we are working with
def init(model_route, id_camera):
    # Loading trained model
    model = torch.jit.load(model_route)
    # Necessary to do this when we are not training
    model.eval()
    # Initialization of the video capture
    vid = cv2.VideoCapture(id_camera)
    # Necessary to do this when we are not training
    torch.set_grad_enabled(False)
    return model, vid

# Recieve a video capture and return the frame and the image of the an instant t
def get_image(vid):
    ret, frame = vid.read()
    # Transform to tensor
    image = transforms.ToTensor()(frame)
    # image and frame that the video capture is seeing at this moment
    return frame, image

# Prepocessing the image before feeding the model
def transform_frame(image):
    # Obtener la dimensión menor de la image
    min_size = min(image.size(dim=1), image.size(dim=2))
    # Obtener la dimensión mayor de la image
    max_size = max(image.size(dim=1), image.size(dim=2))
    # Recortar el cuadrado central
    image = functional.crop(
        image, 0, (max_size-min_size)//2, min_size, min_size)
    # Convertir en escala de grises
    image = transforms.Grayscale(num_output_channels=1)(image)
    # Invertir la image para fondo oscuro y trazos claros
    image = functional.invert(image)
    # Contraste
    image = functional.autocontrast(image)
    # Resize to 28x28
    image = transforms.Resize((28, 28))(image)
    # Escalar valores a 0-255
    image = image*255.0
    return image

#Show the frame with the output info of the instant t 
def visual_testing(salida, image, frame, count):
    # Obtener array con la imagen de salida
    tratada = image.squeeze().unsqueeze(dim=-1)
    # Concatenar tres veces la imagen para pasar a tres canales
    tratada = torch.cat([tratada, tratada, tratada], dim=2).numpy()
    # Superponer en el frame la imagen
    frame[0:28, 0:28] = tratada
    # Etiqueta inferida
    label_inferida = salida[2].item()
    # Tensor con las probabilidades de todas las clases
    probabilidades = salida[1].squeeze()
    # Raw certainty
    certainty = salida[0].squeeze()
    #Debugging
    logging.basicConfig(filename='./logs/log_filename.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    text_debugging = f'ID:{count} Guest:{label_inferida} - probabilities: {probabilidades} - certainty: {certainty}'
    logging.debug(text_debugging)
    image_text = f'{label_inferida} - {certainty}'
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
    frame = cv2.putText(frame, image_text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    return tratada


