import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        
        Implementacion de ArcFace
        
        Args:
            in_features: Tamaño del embedding vector (input)
            out_features: Números de clases
            s: Factor de escalado
            m: margen añadido entre clases en el espacio angular
            easy_margin: Usar solamente si la version base se vuelve inestable
        """

        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) # inicializa la clase para los weights

        # Calcula cos(m) y sen(m) para temas de eficencia
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) # umbral para el margen de decision
        self.mm = math.sin(math.pi - m) * m # margen de penalización

    def forward(self, input, label):
        """
        
        Forward prop para ArcFace
        
        Args:
            input: Inputs las dimensiones del embedding tensor [batch_size, in_features]
            label: Labels con la dimension del [batch_size]
        Returns:
            Output: Logits (resultado tras pasar por una funcion de activacion) with shape [batch_size, out_features] to pass it to the CrossEntropyLoss
        """ 

        # Normalize both inputs features and weight matrix
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # Cosine similarity between features and weights
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 0.99999)) # sin(θ) from cos(θ)

        # Compute cos(θ + m) using trigonometric identity
        phi = cosine * self.cos_m - sine * self.sin_m

        # Decide whether to apply margin based on thresholding (remember just use it, if the model becomes unstable)
        if self.easy_margin:
            # Use cosine if it is positive, else keep original
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Use original phi only if above threshold, else apply modified margin
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One Hot to enconde labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # Apply arc margin only to the correct class logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output