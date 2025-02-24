# 🌟 Cuantización de Modelos en PyTorch: ¡Reduciendo la Complejidad sin Sacrificar Precisión! 🚀
## Introducción
La cuantización de modelos es una técnica esencial en el campo del aprendizaje automático que permite reducir el tamaño y la complejidad de los modelos sin sacrificar significativamente su precisión. En este README, exploraremos cómo implementarla utilizando PyTorch, y aprenderemos a aplicar diferentes técnicas de cuantización para mejorar el rendimiento de nuestros modelos. ¡Prepárate para llevar tus modelos al siguiente nivel!

## Técnicas de Cuantización:

### 🧠 Cuantización Estática
En esta técnica, se analizan los datos de entrada y salida del modelo antes de la inferencia para determinar los rangos adecuados para la cuantización. Los pesos y las activaciones se cuantizan utilizando estos rangos predefinidos, permitiendo optimizaciones adicionales.
```
import torch
import torch.quantization

# Crear un modelo sencillo
class ModeloSencillo(torch.nn.Module):
    def __init__(self):
        super(ModeloSencillo, self).__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Instanciar el modelo y prepararlo para cuantización
model = ModeloSencillo()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Simular datos de calibración
datos_calibracion = torch.randn(100, 10)
model(datos_calibracion)

# Cuantizar el modelo
torch.quantization.convert(model, inplace=True)

# Mostrar el modelo cuantizado
print(model)
```
### ⚡ Cuantización Dinámica
En esta técnica, los pesos del modelo se cuantizan antes de la inferencia, pero las activaciones se cuantizan dinámicamente durante la inferencia. Esta técnica es útil para modelos que se ejecutan en dispositivos con capacidades de cómputo más limitadas.
```
import torch.quantization

# Cuantización dinámica del modelo
model_dynamic = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Mostrar el modelo cuantizado dinámicamente
print(model_dynamic)
```
### 🔥 Cuantización de Entrenamiento con Conciencia Cuantizada (QAT)
Esta técnica ajusta el modelo durante el entrenamiento para que sea robusto a la cuantización. Aunque es más compleja, puede resultar en modelos cuantizados con mejor desempeño.
```
!pip install torch

# Importamos las librerías necesarias
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import TensorDataset, DataLoader

# Definimos el dispositivo (GPU si está disponible, CPU en caso contrario)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear un modelo sencillo
class ModeloSencillo(nn.Module):
    def __init__(self):
        super(ModeloSencillo, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Instanciar el modelo y moverlo al dispositivo
model = ModeloSencillo().to(device)

# Preparar el modelo para QAT (Quantization Aware Training)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Definir datos simulados de entrenamiento y moverlos al dispositivo
datos = torch.randn(100, 10).to(device)
etiquetas = torch.randint(0, 2, (100,)).to(device)
dataset = TensorDataset(datos, etiquetas)
dataloader = DataLoader(dataset, batch_size=10)

# Definir la función de pérdida
criterion = nn.CrossEntropyLoss()

# Definir el optimizador
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Entrenar el modelo
for epoch in range(10):
    for datos, etiquetas in dataloader:
        optim.zero_grad()
        salida = model(datos)
        perdida = criterion(salida, etiquetas)
        perdida.backward()
        optim.step()

# Convertir a modelo cuantizado
torch.quantization.convert(model, inplace=True)

# Mostrar el modelo cuantizado
print(model)
```
## Conclusión
La cuantización de modelos es una técnica poderosa que puede mejorar significativamente el rendimiento de tus modelos, especialmente en entornos con recursos limitados. Con PyTorch, implementar estas técnicas es sencillo y eficiente. 

## 🏷️ Licencia
Este proyecto está bajo la licencia MIT. Consulte el archivo de LICENCIA para obtener más detalles.
