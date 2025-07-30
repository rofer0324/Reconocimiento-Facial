# 🧠 Prototipo de Reconocimiento Facial para Registro de Asistencia

## 👥 Estudiantes

- David Rodríguez
- Fausto Rivera
- Miguel Hidalgo

---

## 📌 Resumen del Proyecto

Este proyecto desarrolla un **sistema automatizado de registro de asistencia** basado en **reconocimiento facial** e **inteligencia artificial**.  

Utilizando herramientas como **Python**, **OpenCV**, **PyTorch** y el modelo **ArcFace**, se busca lograr una **detección precisa de rostros** en entornos académicos.

El sistema tiene como objetivos principales:

- ✅ Mejorar la **precisión** del registro de asistencia.
- ⏱️ **Reducir el tiempo** requerido en procesos manuales.
- 🔐 **Aumentar la seguridad**, eliminando listas físicas y tarjetas de identificación.

Este prototipo será **probado en un aula controlada** con un grupo de **estudiantes voluntarios**.

### 🔍 Tecnologías Implementadas

- **Python** para el desarrollo general.
- **OpenCV** para captura y procesamiento de imágenes.
- **PyTorch** para la implementación de modelos de aprendizaje profundo.
- **ArcFace** para el reconocimiento facial de alta precisión.  
  👉 Referencia: [https://insightface.ai/arcface](https://insightface.ai/arcface)
- **Flask** para el desarrollo de una interfaz web ligera.

### 🌐 Funcionalidades de la Interfaz Web

- Visualización de la **cámara en vivo**.
- Consulta del **registro de asistencias en tiempo real**.

---

## ⚙️ Instalación de Librerías

El entorno de trabajo está gestionado mediante **Conda**. Asegúrate de tenerlo instalado antes de continuar.

1. Crea el entorno virtual usando el archivo `env_asistencia.yml`:
   ```bash
   conda env create -f env_asistencia.yml
   ```

2. Activa el entorno:
   ```bash
   conda activate asistencia
   ```
---
## 🧬 Entrenamiento del Modelo
El entrenamiento del modelo de reconocimiento facial se realiza utilizando una implementación personalizada de ArcFace basada en el siguiente repositorio:

🔗 Repositorio Base: https://github.com/CharizardLyon/ArcFace-Implementation

## ⚙️ Configuración utilizada (config/default.yml)
```yml
# Training settings
epochs: 10
batch_size: 32
learning_rate: 0.001
device: cuda
checkpoint_dir: checkpoints

# Model settings
backbone: resnet50
embedding_size: 512
num_classes: 17

# Arcface settings
arcface_margin: 0.5
arcface_scale: 45
arcface_easy_margin: false

# Data paths
train_data_path: ../Reconocimiento-Facial/data/preprocessed/train
val_data_path: ../Reconocimiento-Facial/data/preprocessed/test
seed: 42
```

Ejecuta el entrenamiento con:

```bash
python main.py
```
Esto generará los pesos del modelo en la carpeta `checkpoints/`.

## 🖼️ Generación de Embeddings
1. Prepara una carpeta inference_images/ con una imagen por persona (ej.: Alice.jpg, Bob.jpg, etc.).

2. Ejecuta el siguiente comando para generar los embeddings:

```bash
python precompute_embeddings.py --checkpoint checkpoints/model_epoch.pth --image_dir inference_images --save_dir embeddings
```
Esto generará los archivos:

* `embeddings/embeddings.npy`: vectores faciales generados por el modelo.

* `embeddings/paths.npy`: rutas correspondientes a cada vector de embedding.

⚠️ Estos archivos son esenciales para la etapa de inferencia en tiempo real.

---

## 🌐 Interfaz Web (Flask GUI)

El repositorio base también incluye una interfaz con Flask que puedes ejecutar localmente para visualizar los resultados:

```bash
cd GUI
python app.py
```

Luego abre tu navegador en:
* 📍 http://localhost:5000

Allí podrás ver:

* 🎥 Video en vivo desde la webcam

* 🧑‍🤝‍🧑 Nombres detectados en pantalla

* 📋 Registro de asistencias con marcas de tiempo

* 📤 Opción para exportar los datos a CSV

✅ Recuerda: los embeddings deben ser generados antes de usar la GUI.

---

## 🧠 Recomendaciones

- Se recomienda ejecutar este proyecto en una máquina con **soporte GPU**, especialmente para el entrenamiento o inferencia con modelos como ArcFace.
- Asegúrate de contar con los drivers y bibliotecas CUDA adecuados si usas una GPU NVIDIA.

---

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.  
Consulta el archivo [LICENSE](./LICENSE) para más detalles.
