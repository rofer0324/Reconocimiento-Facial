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
## 🚀 Pasos para Ejecutar el Proyecto

1. Preparar la estructura del proyecto

Verifica que la carpeta Data existe y contiene la subcarpeta crudo con los videos sin procesar de los rostros.

2. Ejecutar el script de detección de rostros

* Abre y ejecuta el archivo `Deteccion_Rostros.py`.

* Antes de ejecutarlo, edita el diccionario que define los nombres de las personas y los videos correspondientes.

3. Entrenar el modelo

* Ejecuta el cuaderno Jupyter `train_model3_arcface_torch.ipynb`.

* Este proceso generará los embeddings y ajustará el modelo con las clases definidas.

4. Evaluar el modelo

* Utiliza el cuaderno `Evaluacion_arcface.ipynb` para verificar el desempeño del modelo con las clases entrenadas.

5. Inferencia en vivo y registro de asistencia

* Ejecuta el cuaderno `inference.ipynb`.

* Realizará el reconocimiento facial en tiempo real.

* Registrará la primera y última vez que cada persona fue detectada en asistencia.csv.

---

## 🧠 Recomendaciones

- Se recomienda ejecutar este proyecto en una máquina con **soporte GPU**, especialmente para el entrenamiento o inferencia con modelos como ArcFace.
- Asegúrate de contar con los drivers y bibliotecas CUDA adecuados si usas una GPU NVIDIA.

---

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.  
Consulta el archivo [LICENSE](./LICENSE) para más detalles.

