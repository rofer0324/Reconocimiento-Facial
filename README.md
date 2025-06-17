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

1. Clona este repositorio y navega al directorio del proyecto:
   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-proyecto>
   ```

2. Crea el entorno virtual usando el archivo `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```

3. Activa el entorno:
   ```bash
   conda activate <nombre_del_entorno>
   ```

---

## 🧠 Recomendaciones

- Se recomienda ejecutar este proyecto en una máquina con **soporte GPU**, especialmente para el entrenamiento o inferencia con modelos como ArcFace.
- Asegúrate de contar con los drivers y bibliotecas CUDA adecuados si usas una GPU NVIDIA.

---

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.  
Consulta el archivo [LICENSE](./LICENSE) para más detalles.

