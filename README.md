# Instalacion de Librerias

# Prototipo de Reconocimiento de personas para el registro de asistencia

Proyecto desarrollado para la materia de Tópicos Especiales I / Visión Artificial - 2024

Estudiantes:

* David Rodríguez.
* Anderson González.
* María Donadío.
* Fausto Rivera.

# Resumen del Proyecto:

Este proyecto desarrolla un **sistema automatizado de registro de asistencia** utilizando tecnologías de **reconocimiento facial** e  **inteligencia artificial** .

Se implementan herramientas como  **Python** , **OpenCV** y **TensorFlow** para lograr una **detección precisa de rostros** en un entorno académico.

A través del reconocimiento facial, el sistema:

* Mejora la **precisión** en el registro de asistencia.
* **Reduce el tiempo** de registro manual.
* **Aumenta la seguridad** eliminando la necesidad de métodos tradicionales como listas de firma o tarjetas de identificación.

El objetivo es construir un **prototipo funcional** que será **probado en un aula controlada** con un grupo de  **estudiantes voluntarios** .

Además, el sistema cuenta con una **interfaz web** desarrollada con **Flask** y  **Django** , que permite:

* Visualizar la  **cámara en vivo** .
* Ver el **registro actualizado de asistencias** en tiempo real.

# Instalación de Librerías

Para configurar el entorno y todas las dependencias necesarias, utiliza el archivo `environment.yml` de Conda.

1. Clona este repositorio y navega a la carpeta del proyecto.
2. Crea un entorno virtual con el siguiente comando:

```python
$ conda env create -f environment.yml
```

3. Activa el entorno:
```python
$ conda activate <nombre_del_entorno>
```

# Recomendaciones:

Utilizar un entorno configurado para trabajar con gpu para el entrenamiento del modelo.
