# Titanic Data Analysis with Streamlit 🚢

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub Repo stars](https://img.shields.io/github/stars/MrEhDev/Titanic-data-analysis?style=social)
![GitHub forks](https://img.shields.io/github/forks/MrEhDev/Titanic-data-analysis?style=social)

¡Bienvenido a un análisis interactivo y visual del legendario conjunto de datos del **Titanic**! 🌊 Este proyecto te permite explorar factores clave que influyeron en la supervivencia de los pasajeros de manera dinámica y sencilla, todo desarrollado con **Streamlit**, una potente herramienta de Python que facilita la creación de aplicaciones web interactivas sin complicaciones.

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## 🔍 Descripción del proyecto

Esta aplicación no solo te permite explorar visualmente los datos, sino que también incluye un modelo de **Machine Learning** que predice si un pasajero del Titanic hubiera sobrevivido según su perfil. Toda la interfaz ha sido desarrollada con **Streamlit**, una biblioteca de Python que permite crear dashboards y aplicaciones web de forma rápida y sencilla con tan solo escribir código en Python.

Con esta aplicación, podrás:
- **Explorar estadísticas clave** como la supervivencia por clase de pasajeros, sexo, edad y más.
- **Visualizar gráficos interactivos** que facilitan la comprensión de los datos.
- **Hacer predicciones** sobre la supervivencia de un pasajero utilizando un modelo de aprendizaje automático entrenado con el conjunto de datos del Titanic.

## 💡 ¿Por qué Streamlit?

**Streamlit** es la herramienta ideal para proyectos como este, ya que permite construir aplicaciones interactivas sin necesidad de aprender tecnologías web como HTML, CSS o JavaScript. En lugar de eso, simplemente utilizamos **Python** y las bibliotecas de datos que ya conocemos y amamos, como **pandas**, **matplotlib** y **scikit-learn**.

### Características de Streamlit en esta aplicación:
- **Simplicidad**: Con solo unas pocas líneas de código puedes desplegar gráficas y modelos.
- **Interactividad**: Los usuarios pueden ajustar parámetros, como las características de los pasajeros, para generar predicciones personalizadas.
- **Gráficos dinámicos**: Visualiza los resultados de manera instantánea, ajustando diferentes variables de los pasajeros.

## 🧭 Funcionalidades Principales

### 📊 Exploración de Datos
- **Análisis interactivo**: Explora la relación entre factores como la clase del boleto, el género y la edad, y cómo estos influyeron en la probabilidad de supervivencia.
- **Gráficos dinámicos**: Observa cómo las tasas de supervivencia cambian según diferentes grupos de pasajeros, visualizando rápidamente las estadísticas clave.

### 🔮 Modelo de Predicción de Supervivencia
Incluye un **clasificador de Machine Learning** que predice la supervivencia de los pasajeros. Introduce características como la clase, el género, la edad y el número de familiares a bordo, y observa si el modelo predice que el pasajero habría sobrevivido.

### 🌐 Visualización de Tablas
Se ofrece una tabla interactiva que permite explorar los datos de los pasajeros, con opciones para filtrar y ordenar según diferentes criterios.

## 📦 Instalación y Ejecución

Sigue estos pasos para clonar y ejecutar la aplicación en tu máquina local:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tuusuario/titanic-streamlit-analysis.git
   ```

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación**:
   ```bash
   streamlit run titanic_ai_model.py
   ```

4. **Abre la aplicación** en tu navegador en la siguiente dirección: `http://localhost:8501`

## 🔧 Tecnologías Utilizadas

- **Streamlit**: Para la creación de la interfaz web interactiva.
- **Pandas**: Manejo de datos estructurados.
- **Scikit-learn**: Implementación del modelo de Machine Learning.
- **Matplotlib / Seaborn**: Visualización de datos a través de gráficos.
- **HTML/CSS**: Para personalizar la tabla y otros componentes visuales.

## 📈 Variables del conjunto de datos del Titanic

| Variable  | Descripción                                | Valores Clave              |
|-----------|--------------------------------------------|----------------------------|
| `survival`| Supervivencia                               | 0 = No, 1 = Sí             |
| `pclass`  | Clase del billete                           | 1 = 1ª, 2 = 2ª, 3 = 3ª     |
| `sex`     | Género                                      |                            |
| `age`     | Edad                                        |                            |
| `sibsp`   | Nº de hermanos/esposos a bordo              |                            |
| `parch`   | Nº de padres/hijos a bordo                  |                            |
| `fare`    | Tarifa del billete                          |                            |
| `embarked`| Puerto de embarque                          | C = Cherburgo, Q = Queenstown, S = Southampton |

## 🌟 Valor del proyecto

Este proyecto no solo sirve como una herramienta educativa para explorar datos históricos, sino que también pone en práctica conceptos clave de **análisis de datos**, **Machine Learning**, y **desarrollo de aplicaciones web** con **Streamlit**. La simplicidad y elegancia de Streamlit nos permite crear potentes herramientas interactivas sin salir del entorno de Python, lo que lo convierte en una solución ideal para científicos de datos que desean compartir sus hallazgos de manera accesible.

## 🚀 Despliegue en Streamlit Cloud

Esta aplicación puede ser desplegada fácilmente en [Streamlit Cloud](https://streamlit.io/cloud) para compartirla con otros. Solo necesitas enlazar el repositorio de GitHub y Streamlit Cloud se encargará del resto.
