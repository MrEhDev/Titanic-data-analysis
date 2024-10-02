# Titanic Data Analysis with Streamlit 🚢

Welcome to an interactive and visual analysis of the famous **Titanic dataset**! 🌊 This project allows you to explore key factors that influenced passenger survival dynamically and effortlessly, all developed with **Streamlit**, a powerful Python tool that simplifies the creation of interactive web applications.

![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## 🔍 Project Overview

This application not only lets you visually explore the data, but also includes a **Machine Learning model** that predicts whether a passenger would have survived based on their profile. The entire interface is developed using **Streamlit**, a Python library that allows you to quickly and easily build web apps and dashboards by just writing Python code.

With this application, you can:
- **Explore key statistics** such as survival rates by passenger class, gender, age, and more.
- **Visualize interactive charts** to gain better insights into the data.
- **Make predictions** about a passenger’s survival using a machine learning model trained on the Titanic dataset.

## 💡 Why Streamlit?

**Streamlit** is the ideal tool for projects like this because it allows us to build interactive applications without needing to learn web technologies like HTML, CSS, or JavaScript. Instead, we use **Python** and familiar data libraries like **pandas**, **matplotlib**, and **scikit-learn**.

### Key Streamlit Features in this Application:
- **Simplicity**: With just a few lines of code, you can deploy charts and models.
- **Interactivity**: Users can adjust parameters, like passenger features, to generate personalized predictions.
- **Dynamic Visualizations**: Instantly visualize results by tweaking various passenger characteristics.

## 🧭 Main Features

### 📊 Data Exploration
- **Interactive analysis**: Explore the relationship between factors such as ticket class, gender, and age, and how they influenced survival probability.
- **Dynamic charts**: See how survival rates change across different passenger groups, with key statistics visualized quickly.

### 🔮 Survival Prediction Model
This application includes a **Machine Learning classifier** that predicts passenger survival. Enter features such as class, gender, age, and the number of family members on board to see whether the model predicts the passenger would have survived.

### 🌐 Data Table Visualization
An interactive table lets you explore passenger data with options to filter and sort by different criteria.

## 📦 Installation and Usage

Follow these steps to clone and run the application locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/titanic-streamlit-analysis.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run titanic_ai_model.py
   ```

4. **Open the app** in your browser at the following address: `http://localhost:8501`

## 🔧 Technologies Used

- **Streamlit**: For building the interactive web interface.
- **Pandas**: For handling structured data.
- **Scikit-learn**: For the Machine Learning model implementation.
- **Matplotlib / Seaborn**: For data visualization through graphs.
- **HTML/CSS**: For customizing the table and other visual components.

## 📈 Titanic Dataset Variables

| Variable  | Definition                                 | Key Values                  |
|-----------|--------------------------------------------|-----------------------------|
| `survival`| Survival                                   | 0 = No, 1 = Yes             |
| `pclass`  | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd   |
| `sex`     | Gender                                     |                             |
| `age`     | Age                                        |                             |
| `sibsp`   | Number of siblings/spouses aboard          |                             |
| `parch`   | Number of parents/children aboard          |                             |
| `fare`    | Fare                                       |                             |
| `embarked`| Port of embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

## 🌟 Project Value

This project not only serves as an educational tool for exploring historical data but also applies key concepts in **data analysis**, **Machine Learning**, and **web application development** with **Streamlit**. The simplicity and elegance of Streamlit allow us to create powerful interactive tools while staying entirely within the Python environment, making it an ideal solution for data scientists looking to share their findings in an accessible way.

## 🚀 Deployment on Streamlit Cloud

This application can be easily deployed on [Streamlit Cloud](https://streamlit.io/cloud) for sharing with others. Simply connect the GitHub repository, and Streamlit Cloud will handle the rest.

