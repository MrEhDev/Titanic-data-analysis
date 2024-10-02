import streamlit as st
import pandas as pd
import copy
import plotly.express as px
from titanic_ai_model import get_model


def clean_data(df):
    survived_dict = { 1: 'Sí', 0: 'No'}
    pclass_dict = {1: 'Primera clase', 2: 'Segunda clase', 3: 'Tercera clase'}
    sex_dict = {'male' : 'Hombre', 'female' : 'Mujer'}
    embarkament_dict = { 'C' : 'Cherbourg', 'Q' : 'Queenstown', 'S' : 'Southampton'}

    df.replace({'Survived': survived_dict}, inplace = True)
    df.replace({'Sex': sex_dict}, inplace = True)
    df.replace({'Pclas': pclass_dict}, inplace = True)
    df.replace({'Embarked': embarkament_dict}, inplace = True)

    df.dropna(subset = ['Fare'], inplace = True)
    df.dropna(subset = ['Age'], inplace = True)
    df.dropna(subset = ['Embarked'], inplace = True)

    df['count'] = 1

    return df


@st.cache_data
def get_data(data):
    df = pd.read_csv(data)
    df = clean_data(df)

    return df

@st.cache_data
def get_values(col):
    
    return sorted(st.session_state['df'][col].unique())

def update_df():
    st.session_state['df_filtrado'] = st.session_state['df'][
        (st.session_state["df"]["Survived"].isin(st.session_state["Survived"])) &
        (st.session_state["df"]["Pclass"].isin(st.session_state["Pclass"])) &
        (st.session_state["df"]["Sex"].isin(st.session_state["Sex"])) &
        (st.session_state["df"]["Embarked"].isin(st.session_state["Embarked"])) & 
        ((st.session_state["df"]["Age"] >= st.session_state["Age"][0]) & (st.session_state["df"]["Age"] <= st.session_state["Age"][1])) &
        ((st.session_state["df"]["SibSp"] >= st.session_state["SibSp"][0]) & (st.session_state["df"]["SibSp"] <= st.session_state["SibSp"][1])) &
        ((st.session_state["df"]["Parch"] >= st.session_state["Parch"][0]) & (st.session_state["df"]["Parch"] <= st.session_state["Parch"][1])) &
        ((st.session_state["df"]["Fare"] >= st.session_state["Fare"][0]) & (st.session_state["df"]["Fare"] <= st.session_state["Fare"][1])) 
    ]

def generate_plot(var1, var2,var3, color_var, num_var, plot_type):
    if plot_type == 'Bar':

        fig = px.bar(
            st.session_state['df_filtrado'],
            x = var1,
            y = num_var,
            color = color_var,
        )
    elif plot_type == 'Pie':
        
        fig = px.pie(
            st.session_state['df_filtrado'],
            values= num_var,
            names= var1,
        )

    elif plot_type == 'Scatter':
        
        fig = px.scatter(
            st.session_state['df_filtrado'],
            x = var1,
            y = var2,
            size= num_var,
            color= color_var
        )

    elif plot_type == 'Heatmap':
        
        fig = px.density_heatmap(
            st.session_state['df_filtrado'],
            x = var1,
            y = var2,
            z = num_var,
            text_auto= True,
        )

    elif plot_type == 'Treemap':
        
        fig = px.treemap(
            st.session_state['df_filtrado'],
            path = [var1, var2, var3],
            values= num_var,
            color= color_var,
        )

    return fig



st.set_page_config(
    page_title= 'Streamlit Titanic', 
    page_icon='🚢', 
    layout="wide", 
    initial_sidebar_state="expanded", 
    menu_items=None
)



if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

if 'df_filtrado' not in st.session_state:
    st.session_state['df_filtrado'] = pd.DataFrame()


st.header('Datos de pasajeros del Titanic 🚢')
data = './titanic-train.csv'
st.session_state['df'] = get_data(data)


def page_1():
    st.subheader('🚢 - Descripción de datos')
    with  st.expander("Descripción"):
        st.markdown(
            open(r'./titanic-data-analysis.html').read(),
            unsafe_allow_html=True

        )
    with  st.expander("Dataframe"):
        st.write(st.session_state['df'])
    with  st.expander("Panda describe"):
        st.write(st.session_state['df'].describe())
    

def page_2():
    st.subheader('📈 - Análisis de datos')

    if st.session_state['df_filtrado'].empty:
        st.session_state['df_filtrado'] = copy.copy(st.session_state['df'] )

    col_plot_1, col_plot_2 = st.columns([5,1])

    with col_plot_1:

        fig_plot = st.empty()

    with col_plot_2:

        plot_type = st.selectbox(
            "Tipo de gráfico",
            options=['Bar', 'Pie','Scatter', 'Heatmap', 'Treemap'],
        )

        var1 = st.selectbox(
            '1ª Variable',
            options= st.session_state['df_filtrado'].columns,
        )

        num_var = st.selectbox(
            'Variable Numérica',
            options= ['count', 'Fare', 'Age', 'Sibsp', 'Parch']
        )

        color_var = st.selectbox(
            'Variable de color',
            options= st.session_state['df_filtrado'].columns,
        )

        var2 = st.selectbox(
            '2ª Variable',
            options= st.session_state['df_filtrado'].columns,
        )

        var3 = st.selectbox(
            '3ª Variable',
            options= st.session_state['df_filtrado'].columns,
        )

            # Filtros de la página 2
    with st.expander('Filtros'):

        with st.form(key= 'filter_form'):

            col_fil_1,col_fil_2 = st.columns([1,1])
                        
            with col_fil_1:
                surv_values = get_values('Survived')
                sel_surv = st.multiselect(
                    'Sobreviviente',
                    options= surv_values,
                    help='Sobreviviente: Sí o No',
                    default= surv_values,
                    key= 'Survived',
                )

                pclass_values = get_values('Pclass')
                sel_pclass = st.multiselect(
                    'Clase',
                    options= pclass_values,
                    help='Clase',
                    default= pclass_values,
                    key= 'Pclass',
                )

                sex_values = get_values('Sex')
                sel_sex = st.multiselect(
                    'Sexo',
                    options= sex_values,
                    help='Sexo',
                    default= sex_values,
                    key= 'Sex',
                )

                embarked_values = get_values('Embarked')
                sel_embarked = st.multiselect(
                    'Embarque',
                    options= embarked_values,
                    help='Puerta de embarque',
                    default= embarked_values,
                    key= 'Embarked',
                )
            
            with col_fil_2:

                age_value = get_values('Age')
                sel_age = st.slider(
                    'Edad',
                    min_value= min(age_value),
                    max_value= max(age_value),
                    value= [min(age_value), max(age_value)],
                    key= 'Age',
                )
                
                sibsp_value = get_values('SibSp')
                sel_sib = st.slider(
                    'Hermanos a bordo',
                    min_value= min(sibsp_value),
                    max_value= max(sibsp_value),
                    value= [min(sibsp_value), max(sibsp_value)],
                    key= 'SibSp',
                )
                
                parch_value = get_values('Parch')
                sel_parch = st.slider(
                    'Hijos a bordo',
                    min_value= min(parch_value),
                    max_value= max(parch_value),
                    value= [min(parch_value), max(parch_value)],
                    key= 'Parch',
                )
                
                fare_value = get_values('Fare')
                sel_fare = st.slider(
                    'Tarifa pagada',
                    min_value= min(fare_value),
                    max_value= max(fare_value),
                    value= [min(fare_value), max(fare_value)],
                    key= 'Fare',
                )
        
            submit = st.form_submit_button('Filtrar')

        if submit:
            update_df()



    fig = generate_plot(var1, var2,var3, color_var, num_var, plot_type)
    fig_plot.write(fig)

    st.write(st.session_state['df_filtrado'])


def page_3():
    st.subheader('🤖 - Predicción de supervivencia mediante Inteligencia artificial')
    model= get_model
    
    with st.form("prediction form"):
    
        col_pred_1, col_pred_2, col_pred_3 = st.columns([1,1,1])
    
        with col_pred_1:
            class_input = st.selectbox(
                    "Clase",
                    options = [1,2,3]
                    )
                    
            age_input = st.number_input(
                    "Edad",
                    min_value = 0.0,
                    max_value = 100.0,
                    )
                    
            sibsp_input = st.number_input(
                    "Hermanos embarcados",
                    min_value = 0,
                    max_value = 10,
                    )
                    
        with col_pred_2:
            parch_input = st.number_input(
                    "Hijos embarcados",
                    min_value = 0,
                    max_value = 10,
                    )
                    
            fare_input = st.number_input(
                    "Tarifa",
                    min_value = 0.0,
                    max_value = 1000.0,
                    )
        
        with col_pred_3:        
            sex_input = st.toggle(
                    "Sexo",
                    )
                    
            q_input = st.toggle(
                    "Embarcado en Queenstown",
                    )
                    
            s_input = st.toggle(
                    "Embarcado en Southhampton",
                    )

            submit_prediction = st.form_submit_button("Predecir")
        
        if submit_prediction:
            
            input_vector = [[
                    class_input,
                    age_input,
                    sibsp_input,
                    parch_input,
                    fare_input,
                    sex_input,
                    q_input,
                    s_input,
                    ]]
            
            y_pred = model.predict(input_vector)

            if y_pred:
                
                st.success("¡Es probable que el pasajero sobreviva!")
                
            else:
                
                st.error("Es probable que el pasajero muera...")

pg = st.navigation(
    {'D&A': [
        st.Page(page_1, title= 'Descripción de datos', icon='🚢'),
        st.Page(page_2, title='Análisis de datos',icon='📈'),
    ],
    'AI':[
        st.Page(page_3, title= 'Inteligencia artificial', icon='🤖')

    ]
    }
)
pg.run()