import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Анализ данных о сне и образе жизни", layout="wide",initial_sidebar_state="auto",)

start_img = Image.open('src/im.png')
df = pd.read_csv('dataset/Sleep_health_and_lifestyle_dataset.csv')

st.title("Анализ данных о сне и образе жизни")
st.divider()

st.header("Sleep health and lifestyle dataset")
st.image(start_img, width=700);

st.markdown("Dataset с `Kaggle`, содержащий информацию о *состоянии сна* людей.")
df

st.markdown("""
Person ID — идентификатор человека

Gender — пол

Age — возраст

Occupation — профессия

Sleep Duration — длительность сна (часы)

Quality of Sleep — качество сна 

Physical Activity Level — уровень физической активности

Stress Level — уровень стресса

BMI Category — категория ИМТ (по BMI)

Blood Pressure — артериальное давление 

Heart Rate — частота сердечных сокращений (удары в мин)

Daily Steps — количество шагов в день

Sleep Disorder — наличие или отсутствие расстройства сна
""")

# визуализация представленных в датасете данных, используя контролы стримлита
st.divider()
st.header("Некоторая визуализация представленного датасета")
st.markdown("> Выберите интересующую вкладку")

# сайдбар
st.sidebar.header("Фильтры данных для визуализации")
age_range = st.sidebar.slider("Возраст:",
                            min_value=int(df['Age'].min()),
                            max_value=int(df['Age'].max()),
                            value=(int(df['Age'].min()), 60))

selected_gender = st.sidebar.multiselect("Пол:",
                                    options=df['Gender'].unique(),
                                     default=df['Gender'].unique()
                                     )

quality_range = st.sidebar.slider("Качество сна:",
    min_value=int(df['Quality of Sleep'].min()),
    max_value=int(df['Quality of Sleep'].max()),
    value=(int(df['Quality of Sleep'].min()), 9)
)

upd_df = df[
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Gender'].isin(selected_gender)) &
    (df['Quality of Sleep'].between(quality_range[0], quality_range[1]))

]

def gender_is_not_selected_message(msg : str):
    if not selected_gender:
            if msg:
                st.warning(f"{msg}, выберите интересующий вас пол или оба сразу, используя фильтр \"Пол\" на сайдбаре") # вместо st.write()
            else:
                st.warning("Выберите интересующий вас пол или оба сразу, используя фильтр \"Пол\" на сайдбаре")


# вкладки для визуализации отфильтрованных данных
#metric_tab, , hyperbola_tab"📈 Показатели", "Зависимости"
distribution_tab, = st.tabs(["📊 Распределения"])


#with metric_tab:
#    gender_is_not_selected_message("Чтобы отобразить диаграмму и график")

#    with st.container():
#       stats_by_gender = upd_df.groupby('Gender')[['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']].mean().clip(upper=200)
#       st.subheader("Таблица средних значений")
#        st.dataframe(stats_by_gender.style.format("{:.2f}"))
#        st.bar_chart(stats_by_gender)
        
#        st.write("Средние значения по полу: продолжительность сна, качество сна, уровень физической активности, уровень стресса, пульс, шаги в день.")

with distribution_tab:
    st.subheader("Распределение анализируемых показателей")
    st.write("Построение гистограммы распределения вместе с кривой плотности для выбранного показателя с разделением по полу.")
    
    if selected_gender:
        selected_column = st.selectbox(
            "Выберите показатель:",
            ['Daily Steps', 'Sleep Duration', 'Physical Activity Level', 'Heart Rate']
        )
        figure, axes = plt.subplots()
        sb.histplot(data=upd_df, x=selected_column, kde=True, hue='Gender', multiple='stack')
        st.pyplot(figure)
    else:
        gender_is_not_selected_message(None)

#with hyperbola_tab:
#    st.subheader("Гиперболическая зависимость между двумя признаками")


#    numeric_columns = upd_df.select_dtypes(include=['int64', 'float64']).columns.tolist()


#    x_var = st.selectbox("Выберите независимую переменную (X):", numeric_columns, index=numeric_columns.index("Sleep Duration") if "Sleep Duration" in numeric_columns else 0)
#    y_var = st.selectbox("Выберите зависимую переменную (Y):", numeric_columns, index=numeric_columns.index("Age") if "Age" in numeric_columns else 1)


#    xy_df = upd_df[[x_var, y_var]].dropna()


#    X = (1 / xy_df[x_var]).values.reshape(-1, 1)
#    y = xy_df[y_var].values

#    model = LinearRegression()
#    model.fit(X, y)


#    x_range = np.linspace(1 / xy_df[x_var].max(), 1 / xy_df[x_var].min(), 100).reshape(-1, 1)
#    y_pred = model.predict(x_range)


#    fig, ax = plt.subplots()
#    sb.scatterplot(data=xy_df, x=x_var, y=y_var, ax=ax, label="Данные")
#    ax.plot(1 / x_range, y_pred, color='red', label='Гиперболическая модель')

#    ax.set_xlabel(x_var)
#    ax.set_ylabel(y_var)
#    ax.legend()
#    st.pyplot(fig)

#    st.write(f"Гиперболическая модель вида **{y_var} = a / {x_var} + b** визуализирована красной линией.")
