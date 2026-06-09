import streamlit as st
import pickle
import analisis as anl

st.set_page_config(page_title="Maratón de Boston 2017", layout="wide")
st.title("Análisis de Rendimiento - Maratón de Boston 2017")

tab1, tab2, tab3 = st.tabs(["Retos", "Hallazgos", "Estimador de Tiempo"])

csv = anl.cargar_datos("marathon_results_2017.csv")

df = anl.crear_columnas(csv)

with open('modelo_maraton.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('metricas_modelo.pkl', 'rb') as f:
    metricas = pickle.load(f)

with tab3:
    st.header("Simulador de Predicción de Tiempo")
    st.write("El modelo de regresión lineal utiliza 5 variables para predecir el tiempo final: edad, género, y los tiempos parciales en 5K, 10K y media maratón. "
    "Estas variables se han elegido razonadamente. \n"
    "La edad y el género son datos básicos que cualquier corredor conoce y que, como hemos visto en el análisis, "
    "tienen un impacto real en el rendimiento. \n"
    "En cuanto a los tiempos parciales, se han escogido el 5K, el 10K y la media maratón. "
    "Usar tiempos más avanzados como el 40K haría la predicción casi inservible: con dos kilómetros por delante "
    "cualquier aproximación sería precisa, pero completamente inútil en la práctica. "
    "Los tres tiempos elegidos representan un equilibrio: son lo suficientemente tempranos para ser útiles "
    "y lo suficientemente informativos para que el modelo funcione bien. \n"
    "La procedencia geográfica se descartó conscientemente. Como vimos en el Hallazgo 2, aunque los corredores "
    "internacionales tienden a tener mejores tiempos que los locales, esta diferencia no la causa el origen en sí mismo. Una vez que el modelo ya conoce cómo corre una persona a través "
    "de sus tiempos parciales, saber de dónde viene no añade ninguna información adicional. "
    "Incluirla habría añadido ruido al modelo sin mejorar la predicción.\n\n"
    "A continuación se presenta el simulador con unos datos por defecto totalmente personalizables:\n")
    
    col1, col2 = st.columns(2)
    with col1:
        edad = st.number_input("Edad", 18, 100, 35)
        genero = st.selectbox("Sexo", ["Hombre", "Mujer"])
        genero_val = 0 if genero == "Hombre" else 1
    
    with col2:
        st.write("Tiempo en 5K (horas, minutos, segundos)")
        c1, c2, c3 = st.columns(3)
        h5k = c1.number_input("h", 0, 1, 0, key="h5k", label_visibility="collapsed")
        m5k = c2.number_input("min", 0, 59, 20, key="m5k", label_visibility="collapsed")
        s5k = c3.number_input("seg", 0, 59, 0, key="s5k", label_visibility="collapsed")
        t_5k = h5k * 60 + m5k + s5k / 60

        st.write("Tiempo en 10K (horas, minutos, segundos)")
        c1, c2, c3 = st.columns(3)
        h10k = c1.number_input("h", 0, 2, 0, key="h10k", label_visibility="collapsed")
        m10k = c2.number_input("min", 0, 59, 40, key="m10k", label_visibility="collapsed")
        s10k = c3.number_input("seg", 0, 59, 0, key="s10k", label_visibility="collapsed")
        t_10k = h10k * 60 + m10k + s10k / 60

        st.write("Tiempo en Media Maratón (horas, minutos, segundos)")
        c1, c2, c3 = st.columns(3)
        hh = c1.number_input("h", 0, 4, 1, key="hhalf", label_visibility="collapsed")
        mh = c2.number_input("min", 0, 59, 40, key="mhalf", label_visibility="collapsed")
        sh = c3.number_input("seg", 0, 59, 0, key="shalf", label_visibility="collapsed")
        t_half = hh * 60 + mh + sh / 60

    if st.button("Calcular Predicción"):
        prediccion = modelo.predict([[edad, genero_val, t_5k, t_10k, t_half]]).item()
        st.metric("Tiempo Estimado de Meta", f"{int(prediccion // 60)} horas y {int(prediccion % 60)} minutos")

    st.divider()
    st.subheader("Evaluación del Modelo")
    st.write("Para evaluar la precisión del modelo se han utilizado dos métricas calculadas sobre el " 
    "20% de los datos que el modelo no vio durante el entrenamiento (X_test): el Error Medio Absoluto (MAE) " 
    "y la Desviación Estandar de los residuos. El MAE mide cuantos minutos se equivoca de media el modelo " 
    "en sus predicciones, mientras que la variabilidad indica cuanto oscilan esos errores alrededor de la " 
    "media. Es decir, si el modelo falla de forma consistente o muy irregular según el corredor.")

    met1, met2 = st.columns(2)
    met1.metric("Error Medio Absoluto (MAE)", f"{int(metricas["mae"])}min {int((metricas["mae"] % 1) * 60)}seg")
    met2.metric("Variabilidad de la Predicción", f"{int(metricas['variabilidad'])}min {int((metricas["variabilidad"] % 1) * 60)}seg")
    st.info(f"El modelo se equivoca de media {int(metricas["mae"])} minutos {int((metricas["mae"] % 1) * 60)} segundos, lo que en el contexto de una maratón (3-6 horas) es un error razonablemente bajo, lo que equivale a un fallo de unos 20s/km. "
    f"La variabilidad de {int(metricas['variabilidad'])} minutos y {int((metricas["variabilidad"] % 1) * 60)} segundos indica que hay corredores para los que el modelo funciona muy bien y otros donde el error es mayor, probablemente corredores con un ritmo muy irregular o perfiles más atípicos.")

    st.subheader("Importancia de Variables en el Modelo")

    fig_imp = anl.importancia_variables(metricas)
    st.pyplot(fig_imp)

    st.write("El gráfico muestra la importancia de cada variable usando coeficientes estandarizados (coeficiente × desviación típica), "
         "lo que permite comparar variables de escalas muy distintas de forma justa. "
         "Con coeficientes brutos, la Media Maratón parecía poco relevante porque su coeficiente unitario es pequeño: "
         "cada minuto extra en la media solo añade ~1.5 minutos al tiempo final. "
         "Sin embargo, al tener una desviación típica de 15-20 minutos entre corredores, su impacto real es el mayor de todos.\n\n"
         "El Género, al ser una variable binaria, tiene una desviación típica cercana a 0.5, lo que limita su peso real pese a "
         "tener un coeficiente aparentemente relevante. La Edad acumula efecto progresivamente pero sigue siendo secundaria. "
         "En definitiva, los tiempos parciales dominan la predicción, siendo la Media Maratón la variable más influyente al "
         "combinar un coeficiente significativo con la mayor variabilidad entre corredores.")

    st.subheader("Predicción vs Realidad")

    col1, col2 = st.columns(2)
    with col1:
        fig_pred = anl.pred_real(metricas)
        st.pyplot(fig_pred)

    with col2:
        st.write("Como podemos ver esta gráfica corrobora todo lo que hemos visto hasta ahora. " 
        "La línea roja atravesa la nube de puntos, que tiene una dispersión hacia las afueras bastante limitada. " 
        "Aunque podemos observar como en los extremos de la nube el modelo tiende a predecir un tiempo mayor en los extremos, mientras que en en la zona céntrica predice valores ligeramente más bajos a los esperados. " 
        "También se tiene presencia de unos pocos outliers, donde solo uno está muy lejos de la predicción.\n\n" 
        "Como conlusión me gustaría resaltar más que nada la importancia de la limpieza de datos y el análisis crítico de los datos: " 
        "Si no se limpian los datos puedes tener valores atípicos que afecten al modelo ya sea por fallos en el guardado de los datos o por intenciones maliciosas de otras personas, que romperían nuestra predicción. " 
        "También tenemos cuidado a la hora de que datos estamos dispuestos a usar, si hubiesemos tenido en cuenta lo columna de los 40km para la regresión, el resultado sería casi perfecto, " 
        "pero sería una aplicación que solo se podría usar en un momento específico (en el que además solo piensas en llegar a la meta).\n\n" 
        "Con los datos actuales, creo que es un modelo bastante compacto con alguna mejora no muy significativa. " 
        "Pero, se podrían mejorar incluyendo otros datos de los corredores. " 
        "En primer lugar, la frecuencia cardiaca y la relación con su frecuencia máxima nos ayudaría a predecir el desgaste que va a experimentar. " 
        "En segundo lugar, el número de maratones previas, nos serviría para ver si hay diferencia en como gestiona el ritmo en función de esa experiencía. " 
        "Por último, poder comparar las condiciones entre diferentes años para tener en cuenta si la lluvia, humedad o viento tienen impacto. " 
        "Aunque, esta última formaría parte de un proyecto más complejo a mayor escala.")
    
with tab1:
    st.header(anl.titulos[0])

    fila1_col1, fila1_col2, fila1_col3 = st.columns(3)

    with fila1_col1:
        st.subheader(anl.titulos[1])
        fig1, texto1 = anl.reto1(df)
        st.pyplot(fig1)
        st.info(texto1)

    with fila1_col2:
        st.subheader(anl.titulos[2])
        fig2, texto2 = anl.reto2(df)
        st.pyplot(fig2)
        st.info(texto2)

    with fila1_col3:
        st.subheader(anl.titulos[3])
        fig3, texto3 = anl.reto3(df)
        st.pyplot(fig3)
        st.info(texto3)

    fila2_col1, fila2_col2, fila2_col3 = st.columns(3)

    with fila2_col1:
        st.subheader(anl.titulos[4])
        fig4, texto4 = anl.reto4(df)
        st.pyplot(fig4)
        st.info(texto4)

    with fila2_col2:
        st.subheader(anl.titulos[5])
        fig5, texto5 = anl.reto5(df)
        st.pyplot(fig5)
        st.info(texto5)

    with fila2_col3:
        st.subheader(anl.titulos[6])
        fig6, texto6 = anl.reto6(df)
        st.pyplot(fig6)
        st.info(texto6)

    fila3_col1, fila3_col2, fila3_col3 = st.columns(3)

    with fila3_col1:
        st.subheader(anl.titulos[7])
        fig7, texto7 = anl.reto7(df)
        st.pyplot(fig7)
        st.info(texto7)

    with fila3_col2:
        st.subheader(anl.titulos[8])
        fig8, texto8 = anl.reto8(df)
        st.pyplot(fig8)
        st.info(texto8)

    with fila3_col3:
        st.subheader(anl.titulos[9])
        fig9, texto9 = anl.reto9(df)
        st.pyplot(fig9)
        st.info(texto9)

    fila4_col1, fila4_col2, fila4_col3 = st.columns(3)

    with fila4_col1:
        st.subheader(anl.titulos[10])
        fig10, texto10 = anl.reto10(df)
        st.pyplot(fig10)
        st.info(texto10)

    with fila4_col2:
        st.subheader(anl.titulos[11])
        fig11, texto11 = anl.reto11(df)
        st.pyplot(fig11)
        st.info(texto11)

    with fila4_col3:
        st.subheader(anl.titulos[12])
        fig12, texto12 = anl.reto12(df)
        st.pyplot(fig12)
        st.info(texto12)

    fila5_col1, fila5_col2, fila5_col3 = st.columns(3)

    with fila5_col1:
        st.subheader(anl.titulos[13])
        fig13, texto13 = anl.reto13(df)
        st.pyplot(fig13)
        st.info(texto13)

    with fila5_col2:
        st.subheader(anl.titulos[14])
        fig14, texto14 = anl.reto14(df)
        st.pyplot(fig14)
        st.info(texto14)

    with fila5_col3:
        st.subheader(anl.titulos[15])
        fig15, texto15 = anl.reto15(df)
        st.pyplot(fig15)
        st.info(texto15)

    fila6_col1, fila6_col2, fila6_col3 = st.columns(3)

    with fila6_col1:
        st.subheader(anl.titulos[16])
        fig16, texto16 = anl.reto16(df)
        st.pyplot(fig16)
        st.info(texto16)

    with fila6_col2:
        st.subheader(anl.titulos[17])
        fig17, texto17 = anl.reto17(df)
        st.pyplot(fig17)
        st.info(texto17)

    with fila6_col3:
        st.subheader(anl.titulos[18])
        fig18, texto18 = anl.reto18(df)
        st.pyplot(fig18)
        st.info(texto18)

    fila7_col1, fila7_col2 = st.columns(2)

    with fila7_col1:
        st.subheader(anl.titulos[19])
        fig19, texto19 = anl.reto19(df)
        st.pyplot(fig19)
        st.info(texto19)

    with fila7_col2:
        st.subheader(anl.titulos[20])
        fig20, texto20 = anl.reto20(df)
        st.pyplot(fig20)
        st.info(texto20)

with tab2:
    st.header(anl.titulos[26])

    st.subheader(anl.titulos[21])
    fig21, texto21 = anl.hallazgo1(df)
    st.pyplot(fig21)
    st.info(texto21)

    st.subheader(anl.titulos[22])
    fig22, texto22 = anl.hallazgo2(df)
    st.pyplot(fig22)
    st.info(texto22)

    st.subheader(anl.titulos[23])
    fig23, texto23 = anl.hallazgo3(df)
    st.pyplot(fig23)
    st.info(texto23)

    st.subheader(anl.titulos[24])
    fig24, texto24 = anl.hallazgo4(df)
    st.pyplot(fig24)
    st.info(texto24)

    st.subheader(anl.titulos[25])
    fig25, texto25 = anl.hallazgo5(df)
    st.pyplot(fig25)
    st.info(texto25)