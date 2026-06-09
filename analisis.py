import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

titulos = ["Listado de Retos Técnicos",
           "Reto 1: Distribución de Edad",
           "Reto 2: Distribución por Sexo",
           "Reto 3: Tiempo por Género",
           "Reto 4: Top 10 Países",
           "Reto 5: Correlación de Tiempos Parciales",
           "Reto 6: Edad vs Velocidad",
           "Reto 7: Tiempo de Carrera",
           "Reto 8: Corredores Élite vs Amateur",
           "Reto 9: Consistencia de Ritmo por Género",
           "Reto 10: Mediana de Tiempo por Edad",
           "Reto 11: Relación Tiempos 5K y 40K",
           "Reto 12: Origen de Corredores",
           "Reto 13: Análisis de Fatiga Acumulada",
           "Reto 14: Rangos de Velocidad",
           "Reto 15: Categorías Competitivas",
           "Reto 16: Densidad 10K vs Tiempo Final",
           "Reto 17: Rendimiento Corredores Mayores",
           "Reto 18: Velocidad por País",
           "Reto 19: Identificación de Outliers",
           "Reto 20: Ritmo por Kilómetro (Pace)",
           "Hallazgo 1: Desaceleración por Grupo de Edad",
           "Hallazgo 2: Ritmo en función de la Procedencia",
           "Hallazgo 3: Aguante Top 100 Corredores",
           "Hallazgo 4: Caída de Rendimiento en función de edad y género",
           "Hallazgo 5: Tiempos de llegada de Corredores",
           "Listado de Retos"]

@st.cache_data
def cargar_datos(csv):
    df = pd.read_csv(csv)
    df['Gender_num'] = df['M/F'].map({'M': 0, 'F': 1})
    col_tiempo = ["5K", "10K", "15K", "20K", "Half", "25K", "30K", "35K", "40K", "Official Time"]
    col_modelo = ["Age", "Gender_num", "5Kmin", "10Kmin", "Halfmin", "Official Timemin"]
    
    for col in col_tiempo:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["-", "0", "0:00:00"], pd.NA)
        duracion = pd.to_timedelta(df[col], errors="coerce")
        df[col + "min"] = duracion.dt.total_seconds() / 60
    df = df.dropna(subset=col_modelo)
    return df

@st.cache_data
def crear_columnas(df):
    df["Speed_kmh"] = 42.195 / (df["Official Timemin"] / 60)

    df["Ritmo_1mitad"] = 21.0975 / (df["Halfmin"] / 60)
    df["Ritmo_2mitad"] = 21.0975 / ((df["Official Timemin"] - df["Halfmin"]) / 60)
    df["Dif_ritmo"] = df["Ritmo_1mitad"] - df["Ritmo_2mitad"]


    df["Group"] = "Amateur"
    df.loc[df["Official Timemin"]< 150, "Group"] = "Elite"

    df["Grupo_Edad"] = "70+"
    df.loc[df["Age"]< 70, "Grupo_Edad"] = "60-69"
    df.loc[df["Age"]< 60, "Grupo_Edad"] = "50-59"
    df.loc[df["Age"]< 50, "Grupo_Edad"] = "40-49"
    df.loc[df["Age"]< 40, "Grupo_Edad"] = "30-39"
    df.loc[df["Age"]< 30, "Grupo_Edad"] = "18-29"

    df["Pace_Group"] = pd.cut(df["Speed_kmh"],
                              bins=[0, 8, 9, 10, 11, 12, 13, 14, 15, float("inf")],
                              labels=["8-", "8-9", "9-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15+"])

    df["Categoria_Oficial"] = df["M/F"] + " " + df["Grupo_Edad"]

    desgaste = df["Official Timemin"] / df["5Kmin"]
    df["Tiempo_Esperado"] = df["5Kmin"] * desgaste.median()
    df["Desviacion"] = df["Official Timemin"] - df["Tiempo_Esperado"]

    df["Pace_min_km"] = df["Official Timemin"] / 42.195

    df["Desaceleración"] = df["Ritmo_2mitad"]/df["Ritmo_1mitad"]

    df["Procedencia"] = "Internacional"
    df.loc[df["Country"]=="USA", "Procedencia"] = "Estados Unidos"
    df.loc[df["State"]=="MA", "Procedencia"] = "Massachusetts"

    df["0-25K"] = df['25Kmin'] / 25
    df['25-30K'] = (df['30Kmin'] - df['25Kmin']) / 5
    df['30-35K'] = (df['35Kmin'] - df['30Kmin']) / 5
    df['35K-Final'] = (df['Official Timemin'] - df['35Kmin']) / 7.195

    df['variabilidad'] = df[['5Kmin', '10Kmin', '20Kmin', '25Kmin', '30Kmin', '35Kmin', '40Kmin']].std(axis=1)

    return df

def reto1(df):
    joven = df["Age"].min()
    mayor = df["Age"].max()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Edad")
    ax.set_ylabel("Número de corredores")
    ax.axvline(joven, color="blue", linestyle="--", label=f"Mín: {joven}")
    ax.axvline(mayor, color="red", linestyle="--", label=f"Máx: {mayor}")
    ax.legend()

    texto = ("ANÁLISIS TÉCNICO: "
             "Observamos un histograma de edades dividido en 20 bins, con una curva de densidad. "
             "La mayor parte de los corredores tienen entre 30 y 55 años. "
             f"El participante más joven tiene {joven} años y el mayor {mayor}.")
    
    return fig, texto

def reto2(df): 
    count_genero = df["M/F"].value_counts()
    corredores = count_genero.sum()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(count_genero, labels = ["Hombres", "Mujeres"], colors = ["blue", "pink"], autopct = '%1.1f%%')

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico de sectores muestra la distribución por sexo de los corredores. "
             f"El {(count_genero['M'] / corredores) * 100:.1f}% son hombres y el {(count_genero['F'] / corredores) * 100:.1f}% son mujeres.")
    
    return fig, texto

def reto3(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=df, x="M/F", y="Official Timemin")
    ax.set_xlabel("Sexo")
    ax.set_ylabel("Tiempo oficial (minutos)")
    ax.set_xticks([0, 1], ["Hombres", "Mujeres"])

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico de caja muestra la distribución de los tiempos oficiales por sexo. "
             "Los hombres tienen una mediana de tiempo más baja que las mujeres, lo que indica que en promedio corren más rápido. "
             "Sin embargo, se puede ver que el corredor más lento es un hombre, lo que sugiere mayor variabilidad de lo esperado.")
    
    return fig, texto

def reto4(df):
    top10_paises = df["Country"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=top10_paises.index, y=top10_paises.values)
    ax.set_xlabel("País")
    ax.set_ylabel("Número de Corredores")

    texto = ("ANÁLISIS TÉCNICO: "
             "En el gráfico se puede observar a la perfección una gran dominanza por parte de USA, con la mayor parte de corredores. "
             "De forma seguida está Canada que sigue representando una parte significativa, aunque a partir de ella el resto de países tienen muy poco peso.")
    
    return fig, texto

def reto5(df):
    col_interes = ["5Kmin", "10Kmin", "20Kmin", "Official Timemin"]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(df[col_interes].corr(), cmap = "Reds", annot=True, fmt=".2f")
    
    texto = ("ANÁLISIS TÉCNICO: "
             "Se observa una correlación muy alta y efectiva entre resultados. "
             "Pese a ello, podemos observar como cuanto más cercanas son las distancias más precisa es la correlación. "
             "Por lo tanto, toda aproximación es buena, pero aumenta la exactitud al acercarnos a la distancia objetivo.")
    
    return fig, texto

def reto6(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(data = df, x = "Age", y = "Speed_kmh")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Velocidad Media (km/h)")

    texto = ("ANÁLISIS TÉCNICO: "
            "Podemos observar como al principio crece mucho la velocidad media, hasta los 30 años, donde empieza a decrecer. "
            "También se observa al final movimientos bruscos en la gráfica, al haber pocos corredores de más de 70 años. ")
    
    return fig, texto

def reto7(df):
    media1 = df["Ritmo_1mitad"].mean()
    media2 = df["Ritmo_2mitad"].mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(df["Dif_ritmo"], bins=40, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--", label = "Sin diferencia")
    ax.axvline(df["Dif_ritmo"].mean(), color="orange", linestyle="--", label = "Diferencia media")
    ax.set_xlim(-2.5, 7.5)
    ax.set_xlabel("Diferencia de ritmo (1ª mitad − 2ª mitad) en km/h")
    ax.set_ylabel("Número de corredores")
    ax.legend()
    
    texto = ("ANÁLISIS TÉCNICO: "
             f"La diferencia media de ritmo entre la primera y segunda mitad es de {media1 - media2:.2f} km/h. "
             "La línea roja marca el equilibrio perfecto (sin diferencia). Aunque, la mayoría de corredores caen a la derecha, "
             "lo que indica que casi todos pierden velocidad en la segunda mitad. "
             "La distribución es asimétrica: hay una cola hacia la derecha con corredores que sufren una caída muy brusca de ritmo, "
             "debido a la fatiga, mientras que muy pocos logran acelerar en la segunda mitad (valores negativos).")

    return fig, texto

def reto8(df):
    conteo = df.groupby("Group")["Official Timemin"].count()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(data = df, x = "Official Timemin", hue = "Group", cut = 0, bw_adjust=0.2)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Densidad")
    
    texto = ("ANÁLISIS TÉCNICO: "
             "Podemos observar como casi todos los corredores se concentran entre los 150 min y los 350 min. "
             "Esto nos demuestra que son una categoría totalmente distinta los elite frente a los amateur. "
             f"En total hay {conteo["Amateur"]} corredores amateur frente a {conteo["Elite"]} de elite.")

    return fig, texto

def reto9(df):
    metricas = df.groupby("M/F")["Speed_kmh"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(data = df, x = "M/F", y = "Speed_kmh")
    ax.set_xlabel("Género: M (Masculino), F (Femenino)")
    ax.set_ylabel("Velocidad (km/h)")

    texto = ("ANÁLISIS TÉCNICO: "
             "Podemos observar como tanto la velocidad de los hombres es más alta en la mayoría de cuartiles como en la mediana. "
             f"La media de los hombres ({metricas.loc['M', 'mean']:.2f}km/h) es mayor en {metricas.loc["M","mean"] - metricas.loc["F", "mean"]:.2f}km/h a la de las mujeres ({metricas.loc["F", "mean"]:.2f}km/h). "
             "Mientras que en la consistencia, observamos como las mujeres se mantienen más constantes frente a la media. "
             f"La desviación típica lo explica a la perfección: en las mujeres es: {metricas.loc['F', 'std']:.2f}, frente a la de los hombres: {metricas.loc['M', 'std']:.2f}.")

    return fig, texto

def reto10(df):
    estad_edad = df.groupby("Grupo_Edad")["Official Timemin"].median()
    mejor_grupo = estad_edad.idxmin()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(estad_edad.index, estad_edad.values, marker="o")
    ax.set_xlabel("Grupo de Edad")
    ax.set_ylabel("Tiempo final (min)")

    texto = ("ANÁLISIS TÉCNICO: "
             f"Podemos observar que el mejor grupo es {mejor_grupo}, con un tiempo mediano de {estad_edad.loc[mejor_grupo]:.0f} minutos. "
             "Tanto su grupo de la izquierda (18-29) como el de la derecha (40-49) se encuentran ligeramente por encima, "
             "pero cuanto más te alejas más aumenta la mediana y el resto de cuartiles del tiempo para completar la carrera.")

    return fig, texto

def reto11(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="5Kmin", y="40Kmin")
    plt.plot([df["5Kmin"].min(), df["5Kmin"].max()], [df["5Kmin"].min()*8, df["5Kmin"].max()*8], color="red")
    ax.set_xlabel("Tiempo en 5K (min)")
    ax.set_ylabel("Tiempo en 40K (min)")

    texto = ("ANÁLISIS TÉCNICO: "
             "La nube de puntos se expande a medida que aumenta el tiempo. "
             "La mayoría de los corredores caen por encima de la línea roja, lo que indica que la mayoría pierde eficiencia y tarda más "
             "de 8 veces su tiempo en los primeros 5k, que al llegar al km 40. "
             "También observamos múltiples outliers con tiempo muy malos en una modalidad y muy buenos en otra.")

    return fig, texto

def reto12(df):
    estad_estado = df.groupby("State")["Speed_kmh"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=estad_estado.values, y=estad_estado.index)
    ax.set_xlabel("Velocidad Media (km/h)")
    ax.set_ylabel("Estado")
    ax.axvline(df["Speed_kmh"].mean(), color="red")
    ax.tick_params(axis="y", labelsize = 5)

    texto = ("ANÁLISIS TÉCNICO: "
             f"En este gráfico incluimos todos los estados y provincias presentes. "
             f"El estado con la media más alta es {estad_estado.idxmax()} y el de la más baja es {estad_estado.idxmin()}. "
             "La línea roja indica la media de toda la maratón; los estados a la derecha están por encima "
             "del rendimiento promedio y los de la izquierda por debajo.")

    return fig, texto

def reto13(df):
    tramos = {"0-5K": (df["5Kmin"], 5),
              "5-10K":  (df["10Kmin"] - df["5Kmin"], 5.0),
              "10-15K": (df["15Kmin"] - df["10Kmin"], 5.0),
              "15-20K": (df["20Kmin"] - df["15Kmin"], 5.0),
              "20-25K": (df["25Kmin"] - df["20Kmin"], 5.0),
              "25-30K": (df["30Kmin"] - df["25Kmin"], 5.0),
              "30-35K": (df["35Kmin"] - df["30Kmin"], 5.0),
              "35-40K": (df["40Kmin"] - df["35Kmin"], 5.0),
              "40K-F": (df["Official Timemin"] - df["40Kmin"], 2.195),}
    labels = list(tramos.keys())
    avg_speeds = [dist / (tiempo / 60).mean() for tiempo, dist in tramos.values()]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(labels, avg_speeds, marker="o")
    ax.set_xlabel("Tramo")
    ax.set_ylabel("Velocidad media del tramo (km/h)")
    ax.tick_params(axis="x", rotation=45)

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico muestra la velocidad media entre checkpoints consecutivos, "
             "lo que refleja la fatiga acumulada de forma más precisa si lo hiciesemos con la acumulada desde el principio. "
             "El descenso se acelera a partir del tramo 25-30K, confirmando que el km 25 es el punto crítico donde "
             "la mayoría de corredores empiezan a sufrir el 'muro'. Aunque a partir del km 30 se puede mostrar una leve recuperación puntual.")

    return fig, texto

def reto14(df):
    pace_orden = ["8-", "8-9", "9-10", "10-11", "11-12", "12-13", "13-14", "14-15", "15+"]
    conteo_pace = df["Pace_Group"].value_counts().reindex(pace_orden)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=conteo_pace.index, y=conteo_pace.values)
    ax.set_xlabel("Rango de Velocidad (km/h)")
    ax.set_ylabel("Número de Corredores")

    texto = ("ANÁLISIS TÉCNICO: "
             "En este gráfico se muestra la distribución de corredores según su velocidad media en km/h. "
             "La mayoría de los corredores se encuentran en el rango de 8 a 10 km/h, lo que es típico para maratones. "
             "A medida que aumentamos el rango de velocidad, el número de corredores disminuye significativamente.")

    return fig, texto

def reto15(df):
    top_divisiones = df["Categoria_Oficial"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=top_divisiones.values, y=top_divisiones.index, ax=ax)
    ax.set_xlabel("Número de Corredores")
    ax.set_ylabel("División")

    texto = ("ANÁLISIS TÉCNICO: "
             "Dado que la columna original 'Division' contiene errores de formato, hemos generado "
             f"las categorías competitivas. La categoría con mayor participación es {top_divisiones.idxmax()}. "
             "Se observa que los rangos de edad entre 40 y 59 años (Master) concentran el mayor "
             "volumen de corredores en este maratón.")

    return fig, texto

def reto16(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.hexbin(df["10Kmin"], df["Official Timemin"], cmap="Blues")
    ax.set_xlabel("Tiempo en 10K (min)")
    ax.set_ylabel("Tiempo final (min)")

    texto = ("ANÁLISIS TÉCNICO: "
             f"El gráfico muestra una relación lineal, con un coeficiente de correlación de Pearson de r = {df["10Kmin"].corr(df["Official Timemin"]):.2f}. "
             "El área más oscura es donde mayor cantidad de corredores hay, de forma que se ve que la mayoría "
             "de participantes que cruzan los 10km en 50 min acaban la maratón en unos 220 minutos. "
             "La dispersión es mayor en los ritmos más lentos, al haber mayor variación de tiempos posibles.")

    return fig, texto

def reto17(df):
    df_senior = df[df["Age"] >= 60].copy()
 
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(data=df_senior, x="Official Timemin", hue = "M/F", fill=True)
    ax.set_xlabel("Tiempo Oficial (min)")
    ax.set_ylabel("Densidad")

    texto = ("ANÁLISIS TÉCNICO: "
             "Podemos observar que tanto hombres como mujeres mayores de 60 años tienen una distribución de tiempos más amplia " 
             "y desplazada hacia tiempos más altos en comparación con la población general. Esto refleja el impacto del envejecimiento "
             "en el rendimiento físico, aunque también se observa que algunos corredores mayores de 60 años logran tiempos competitivos.")

    return fig, texto

def reto18(df):
    top5_paises = df["Country"].value_counts().head(5).index
    velocidad_pais = df.groupby("Country")["Speed_kmh"].mean().loc[top5_paises]

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=velocidad_pais.values, y=velocidad_pais.index)
    ax.set_xlabel("Velocidad Media (km/h)")
    ax.set_ylabel("País")

    texto = ("ANÁLISIS TÉCNICO: "
             "En este gráfico se muestra la velocidad media de los corredores de los 5 países con mayor participación. "
             "Podemos observar que el país con la velocidad media más alta no es Estados Unidos, pese a tener el mayor número de "
             "corredores. Por lo que el tamaño del país no tiene porque influir al ritmo de sus corredores.")

    return fig, texto

def reto19(df):
    Q1 = df["Desviacion"].quantile(0.25)
    Q3 = df["Desviacion"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df["Desviacion"] < lower_bound) | (df["Desviacion"] > upper_bound)]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="5Kmin", y="Official Timemin", alpha=0.5)
    sns.scatterplot(data=outliers, x="5Kmin", y="Official Timemin", color="red", label="Outliers")
    ax.set_xlabel("Tiempo en 5K (min)")
    ax.set_ylabel("Tiempo Final (min)")

    texto = ("ANÁLISIS TÉCNICO: "
             "En este gráfico se identifican los outliers en los tiempos finales utilizando un método basado en el rango intercuartílico (IQR). "
             "Los puntos rojos representan a los corredores cuyos tiempos finales son más altos o más bajos que la mayoría de los participantes. "
             "Estos outliers pueden deberse a errores de registro, problemas durante la carrera o rendimientos excepcionalmente buenos o malos. "
             "Aun así, podemos apreciar como la mayoría de outliers se encuentra por encima de la tendencia general, lo que sugiere que se pueda "
             "deber al conocido efecto 'hit the wall', donde los corredores experimentan una caída drástica en su rendimiento debido a la fatiga.")

    return fig, texto

def reto20(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(df["Pace_min_km"], bins=30, kde=True) 
    ax.set_xlabel("Pace (min/km)")
    ax.set_ylabel("Número de Corredores")

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico muestra la distribución del ritmo por kilómetro (pace) de los corredores. "
             "La mayoría de los corredores tienen un pace entre 5 y 7 min/km, lo que es típico para maratones. "
             "También se observa una cola hacia ritmos más lentos, lo que indica que hay corredores que tardan más de 10 min/km, "
             "posiblemente debido a problemas durante la carrera o a un ritmo de caminata.")

    return fig, texto

def hallazgo1(df):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.boxplot(data = df, x = "Grupo_Edad", y = "Desaceleración", hue = "M/F")
    ax.axhline(1, linestyle = "--")
    ax.set_ylim(0.25, 1.5)
    ax.set_xlabel("Grupo de Edad")
    ax.set_ylabel("Factor de Desaceleración (2ª Mitad / 1ª Mitad)")

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico muestra el factor de desaceleración entre las dos mitades de la maratón, segmentado por sexo y grupo de edad. "
             "Este factor se calcula como el cociente entre el ritmo de la segunda mitad y el de la primera: un valor igual a 1 indica "
             "un ritmo perfectamente constante, por encima de 1 significa que el corredor ha perdido velocidad en la segunda mitad. "
             "La conclusión más destacada es que las mujeres se mantienen sistemáticamente más cerca de ese valor 1, lo que indica "
             "una gestión de energía más eficiente y una menor caída de rendimiento relativa. Este patrón se repite en todos los grupos de edad. "
             "Por otro lado, los grupos más jóvenes (18-29) tienden a salir demasiado rápido y pagan las consecuencias en la segunda mitad, "
             "mientras que los corredores de mayor edad (70+) sufren más por limitaciones físicas. "
             "El punto de mayor consistencia se encuentra en los grupos de 40-59 años, que combinan experiencia y capacidad física. "
             "En cualquier caso, ningún grupo logra mantener el factor en 1: todos los grupos, sin excepción, pierden algo de ritmo.")

    return fig, texto

def hallazgo2(df):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.boxplot(data = df, x = "Procedencia", y = "Official Timemin")
    ax.set_xlabel("Procedencia")
    ax.set_ylabel("Tiempo Oficial (minutos)")

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico compara los tiempos oficiales entre corredores de Massachusetts, del resto de EE.UU. e internacionales. "
             "Los locales presentan una mediana más alta, lo que tiene sentido: un corredor internacional que viaja a Boston "
             "suele hacerlo con objetivos claros y mayor preparación, mientras que los locales pueden tomárselo de forma más recreativa. "
             "Sin embargo, hay que ser críticos con este hallazgo: la procedencia no causa el rendimiento. "
             "Si analizásemos la velocidad por kilómetro de cada grupo, las diferencias quedarían explicadas por el ritmo sostenido, "
             "sin que el origen geográfico aporte nada adicional. "
             "Es un patrón interesante, pero la procedencia sería una variable irrelevante en cualquier modelo predictivo.")
    
    return fig, texto

def hallazgo3(df):
    top = 100
    top_inicial_idx = df.sort_values("5Kmin").head(top).index
    checkpoints = ["5Kmin", "10Kmin", "15Kmin", "20Kmin", "25Kmin", "30Kmin", "35Kmin", "40Kmin", "Official Timemin"]
    supervivientes = []
    for p in checkpoints:
        top_actual_idx = df.sort_values(p).head(top).index
        conteo = len(set(top_inicial_idx).intersection(set(top_actual_idx)))
        supervivientes.append(conteo)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(checkpoints, supervivientes)
    ax.set_ylabel("Número de corredores")
    ax.set_xlabel("Checkpoint")

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico muestra cuántos de los 100 corredores más rápidos en los primeros 5K se mantienen dentro del top 100 "
             "a medida que avanza la carrera. La caída más pronunciada ocurre en los primeros checkpoints, lo que sugiere "
             "que una parte de estos corredores podrían ser liebres: atletas que marcan el ritmo en la salida sin intención "
             "de competir por la victoria, y que abandonan el grupo puntero de forma natural en cuanto cumplen su función. "
             "Entre los 20K y 25K el grupo se estabiliza, indicando que los que quedan son corredores reales con capacidad "
             "para mantener ese nivel. Sin embargo, a partir del 25K vuelve a caer, esta vez sí por fatiga: "
             "es el punto clásico del muro, donde las reservas se agotan y los corredores que habían aguantado "
             "el ritmo empiezan a perder posiciones de forma progresiva hasta la meta.")

    return fig, texto

def hallazgo4(df):
    df_plot = df.melt(id_vars=["M/F", "Grupo_Edad"], 
                      value_vars=["0-25K", "25-30K", "30-35K", "35K-Final"],
                      var_name="Tramo", value_name="Velocidad")
    
    grid = sns.FacetGrid(df_plot, col = "Grupo_Edad", row = "M/F")
    grid.map(sns.lineplot, "Tramo", "Velocidad")
    grid.set_axis_labels("Tramo", "Ritmo (min/km)")
    grid.set_titles(col_template="{col_name}", row_template="{row_name}")
    grid.fig.subplots_adjust(top=0.85, hspace=0.3)

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico desglosa la velocidad media por tramos (0-25K, 25-30K, 30-35K y 35K-Final) para cada combinación "
             "de género y grupo de edad, permitiendo ver con precisión dónde y cuánto cae el rendimiento en cada perfil demográfico. "
             "El patrón más llamativo es que la caída de velocidad entre los tramos 25-30K y 30-35K es universal: "
             "se produce en todos los géneros y grupos de edad sin excepción, lo que confirma que el 'muro' no discrimina. "
             "Sin embargo, la magnitud de esa caída y la capacidad de recuperación en el tramo final sí varían significativamente. "
             "Las mujeres muestran una recuperación más notable en el tramo 35K-Final respecto a los hombres, "
             "lo que refuerza la hipótesis de que tienden a reservar más energía en la primera mitad. "
             "En cuanto a la edad, los grupos intermedios (30-49) presentan caídas más contenidas, mientras que los extremos "
             "(18-29 y 70+) acusan más el desgaste, por razones opuestas: exceso de confianza en los jóvenes y limitación física en los mayores.")

    return grid.fig, texto

def hallazgo5(df):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.scatterplot(data=df, x='variabilidad', y='Overall', color = "red", s = 5, ax=ax)
    ax.set_xlabel("Variabilidad de Ritmo")
    ax.set_ylabel("Posición Final")

    texto = ("ANÁLISIS TÉCNICO: "
             "El gráfico enfrenta la variabilidad del ritmo de cada corredor (calculada como la desviación típica de sus tiempos parciales) "
             "con su posición final en la clasificación general. "
             "La tendencia es clara y consistente: a mayor variabilidad de ritmo, peor posición final. "
             "Los corredores que terminan en los primeros puestos presentan tiempos parciales muy homogéneos, "
             "lo que indica que corren a un ritmo constante y controlado desde el inicio. "
             "Por el contrario, los corredores con alta variabilidad suelen ser aquellos que salen demasiado rápido, "
             "acumulan fatiga prematura y sufren una caída brusca de velocidad en los kilómetros finales. "
             "También existe un grupo de alta variabilidad con buena posición final, probablemente corredores tácticos "
             "que aceleran estratégicamente en ciertos tramos sin perder el control general del ritmo. "
             "En definitiva, este hallazgo confirma que la gestión del esfuerzo es tan determinante como la capacidad física.")
    
    return fig, texto

def importancia_variables(metricas):
    coef = pd.Series(metricas["coef"], index=metricas["features"])
    std  = pd.Series(metricas["std_features"], index=metricas["features"])
    importancia = (coef * std).abs().sort_values()

    fig, ax = plt.subplots(figsize=(18, 6))
    importancia.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importancia (minutos)", size = 15)
    ax.tick_params(axis = "both", labelsize = 15)
    plt.tight_layout()
    return fig

def pred_real(metricas):
    fig, ax = plt.subplots(figsize=(8, 8)) 
    ax.scatter(metricas["y_test"], metricas["y_pred"], s=5)
    ax.plot([0,500], [0,500], color = "red", linewidth = 0.5)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_xlabel("Tiempo real (minutos)")
    ax.set_ylabel("Tiempo predicho (minutos)")
    plt.tight_layout()
    return fig