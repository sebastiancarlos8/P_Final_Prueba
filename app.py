import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Insurance Company - EDA",
    page_icon="👤",
    layout="wide"
)

# ============================================================
# CLASE POO: DATA ANALYZER
# ============================================================
class DataAnalyzer:
    """Clase encargada de centralizar operaciones básicas de análisis."""

    def __init__(self, dataframe):
        self.df = dataframe

    def informacion_general(self):
        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def tipos_datos(self):
        return self.df.dtypes.astype(str).to_frame("Tipo de dato")

    def valores_nulos(self):
        nulos = self.df.isnull().sum()
        porcentaje = (nulos / len(self.df) * 100).round(2)

        resultado = pd.DataFrame({
            "Valores nulos": nulos,
            "Porcentaje (%)": porcentaje
        })

        return resultado.sort_values("Valores nulos", ascending=False)

    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(include="object").columns.tolist()
        return numericas, categoricas

    def estadisticas_descriptivas(self):
        return self.df.describe().T

    def estadistica_columna(self, columna):
        serie = self.df[columna].dropna()

        return {
            "Media": np.mean(serie),
            "Mediana": np.median(serie),
            "Moda": serie.mode().iloc[0] if not serie.mode().empty else np.nan,
            "Desviación estándar": np.std(serie),
            "Mínimo": np.min(serie),
            "Máximo": np.max(serie)
        }

    def conteo_categorico(self, columna):
        conteo = self.df[columna].value_counts(dropna=False)
        porcentaje = (conteo / len(self.df) * 100).round(2)

        return pd.DataFrame({
            "Conteo": conteo,
            "Proporción (%)": porcentaje
        })

    def tasa_renovacion(self, columna):
        tabla = pd.crosstab(
            self.df[columna],
            self.df["renewal"],
            normalize="index"
        ) * 100

        tabla = tabla.rename(columns={0: "No", 1: "Sí"})
        return tabla.round(2)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def cargar_dataset(archivo):
    """Carga y valida el archivo CSV."""
    if archivo is None:
        return None

    if not archivo.name.lower().endswith(".csv"):
        st.error("El archivo seleccionado debe estar en formato .csv.")
        return None

    try:
        datos = pd.read_csv(archivo)
        return datos
    except Exception as error:
        st.error(f"No fue posible leer el archivo: {error}")
        return None


def mostrar_metrica_resumen(df):
    total_clientes = len(df)
    total_variables = len(df.columns)
    renovaciones = df["renewal"].sum()
    tasa_renovacion = renovaciones / total_clientes * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Registros", f"{total_clientes:,}")

    with col2:
        st.metric("📊 Variables", f"{total_variables}")

    with col3:
        st.metric("🔄 Renovaciones", f"{int(renovaciones):,}")

    with col4:
        st.metric("📈 Tasa de renovación", f"{tasa_renovacion:.2f}%")


def grafico_histograma(df, columna):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df,
        x=columna,
        kde=True,
        ax=ax
    )
    ax.set_title(f"Distribución de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def grafico_categorico(df, columna):
    conteo = df[columna].value_counts().reset_index()
    conteo.columns = [columna, "Conteo"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=conteo,
        x=columna,
        y="Conteo",
        ax=ax
    )
    ax.set_title(f"Distribución de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def interpretar_distribucion(df, columna):
    serie = df[columna].dropna()

    media = np.mean(serie)
    mediana = np.median(serie)
    desviacion = np.std(serie)

    if media > mediana * 1.10:
        forma = "presenta una posible asimetría positiva, debido a que la media supera a la mediana."
    elif media < mediana * 0.90:
        forma = "presenta una posible asimetría negativa, debido a que la media es inferior a la mediana."
    else:
        forma = "presenta una distribución relativamente cercana entre media y mediana."

    st.info(
        f"**Interpretación:** La variable **{columna}** tiene una media de "
        f"{media:,.2f}, una mediana de {mediana:,.2f} y una desviación estándar "
        f"de {desviacion:,.2f}. En términos descriptivos, {forma}"
    )


def grafico_numerica_renovacion(df, columna):
    datos = df[[columna, "renewal"]].dropna().copy()
    datos["renewal_label"] = datos["renewal"].map({0: "No", 1: "Sí"})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=datos,
        x="renewal_label",
        y=columna,
        ax=ax
    )
    ax.set_title(f"{columna} según renovación")
    ax.set_xlabel("Renovación")
    ax.set_ylabel(columna)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    resumen = datos.groupby("renewal_label")[columna].agg(
        ["mean", "median"]
    ).round(2)

    resumen.columns = ["Media", "Mediana"]
    st.dataframe(resumen, use_container_width=True)

    if len(resumen) == 2:
        diferencia = resumen.loc["Sí", "Media"] - resumen.loc["No", "Media"]

        if diferencia > 0:
            st.info(
                f"Los clientes que renovaron presentan una media de **{columna}** "
                f"superior en {diferencia:,.2f} unidades respecto a quienes no renovaron."
            )
        else:
            st.info(
                f"Los clientes que renovaron presentan una media de **{columna}** "
                f"inferior en {abs(diferencia):,.2f} unidades respecto a quienes no renovaron."
            )


def grafico_categorica_renovacion(df, columna):
    tabla = pd.crosstab(
        df[columna],
        df["renewal"],
        normalize="index"
    ) * 100

    tabla = tabla.rename(columns={0: "No", 1: "Sí"})

    fig, ax = plt.subplots(figsize=(10, 5))
    tabla.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(f"Renovación según {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Proporción (%)")
    ax.legend(title="Renovación")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(tabla.round(2), use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📚 Contenido")

modulos = st.sidebar.selectbox(
    "Seleccione un módulo",
    ["Home", "Caso de Estudio N°3"]
)

# ============================================================
# HOME
# ============================================================
if modulos == "Home":

    st.title("Trabajo Final - Módulo Python Fundamentals")

    st.subheader("Descripción del objetivo del análisis")

    st.markdown(
        """
        El presente proyecto desarrolla una aplicación interactiva para realizar
        un **Análisis Exploratorio de Datos (EDA)** sobre información histórica
        de clientes de una compañía de seguros.

        El análisis busca identificar patrones descriptivos relacionados con la
        **renovación de pólizas**, utilizando estadística descriptiva,
        clasificación de variables y visualizaciones interactivas.

        El proyecto tiene un enfoque de análisis y toma de decisiones, sin
        desarrollar modelos predictivos.
        """
    )

    st.subheader("Elaborado por")

    st.write("**Nombre completo:** Estudiante")
    st.write("**Módulo:** 🐍 Especialización en Python for Analytics")
    st.write("**Año:** 2026")

    st.subheader("Información general del Dataset")

    st.markdown(
        """
        El dataset **InsuranceCompany.csv** contiene información histórica
        relacionada con clientes de una compañía de seguros. Incluye variables
        demográficas, económicas, historial de pagos, comportamiento de
        morosidad, canal de captación, tipo de residencia, valor de la prima
        y puntaje de evaluación del cliente.

        La variable **renewal** representa si el cliente renovó o no su póliza.
        """
    )

    st.subheader("🛠️ Tecnologías utilizadas")

    st.markdown(
        """
        Para el presente proyecto se utilizaron las siguientes tecnologías:

        - 🐍 Python
        - 🎨 Streamlit
        - 🐼 Pandas
        - 🔢 NumPy
        - 📊 Matplotlib
        - 📈 Seaborn
        - 🧩 Programación Orientada a Objetos (POO)
        - 🔗 GitHub
        """
    )

# ============================================================
# CASO DE ESTUDIO N°3
# ============================================================
else:

    st.title("👤 Caso de Estudio N°3")
    st.markdown(
        """
        ### Análisis Exploratorio de Datos - Insurance Company

        Utilice el panel lateral para cargar el archivo **InsuranceCompany.csv**.
        El análisis se ejecutará únicamente después de validar la carga del
        dataset.
        """
    )

    archivo = st.sidebar.file_uploader(
        "📂 Seleccione su archivo CSV",
        type=["csv"]
    )

    datos = cargar_dataset(archivo)

    if datos is not None:

        st.success(
            f"Archivo **{archivo.name}** cargado correctamente."
        )

        analyzer = DataAnalyzer(datos)

        # ----------------------------------------------------
        # VISTA PREVIA Y DIMENSIONES
        # ----------------------------------------------------
        st.subheader("📂 Carga y validación del dataset")

        mostrar_metrica_resumen(datos)

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Vista previa")
            st.dataframe(
                datos.head(),
                use_container_width=True
            )

        with col2:
            st.write("### Dimensiones")
            st.write(
                f"El dataset contiene **{datos.shape[0]:,} filas** "
                f"y **{datos.shape[1]} columnas**."
            )

            st.write("### Variables")
            st.write(", ".join(datos.columns.tolist()))

        st.divider()

        # ----------------------------------------------------
        # TABS DEL EDA
        # ----------------------------------------------------
        tabs = st.tabs([
            "1️⃣ Información",
            "2️⃣ Estadística",
            "3️⃣ Numéricas",
            "4️⃣ Categóricas",
            "5️⃣ Bivariado",
            "6️⃣ Dinámico",
            "7️⃣ Hallazgos"
        ])

        # ====================================================
        # ÍTEM 1
        # ====================================================
        with tabs[0]:

            st.header("Ítem 1: Información general del dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Información mediante .info()")
                st.text(analyzer.informacion_general())

            with col2:
                st.subheader("Tipos de datos")
                st.dataframe(
                    analyzer.tipos_datos(),
                    use_container_width=True
                )

            st.subheader("Valores nulos")
            nulos = analyzer.valores_nulos()

            st.dataframe(
                nulos,
                use_container_width=True
            )

        # ====================================================
        # ÍTEM 2
        # ====================================================
        with tabs[0]:

            st.header("Ítem 2: Clasificación de variables")

            numericas, categoricas = analyzer.clasificar_variables()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔢 Variables numéricas")
                st.metric(
                    "Cantidad",
                    len(numericas)
                )

                for variable in numericas:
                    st.write(f"• {variable}")

            with col2:
                st.subheader("🔤 Variables categóricas")
                st.metric(
                    "Cantidad",
                    len(categoricas)
                )

                for variable in categoricas:
                    st.write(f"• {variable}")

            st.info(
                f"Se identificaron **{len(numericas)} variables numéricas** y "
                f"**{len(categoricas)} variables categóricas** mediante una "
                f"función personalizada integrada en la clase `DataAnalyzer`."
            )

        # ====================================================
        # ÍTEM 3
        # ====================================================
        with tabs[1]:

            st.header("Ítem 3: Estadísticas descriptivas")

            st.write(
                "La tabla resume las principales medidas descriptivas "
                "de las variables numéricas."
            )

            estadisticas = analyzer.estadisticas_descriptivas()
            st.dataframe(
                estadisticas.round(2),
                use_container_width=True
            )

            numericas, _ = analyzer.clasificar_variables()

            variable_estadistica = st.selectbox(
                "Seleccione una variable para profundizar:",
                numericas,
                key="estadistica_variable"
            )

            resultado = analyzer.estadistica_columna(
                variable_estadistica
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Media",
                    f"{resultado['Media']:,.2f}"
                )

            with col2:
                st.metric(
                    "Mediana",
                    f"{resultado['Mediana']:,.2f}"
                )

            with col3:
                st.metric(
                    "Desviación estándar",
                    f"{resultado['Desviación estándar']:,.2f}"
                )

            st.write(
                f"Para **{variable_estadistica}**, la media y la mediana "
                f"permiten observar la tendencia central, mientras que la "
                f"desviación estándar permite evaluar la dispersión de los datos."
            )

        # ====================================================
        # ÍTEM 4
        # ====================================================
        with tabs[1]:

            st.header("Ítem 4: Análisis de valores faltantes")

            nulos = analyzer.valores_nulos()

            solo_nulos = nulos[nulos["Valores nulos"] > 0]

            if solo_nulos.empty:
                st.success("No se identificaron valores faltantes.")
            else:
                st.dataframe(
                    solo_nulos,
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(9, 5))

                sns.barplot(
                    x=solo_nulos.index,
                    y=solo_nulos["Valores nulos"],
                    ax=ax
                )

                ax.set_title("Cantidad de valores faltantes")
                ax.set_xlabel("Variable")
                ax.set_ylabel("Valores nulos")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                mayor_nulo = solo_nulos["Valores nulos"].idxmax()
                cantidad_nulo = solo_nulos.loc[
                    mayor_nulo,
                    "Valores nulos"
                ]

                st.warning(
                    f"La variable con mayor cantidad de valores faltantes es "
                    f"**{mayor_nulo}**, con **{cantidad_nulo:,} registros**. "
                    f"Este aspecto debe considerarse antes de realizar análisis "
                    f"que involucren directamente dicha variable."
                )

        # ====================================================
        # ÍTEM 5
        # ====================================================
        with tabs[2]:

            st.header("Ítem 5: Distribución de variables numéricas")

            numericas, _ = analyzer.clasificar_variables()

            variable_hist = st.selectbox(
                "Seleccione una variable numérica:",
                numericas,
                key="hist_variable"
            )

            grafico_histograma(
                datos,
                variable_hist
            )

            interpretar_distribucion(
                datos,
                variable_hist
            )

        # ====================================================
        # ÍTEM 6
        # ====================================================
        with tabs[3]:

            st.header("Ítem 6: Análisis de variables categóricas")

            _, categoricas = analyzer.clasificar_variables()

            variable_cat = st.selectbox(
                "Seleccione una variable categórica:",
                categoricas,
                key="cat_variable"
            )

            resultado_cat = analyzer.conteo_categorico(
                variable_cat
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Conteos y proporciones")
                st.dataframe(
                    resultado_cat,
                    use_container_width=True
                )

            with col2:
                grafico_categorico(
                    datos,
                    variable_cat
                )

            categoria_mayor = resultado_cat["Conteo"].idxmax()
            cantidad_mayor = resultado_cat.loc[
                categoria_mayor,
                "Conteo"
            ]

            porcentaje_mayor = resultado_cat.loc[
                categoria_mayor,
                "Proporción (%)"
            ]

            st.info(
                f"La categoría con mayor frecuencia en **{variable_cat}** es "
                f"**{categoria_mayor}**, con **{cantidad_mayor:,} registros** "
                f"({porcentaje_mayor:.2f}% del total)."
            )

        # ====================================================
        # ÍTEM 7
        # ====================================================
        with tabs[4]:

            st.header(
                "Ítem 7: Análisis bivariado - Numérica vs categórica"
            )

            st.write(
                "Se compara una variable numérica frente a la variable "
                "categórica objetivo `renewal`."
            )

            numericas, _ = analyzer.clasificar_variables()

            # No utilizar ID como variable analítica principal
            opciones_num = [
                x for x in numericas
                if x != "renewal" and x != "id"
            ]

            variable_bivariada = st.selectbox(
                "Seleccione una variable numérica:",
                opciones_num,
                key="bivariado_num"
            )

            grafico_numerica_renovacion(
                datos,
                variable_bivariada
            )

            st.write(
                f"El gráfico permite comparar la distribución de "
                f"**{variable_bivariada}** entre clientes que renovaron "
                f"y clientes que no renovaron."
            )

        # ====================================================
        # ÍTEM 8
        # ====================================================
        with tabs[4]:

            st.header(
                "Ítem 8: Análisis bivariado - Categórica vs categórica"
            )

            _, categoricas = analyzer.clasificar_variables()

            opciones_cat = [
                x for x in categoricas
                if x != "renewal"
            ]

            variable_cat_bivariada = st.selectbox(
                "Seleccione una variable categórica:",
                opciones_cat,
                key="bivariado_cat"
            )

            grafico_categorica_renovacion(
                datos,
                variable_cat_bivariada
            )

            st.write(
                f"Las proporciones permiten comparar el comportamiento de "
                f"renovación entre las categorías de **{variable_cat_bivariada}**."
            )

        # ====================================================
        # ÍTEM 9
        # ====================================================
        with tabs[5]:

            st.header(
                "Ítem 9: Análisis basado en parámetros seleccionados"
            )

            numericas, categoricas = analyzer.clasificar_variables()

            st.subheader("Selección múltiple de variables")

            variables_seleccionadas = st.multiselect(
                "Seleccione una o más variables numéricas:",
                [
                    x for x in numericas
                    if x != "id" and x != "renewal"
                ],
                default=[
                    "Income",
                    "premium"
                ] if "Income" in numericas and "premium" in numericas else [],
                key="multi_variables"
            )

            if variables_seleccionadas:

                st.write(
                    f"Se seleccionaron **{len(variables_seleccionadas)} "
                    f"variables** para el análisis."
                )

                resumen_seleccionado = datos[
                    variables_seleccionadas
                ].describe().T

                st.dataframe(
                    resumen_seleccionado.round(2),
                    use_container_width=True
                )

            st.divider()

            st.subheader("Filtro dinámico")

            variable_slider = st.selectbox(
                "Seleccione una variable para filtrar:",
                [
                    x for x in numericas
                    if x != "id" and x != "renewal"
                ],
                key="slider_variable"
            )

            minimo = float(datos[variable_slider].min())
            maximo = float(datos[variable_slider].max())

            rango = st.slider(
                f"Seleccione el rango de {variable_slider}:",
                min_value=minimo,
                max_value=maximo,
                value=(minimo, maximo),
                key="rango_slider"
            )

            datos_filtrados = datos[
                datos[variable_slider].between(
                    rango[0],
                    rango[1]
                )
            ]

            st.write(
                f"Registros dentro del rango seleccionado: "
                f"**{len(datos_filtrados):,}**"
            )

            st.dataframe(
                datos_filtrados.head(20),
                use_container_width=True
            )

            mostrar_interpretacion = st.checkbox(
                "Mostrar interpretación del filtro",
                value=True,
                key="mostrar_interpretacion"
            )

            if mostrar_interpretacion:

                porcentaje = (
                    len(datos_filtrados) /
                    len(datos) * 100
                )

                st.info(
                    f"El rango seleccionado concentra "
                    f"**{len(datos_filtrados):,} registros**, equivalentes "
                    f"al **{porcentaje:.2f}%** del dataset."
                )

        # ====================================================
        # ÍTEM 10
        # ====================================================
        with tabs[6]:

            st.header("Ítem 10: Hallazgos clave")

            tasa_general = datos["renewal"].mean() * 100

            income_media_no = datos.loc[
                datos["renewal"] == 0,
                "Income"
            ].mean()

            income_media_si = datos.loc[
                datos["renewal"] == 1,
                "Income"
            ].mean()

            premium_media_no = datos.loc[
                datos["renewal"] == 0,
                "premium"
            ].mean()

            premium_media_si = datos.loc[
                datos["renewal"] == 1,
                "premium"
            ].mean()

            cash_no = datos.loc[
                datos["renewal"] == 0,
                "perc_premium_paid_by_cash_credit"
            ].mean()

            cash_si = datos.loc[
                datos["renewal"] == 1,
                "perc_premium_paid_by_cash_credit"
            ].mean()

            canal_renovacion = (
                datos.groupby("sourcing_channel")["renewal"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
            )

            mejor_canal = canal_renovacion.index[0]
            mejor_tasa_canal = canal_renovacion.iloc[0]

            st.subheader("📌 Resumen general")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Tasa general de renovación",
                    f"{tasa_general:.2f}%"
                )

            with col2:
                st.metric(
                    "Canal con mayor tasa de renovación",
                    f"{mejor_canal} ({mejor_tasa_canal:.2f}%)"
                )

            st.subheader("🔎 Principales insights")

            st.markdown(
                f"""
                **1. Renovación:** La tasa global de renovación del dataset
                es de **{tasa_general:.2f}%**, lo que muestra una mayor
                concentración de clientes en la categoría de renovación.

                **2. Ingreso:** El ingreso mensual promedio es de
                **{income_media_si:,.2f}** entre quienes renovaron y de
                **{income_media_no:,.2f}** entre quienes no renovaron.

                **3. Prima:** El valor promedio de la prima es de
                **{premium_media_si:,.2f}** para clientes que renovaron frente
                a **{premium_media_no:,.2f}** para quienes no renovaron.

                **4. Forma de pago:** El porcentaje promedio de la prima
                pagado mediante efectivo/crédito es de **{cash_si:.2f}%**
                en quienes renovaron frente a **{cash_no:.2f}%** en quienes
                no renovaron.

                **5. Canal de captación:** El canal **{mejor_canal}** presenta
                la mayor tasa descriptiva de renovación, con
                **{mejor_tasa_canal:.2f}%**.
                """
            )

            st.warning(
                "Estos hallazgos corresponden a relaciones descriptivas "
                "observadas en el dataset. No deben interpretarse como "
                "relaciones causales ni como predicciones."
            )

    else:

        st.info(
            "📂 Cargue el archivo **InsuranceCompany.csv** desde el panel "
            "lateral para iniciar el análisis."
        )
