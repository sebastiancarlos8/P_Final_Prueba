"""
============================================================================
 CASO DE ESTUDIO N°3 - INSURANCE COMPANY
 Especialización en Python for Analytics
============================================================================
Aplicación interactiva desarrollada con Streamlit para realizar un Análisis
Exploratorio de Datos (EDA) sobre el dataset InsuranceCompany.csv.
 
Objetivo: identificar patrones DESCRIPTIVOS relacionados con la renovación
de pólizas de seguro (variable objetivo: `renewal`). Este proyecto NO
construye modelos predictivos; su enfoque es puramente exploratorio y de
apoyo a la toma de decisiones.
 
Conceptos del curso aplicados en este archivo (para ubicarlos fácilmente
mientras se revisa el código):
    - Variables y tipos de datos      -> a lo largo de todo el script
    - Funciones                        -> sección "FUNCIONES AUXILIARES"
    - f-strings                        -> usados en casi todos los st.write/
                                           st.markdown/st.info para insertar
                                           valores calculados en el texto
    - Programación Orientada a Objetos -> clase `DataAnalyzer`
    - NumPy y Pandas                   -> cálculos estadísticos y manejo
                                           del DataFrame
    - Matplotlib y Seaborn             -> funciones de graficación
    - Estadística descriptiva          -> media, mediana, moda, desviación
                                           estándar, cuartiles/IQR
    - Widgets de Streamlit             -> sidebar, tabs, columns, selectbox,
                                           multiselect, slider, checkbox
============================================================================
"""
 
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
 
# ============================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ============================================================
# st.set_page_config() debe ejecutarse una sola vez y ser el primer comando
# de Streamlit del script. Define el título de la pestaña del navegador,
# el ícono y si el contenido usa todo el ancho de la pantalla ("wide").
st.set_page_config(
    page_title="Insurance Company - EDA",
    page_icon="🛡️",
    layout="wide"
)
 
 
# ============================================================
# CLASE POO: DATA ANALYZER
# ============================================================
# Esta clase centraliza TODA la lógica de análisis de datos (estadística
# descriptiva, clasificación de variables, tablas de nulos, cruces con la
# variable objetivo, etc.). Concentrar esta lógica en una clase es lo que
# el caso de estudio pide como "uso de Programación Orientada a Objetos":
# en vez de tener funciones sueltas que reciben el DataFrame cada vez,
# la clase lo guarda una sola vez (self.df) y expone métodos de análisis
# listos para usar desde cualquier parte de la interfaz.
class DataAnalyzer:
    """Encapsula las operaciones de análisis exploratorio sobre un DataFrame."""
 
    def __init__(self, dataframe):
        # El DataFrame se guarda como atributo de la instancia para que
        # todos los métodos de la clase puedan reutilizarlo sin tener que
        # volver a pasarlo como parámetro en cada llamada.
        self.df = dataframe
 
    def informacion_general(self):
        """Replica la salida de df.info() como texto, para poder mostrarla
        con st.text(). df.info() imprime directamente en consola, por lo
        que se usa un buffer de texto (StringIO) para "capturar" esa salida
        y poder desplegarla dentro de la app."""
        buffer = StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()
 
    def tipos_datos(self):
        """Devuelve una tabla con el tipo de dato (dtype) de cada columna."""
        return self.df.dtypes.astype(str).to_frame("Tipo de dato")
 
    def valores_nulos(self):
        """Cuenta valores nulos por columna y calcula el porcentaje que
        representan sobre el total de filas. Se ordena de mayor a menor
        para identificar rápidamente las columnas más afectadas."""
        nulos = self.df.isnull().sum()
        porcentaje = (nulos / len(self.df) * 100).round(2)
 
        resultado = pd.DataFrame({
            "Valores nulos": nulos,
            "Porcentaje (%)": porcentaje
        })
 
        return resultado.sort_values("Valores nulos", ascending=False)
 
    def clasificar_variables(self):
        """Función personalizada que separa las columnas del dataset en
        dos grupos según su tipo de dato:
            - Numéricas: cualquier subtipo de número (int, float) mediante
              np.number.
            - Categóricas: todo lo que NO sea numérico (texto/objeto).
        Se usa exclude=np.number (en vez de include="object") para que la
        clasificación funcione igual en pandas 2.x y en pandas 3.x, donde
        el nuevo dtype "string" no siempre es reconocido por include=
        "object". Esto evita revisar manualmente cada columna y permite
        alimentar dinámicamente los selectbox/multiselect de la interfaz."""
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numericas, categoricas
 
    def estadisticas_descriptivas(self):
        """Wrapper sobre df.describe(), transpuesto (.T) para que cada
        variable quede como fila y cada estadístico (media, std, min,
        cuartiles, max) quede como columna: más fácil de leer en una tabla
        ancha como las que usa Streamlit."""
        return self.df.describe().T
 
    def estadistica_columna(self, columna):
        """Calcula manualmente, usando NumPy, las principales medidas de
        tendencia central y dispersión para UNA columna numérica puntual.
        Se usa dropna() para no distorsionar los cálculos con valores
        faltantes."""
        serie = self.df[columna].dropna()
 
        return {
            "Media": np.mean(serie),
            "Mediana": np.median(serie),
            "Moda": serie.mode().iloc[0] if not serie.mode().empty else np.nan,
            "Desviación estándar": np.std(serie),
            "Mínimo": np.min(serie),
            "Máximo": np.max(serie)
        }
 
    def detectar_outliers_iqr(self, columna):
        """Detecta valores atípicos (outliers) usando la regla del Rango
        Intercuartílico (IQR): todo valor por debajo de Q1 - 1.5*IQR o por
        encima de Q3 + 1.5*IQR se considera atípico. Es un complemento
        clásico a la estadística descriptiva para explicar por qué, en
        variables como `Income`, la media puede estar muy alejada de la
        mediana."""
        serie = self.df[columna].dropna()
 
        q1 = np.percentile(serie, 25)
        q3 = np.percentile(serie, 75)
        iqr = q3 - q1
 
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
 
        outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]
 
        return {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Límite inferior": limite_inferior,
            "Límite superior": limite_superior,
            "Cantidad de outliers": len(outliers),
            "Porcentaje (%)": round(len(outliers) / len(serie) * 100, 2)
        }
 
    def conteo_categorico(self, columna):
        """Conteo de frecuencias absolutas y relativas (%) de una variable
        categórica. dropna=False incluye explícitamente los nulos como una
        categoría más, si existieran."""
        conteo = self.df[columna].value_counts(dropna=False)
        porcentaje = (conteo / len(self.df) * 100).round(2)
 
        return pd.DataFrame({
            "Conteo": conteo,
            "Proporción (%)": porcentaje
        })
 
    def tasa_renovacion(self, columna):
        """Tabla de contingencia (crosstab) entre una variable categórica y
        la variable objetivo `renewal`, normalizada por fila (normalize=
        "index") para obtener el porcentaje de renovación dentro de cada
        categoría, y no el porcentaje sobre el total del dataset."""
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
# Estas funciones NO pertenecen a la clase porque no operan sobre "el
# análisis" en sí, sino sobre la interfaz (widgets, gráficos, validaciones
# de carga de archivo). Mantenerlas separadas de la clase DataAnalyzer
# respeta el principio de responsabilidad única: la clase analiza datos,
# las funciones dibujan/validan la interfaz.
 
def mostrar_imagen_segura(ruta, **kwargs):
    """Muestra una imagen (logo institucional, logo de Python, etc.) sin
    romper la aplicación si el archivo no existe en el repositorio. Esto es
    útil porque las imágenes (DMC.png, Python_logo.png) deben subirse junto
    a app.py, y si faltan, la app debe seguir funcionando igual."""
    try:
        st.image(ruta, **kwargs)
    except Exception:
        st.caption(f"ℹ️ No se encontró la imagen `{ruta}` en el repositorio.")
 
 
def cargar_dataset(archivo):
    """Carga y valida el archivo subido por el usuario mediante
    st.file_uploader(). El caso de estudio exige soporte para .csv; se
    agrega también soporte opcional para .xlsx (usando openpyxl) por si se
    necesita cargar una versión del dataset exportada desde Excel. Retorna
    None si no se seleccionó archivo, si la extensión no es válida, o si
    ocurre un error de lectura."""
    if archivo is None:
        return None
 
    nombre = archivo.name.lower()
 
    try:
        if nombre.endswith(".csv"):
            datos = pd.read_csv(archivo)
        elif nombre.endswith(".xlsx"):
            datos = pd.read_excel(archivo)
        else:
            st.error(
                "El archivo seleccionado debe estar en formato .csv o .xlsx."
            )
            return None
        return datos
    except Exception as error:
        st.error(f"No fue posible leer el archivo: {error}")
        return None
 
 
def normalizar_renewal(df):
    """Homogeniza la variable objetivo `renewal` a formato numérico (0/1),
    sin importar si el archivo cargado la trae como texto ("Yes"/"No") o
    ya como número (1/0), como ocurre en InsuranceCompany.csv. Esto hace
    la app más robusta ante variaciones del archivo de origen."""
    if df["renewal"].dtype == object:
        mapeo = {
            "yes": 1, "sí": 1, "si": 1, "1": 1,
            "no": 0, "0": 0
        }
        df["renewal"] = (
            df["renewal"].astype(str).str.strip().str.lower().map(mapeo)
        )
    return df
 
 
def agregar_edad_anios(df):
    """Crea la columna derivada `age_years` a partir de `age_in_days`.
    El dataset original expresa la edad en días, lo cual dificulta la
    interpretación (ej. 12,058 días en vez de ~33 años). Esta función usa
    una operación vectorizada de Pandas/NumPy (división de toda la
    columna en un solo paso, sin loops) para generar una variable mucho
    más intuitiva para el análisis y las visualizaciones."""
    if "age_in_days" in df.columns and "age_years" not in df.columns:
        df["age_years"] = (df["age_in_days"] / 365).round(1)
    return df
 
 
def mostrar_metrica_resumen(df):
    """Muestra 4 métricas clave en la parte superior del dashboard usando
    st.columns() + st.metric(), para dar una lectura ejecutiva inmediata
    del dataset cargado."""
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
    """Histograma + curva de densidad (KDE) para observar la forma de la
    distribución de una variable numérica."""
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
    plt.close(fig)  # libera memoria; evita acumular figuras entre reruns
 
 
def grafico_categorico(df, columna):
    """Gráfico de barras con la frecuencia de cada categoría de una
    variable cualitativa."""
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
    """Genera una interpretación automática y en lenguaje natural (f-string)
    de la forma de la distribución, comparando media y mediana como
    indicador simple de asimetría."""
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
    """Boxplot de una variable numérica separada por grupo de renovación
    (Sí/No). Permite comparar visualmente ambos grupos y detecta outliers
    dentro de cada uno (los puntos fuera de los "bigotes" del boxplot)."""
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
    """Gráfico de barras apiladas (100%) que compara la proporción de
    renovación (Sí/No) entre las categorías de una variable cualitativa."""
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
# SIDEBAR - MENÚ PRINCIPAL DE NAVEGACIÓN
# ============================================================
# st.sidebar.image muestra el logo institucional en la parte superior del
# panel lateral. Se usa mostrar_imagen_segura() para que la app no se
# rompa si el archivo DMC.png todavía no fue subido al repositorio.
mostrar_imagen_segura("DMC.png", width=100)
 
st.sidebar.title("📚 Contenido")
 
# st.sidebar.selectbox crea el menú lateral que exige el caso de estudio
# para navegar entre los módulos "Home" y "Caso de Estudio N°3".
modulos = st.sidebar.selectbox(
    "Seleccione un módulo",
    ["Home", "Caso de Estudio N°3"]
)
 
# ============================================================
# MÓDULO 1: HOME (presentación del proyecto, sin análisis)
# ============================================================
if modulos == "Home":
 
    st.title("Trabajo Final - Módulo Python Fundamentals")
 
    # Logo de Python como imagen de portada del módulo Home.
    mostrar_imagen_segura("Python_logo.png", width=500)
 
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
 
    st.write("**Nombre completo:** David Sebastian Carlos Ipanaque")
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
 
        - 🔗 GitHub
        - 🎨 Streamlit
        - 🐍 Google Colab / Python
        - 🔢 NumPy
        - 🐼 Pandas
        - 📊 Matplotlib
        - 📈 Seaborn
        - 🧩 Programación Orientada a Objetos (POO)
        """
    )
 
# ============================================================
# MÓDULO 2 + EDA: CASO DE ESTUDIO N°3
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
        "📂 Seleccione su archivo",
        type=["csv", "xlsx"]
    )
 
    # Ningún análisis se ejecuta si `datos` es None (archivo no cargado
    # o inválido). Esta validación es la que exige el caso de estudio.
    datos = cargar_dataset(archivo)
 
    if datos is not None:
 
        # Validación adicional: la variable objetivo `renewal` es
        # obligatoria para todo el módulo de EDA (tasas de renovación,
        # análisis bivariado, hallazgos, etc.).
        if "renewal" not in datos.columns:
            st.error(
                "El archivo cargado no contiene la columna `renewal`, "
                "necesaria para este análisis. Verifique que el archivo "
                "corresponda a InsuranceCompany.csv."
            )
            st.stop()
 
        # Preprocesamiento ligero: homogeniza `renewal` a 0/1 y agrega la
        # columna derivada `age_years` (edad en años) para que el resto de
        # la app trabaje siempre sobre datos consistentes.
        datos = normalizar_renewal(datos)
        datos = agregar_edad_anios(datos)
 
        st.success(
            f"Archivo **{archivo.name}** cargado correctamente."
        )
 
        analyzer = DataAnalyzer(datos)
 
        # ----------------------------------------------------
        # VISTA PREVIA Y DIMENSIONES (Módulo 2 - Carga del dataset)
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
        # Se usan 7 pestañas para organizar los 10 ítems de análisis que
        # exige el caso de estudio. Algunas pestañas agrupan 2 ítems
        # relacionados (p. ej. Información + Clasificación de variables)
        # para mantener una navegación más limpia; cada ítem conserva su
        # propio st.header() numerado para que quede claramente identificado.
        tabs = st.tabs([
            "1️⃣-2️⃣ Información y Variables",
            "3️⃣-4️⃣ Estadística y Nulos",
            "5️⃣ Numéricas",
            "6️⃣ Categóricas",
            "7️⃣-8️⃣ Bivariado",
            "9️⃣ Dinámico",
            "🔟 Hallazgos"
        ])
 
        # ====================================================
        # ÍTEM 1: Información general del dataset
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
        # ÍTEM 2: Clasificación de variables
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
        # ÍTEM 3: Estadísticas descriptivas
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
 
            # Complemento: detección de outliers con la regla del IQR.
            # Ayuda a explicar por qué, en variables como `Income`, la
            # media puede quedar muy por encima de la mediana.
            outliers = analyzer.detectar_outliers_iqr(variable_estadistica)
 
            st.write("#### Detección de valores atípicos (regla IQR)")
            st.write(
                f"Se identificaron **{outliers['Cantidad de outliers']:,} "
                f"valores atípicos** ({outliers['Porcentaje (%)']}% del total) "
                f"fuera del rango [{outliers['Límite inferior']:,.2f}, "
                f"{outliers['Límite superior']:,.2f}]."
            )
 
        # ====================================================
        # ÍTEM 4: Análisis de valores faltantes
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
        # ÍTEM 5: Distribución de variables numéricas
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
        # ÍTEM 6: Análisis de variables categóricas
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
        # ÍTEM 7: Análisis bivariado - Numérica vs categórica
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
        # ÍTEM 8: Análisis bivariado - Categórica vs categórica
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
        # ÍTEM 9: Análisis basado en parámetros seleccionados
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
 
            st.subheader("Cruce dinámico con la variable objetivo")
 
            # Widget adicional: el usuario elige una variable CATEGÓRICA
            # cualquiera y la app calcula, "al vuelo", la tasa de
            # renovación por categoría usando el método de la clase
            # `tasa_renovacion`. Esto muestra un análisis verdaderamente
            # dinámico (no precalculado) según el parámetro elegido.
            variable_cruce = st.selectbox(
                "Seleccione una variable categórica para calcular su tasa de renovación:",
                [x for x in categoricas if x != "renewal"],
                key="cruce_categorica"
            )
 
            st.dataframe(
                analyzer.tasa_renovacion(variable_cruce),
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
        # ÍTEM 10: Hallazgos clave
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
                concentración de clientes en la categoría de renovación
                (dataset desbalanceado hacia "Sí renueva").
 
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
 
        # ====================================================
        # CONCLUSIONES FINALES (sección independiente del EDA)
        # ====================================================
        # El caso de estudio exige una sección separada con 5 conclusiones
        # redactadas por el estudiante, enfocadas en toma de decisiones y
        # no en predicción. Se ubica fuera de las tabs porque conceptualmente
        # es un cierre del proyecto completo, no un ítem más del EDA.
        st.divider()
        st.header("📝 Conclusiones finales")
 
        st.markdown(
            """
            **1.** La tasa de renovación del portafolio es alta (por encima del
            90%), lo que sugiere que la compañía mantiene, en general, una
            base de clientes fidelizada; sin embargo, el segmento que no
            renueva —aunque minoritario— representa una oportunidad concreta
            de mejora en retención.
 
            **2.** Los clientes morosos (con pagos atrasados de 3 a 6, de 6 a
            12 o de más de 12 meses) muestran, de forma descriptiva, una menor
            tendencia a renovar su póliza. Esto posiciona el historial de
            morosidad como una señal temprana útil para priorizar acciones
            comerciales de retención.
 
            **3.** El canal de captación (`sourcing_channel`) influye en el
            comportamiento de renovación observado: algunos canales concentran
            tasas de renovación más altas que otros, lo que puede orientar
            decisiones sobre en qué canales invertir esfuerzos comerciales.
 
            **4.** Variables económicas como el ingreso (`Income`) y el valor
            de la prima (`premium`) presentan distribuciones con asimetría y
            valores atípicos relevantes, por lo que cualquier análisis o
            reporte que use sus promedios debe complementarse con la mediana
            y la dispersión para no llegar a conclusiones distorsionadas.
 
            **5.** El puntaje de evaluación del cliente
            (`application_underwriting_score`) presenta una cantidad
            considerable de valores faltantes, por lo que antes de utilizarlo
            en futuros análisis (incluyendo eventuales modelos predictivos)
            se recomienda definir una estrategia explícita de tratamiento de
            datos faltantes.
            """
        )
 
        st.caption(
            "Estas conclusiones se basan en relaciones descriptivas "
            "identificadas mediante el EDA y buscan apoyar decisiones de "
            "negocio (por ejemplo, foco de campañas de retención), sin "
            "constituir un modelo predictivo."
        )
 
    else:
 
        st.info(
            "📂 Cargue el archivo **InsuranceCompany.csv** desde el panel "
            "lateral para iniciar el análisis."
        )
