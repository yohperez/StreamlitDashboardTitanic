import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium

# ===============================================================
# CONFIGURACIÓN DE PÁGINA
# ===============================================================
st.set_page_config(
    page_title="Dashboard Titanic - Análisis Interactivo",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================================
# CARGA Y LIMPIEZA DE DATOS (Bloque 1)
# ===============================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Limpieza de nulos
    mediana_edad = df['Age'].median()
    df['Age'] = df['Age'].fillna(mediana_edad)
    puerto_frecuente = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(puerto_frecuente)
    
    # Variables adicionales
    bins = [0, 12, 18, 60, 100]
    labels = ['Niño (<12)', 'Adolescente (12-18)', 'Adulto (18-60)', 'Anciano (>60)']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Codificación
    df['Sex_Coded'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked_Coded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df

df = load_data()

# Outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
df_sin_outliers = df[df['Fare'] <= limite_superior]

# Matriz de correlación
columnas_interes = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Sex_Coded', 'Embarked_Coded']
matriz_corr = df[columnas_interes].corr()

# Stats puertos
stats_puertos = df.groupby('Embarked').agg(
    Total=('PassengerId', 'count'),
    Supervivientes=('Survived', 'sum')
).reset_index()
stats_puertos['Tasa_Supervivencia'] = (stats_puertos['Supervivientes'] / stats_puertos['Total'] * 100).round(2)

# ===============================================================
# HEADER
# ===============================================================
st.markdown('<div class="main-header">🚢 RMS Titanic - Dashboard Analítico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis interactivo de supervivencia, demografía y economía del desastre</div>', unsafe_allow_html=True)

# ===============================================================
# KPIs EN LA PARTE SUPERIOR
# ===============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Pasajeros</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    survival_rate = df['Survived'].mean() * 100
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <div class="metric-value">{survival_rate:.1f}%</div>
        <div class="metric-label">Tasa Supervivencia</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);">
        <div class="metric-value">{df['Age'].median():.1f}</div>
        <div class="metric-label">Edad Mediana</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_fare = df['Fare'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-value">${avg_fare:.2f}</div>
        <div class="metric-label">Tarifa Promedio</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    outliers_count = len(df[df['Fare'] > limite_superior])
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="metric-value">{outliers_count}</div>
        <div class="metric-label">Outliers Tarifa</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===============================================================
# TABS PRINCIPALES
# ===============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Perfil Demográfico", 
    "💰 Dimensión Económica", 
    "🛡️ Supervivencia",
    "👨‍👩‍👧‍👦 Estructura Familiar", 
    "🗺️ Mapa Geoespacial",
    "📈 Dashboard Ejecutivo"
])

# ===============================================================
# TAB 1: PERFIL DEMOGRÁFICO (Retos 2, 3, 4)
# ===============================================================
with tab1:
    st.header("Perfil Demográfico de los Pasajeros")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Histograma de Edades con KDE (Reto 2)
        fig_age = px.histogram(
            df, x='Age', nbins=30,
            title="Distribución de Edades de los Pasajeros",
            labels={'Age': 'Edad', 'count': 'Frecuencia'},
            color_discrete_sequence=['#3b82f6'],
            opacity=0.7
        )
        fig_age.update_traces(marker_line_width=1, marker_line_color="white")
        fig_age.update_layout(
            height=400,
            xaxis_title="Edad",
            yaxis_title="Frecuencia",
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col_right:
        # Box Plot de Edad por Clase (Reto 4)
        fig_box = px.box(
            df, x='Pclass', y='Age', color='Pclass',
            title="Distribución de Edades por Clase",
            labels={'Pclass': 'Clase', 'Age': 'Edad'},
            color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
        )
        fig_box.update_layout(height=400, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Pasajeros por Clase y Género (Reto 3)
    col1, col2 = st.columns(2)
    
    with col1:
        class_counts = df['Pclass'].value_counts().sort_index().reset_index()
        class_counts.columns = ['Clase', 'Cantidad']
        
        fig_class = px.bar(
            class_counts, x='Clase', y='Cantidad',
            title="Cantidad de Pasajeros por Clase",
            labels={'Clase': 'Clase de Ticket', 'Cantidad': 'Cantidad'},
            color='Clase',
            color_discrete_sequence=['#60a5fa', '#34d399', '#f87171']
        )
        fig_class.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col2:
        gender_counts = df['Sex'].value_counts().reset_index()
        gender_counts.columns = ['Género', 'Cantidad']
        
        fig_gender = px.pie(
            gender_counts, values='Cantidad', names='Género',
            title="Distribución por Género",
            color='Género',
            color_discrete_map={'male': '#60a5fa', 'female': '#f472b6'},
            hole=0.4
        )
        fig_gender.update_layout(height=350, template="plotly_white")
        fig_gender.update_traces(textinfo='percent+label', pull=[0.02, 0.02])
        st.plotly_chart(fig_gender, use_container_width=True)

# ===============================================================
# TAB 2: DIMENSIÓN ECONÓMICA (Retos 5, 6)
# ===============================================================
with tab2:
    st.header("Dimensión Económica y Tarifas")
    
    # Tarifas promedio por clase
    tarifas_promedio = df.groupby('Pclass')['Fare'].mean().reset_index()
    tarifas_promedio.columns = ['Clase', 'Tarifa_Promedio']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tarifas = px.bar(
            tarifas_promedio, x='Clase', y='Tarifa_Promedio',
            title="Tarifas Promedio por Clase (Con Outliers)",
            labels={'Clase': 'Clase', 'Tarifa_Promedio': 'Tarifa Promedio ($)'},
            color='Clase',
            color_discrete_sequence=['#3b82f6', '#8b5cf6', '#ec4899'],
            text='Tarifa_Promedio'
        )
        fig_tarifas.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_tarifas.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_tarifas, use_container_width=True)
    
    with col2:
        # Box plot de tarifas sin outliers
        fig_box_fare = px.box(
            df_sin_outliers, x='Pclass', y='Fare', color='Pclass',
            title="Distribución de Tarifas por Clase (Sin Outliers)",
            labels={'Pclass': 'Clase', 'Fare': 'Tarifa ($)'},
            color_discrete_sequence=['#3b82f6', '#8b5cf6', '#ec4899']
        )
        fig_box_fare.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_box_fare, use_container_width=True)
    
    # Scatter Plot Edad vs Tarifa (Reto 6)
    st.subheader("Relación entre Edad y Tarifa Pagada")
    
    fig_scatter = px.scatter(
        df, x='Age', y='Fare', color='Survived',
        title="Edad vs Tarifa (Supervivientes vs Fallecidos)",
        labels={'Age': 'Edad', 'Fare': 'Tarifa ($)', 'Survived': '¿Sobrevivió?'},
        color_discrete_map={0: '#dc2626', 1: '#16a34a'},
        hover_data=['Pclass', 'Sex', 'Embarked'],
        opacity=0.7
    )
    fig_scatter.update_layout(
        height=500,
        template="plotly_white",
        yaxis_type="log",
        legend_title_text='¿Sobrevivió?'
    )
    fig_scatter.for_each_trace(lambda t: t.update(name='No' if t.name == '0' else 'Sí'))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Info de outliers
    st.info(f"📊 **Límite superior para outliers**: ${limite_superior:.2f} | **Pasajeros con tarifas atípicas**: {len(df[df['Fare'] > limite_superior])}")

# ===============================================================
# TAB 3: PATRONES DE SUPERVIVENCIA (Retos 7, 8, 9)
# ===============================================================
with tab3:
    st.header("Patrones de Supervivencia y Protocolo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supervivencia por Género (Reto 7)
        surv_gender = df.groupby('Sex')['Survived'].mean().reset_index()
        surv_gender.columns = ['Género', 'Tasa_Supervivencia']
        surv_gender['Tasa_Supervivencia'] = surv_gender['Tasa_Supervivencia'] * 100
        
        fig_surv_gender = px.bar(
            surv_gender, x='Género', y='Tasa_Supervivencia',
            title="Proporción de Supervivencia por Género",
            labels={'Género': 'Género', 'Tasa_Supervivencia': 'Tasa de Supervivencia (%)'},
            color='Género',
            color_discrete_map={'male': '#60a5fa', 'female': '#f472b6'},
            text='Tasa_Supervivencia'
        )
        fig_surv_gender.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_surv_gender.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_surv_gender, use_container_width=True)
    
    with col2:
        # Supervivencia por Rango de Edad (Reto 8)
        surv_age = df.groupby('AgeGroup')['Survived'].mean().reset_index()
        surv_age.columns = ['Rango_Edad', 'Tasa_Supervivencia']
        surv_age['Tasa_Supervivencia'] = surv_age['Tasa_Supervivencia'] * 100
        
        fig_surv_age = px.bar(
            surv_age, x='Rango_Edad', y='Tasa_Supervivencia',
            title="Proporción de Supervivencia por Rangos de Edad",
            labels={'Rango_Edad': 'Rango de Edad', 'Tasa_Supervivencia': 'Tasa de Supervivencia (%)'},
            color='Rango_Edad',
            color_discrete_sequence=['#34d399', '#fbbf24', '#60a5fa', '#a78bfa'],
            text='Tasa_Supervivencia'
        )
        fig_surv_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_surv_age.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_surv_age, use_container_width=True)
    
    # Supervivencia por Clase y Género (Reto 9)
    st.subheader("Tasa de Supervivencia Cruzada: Clase y Género")
    
    surv_class_gender = df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
    surv_class_gender.columns = ['Clase', 'Género', 'Tasa_Supervivencia']
    surv_class_gender['Tasa_Supervivencia'] = surv_class_gender['Tasa_Supervivencia'] * 100
    
    fig_class_gender = px.bar(
        surv_class_gender, x='Clase', y='Tasa_Supervivencia', color='Género',
        barmode='group',
        title="Tasa de Supervivencia por Clase y Género",
        labels={'Clase': 'Clase de Pasajero', 'Tasa_Supervivencia': 'Tasa de Supervivencia (%)'},
        color_discrete_map={'male': '#60a5fa', 'female': '#f472b6'},
        text='Tasa_Supervivencia'
    )
    fig_class_gender.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_class_gender.update_layout(height=450, template="plotly_white")
    st.plotly_chart(fig_class_gender, use_container_width=True)

# ===============================================================
# TAB 4: ESTRUCTURA FAMILIAR (Retos 10, 11, 12)
# ===============================================================
with tab4:
    st.header("Estructura Familiar y Correlación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supervivencia por Tamaño de Familia (Reto 10)
        surv_family = df.groupby('FamilySize')['Survived'].mean().reset_index()
        surv_family.columns = ['Tamaño_Familia', 'Tasa_Supervivencia']
        surv_family['Tasa_Supervivencia'] = surv_family['Tasa_Supervivencia'] * 100
        
        fig_family = px.bar(
            surv_family, x='Tamaño_Familia', y='Tasa_Supervivencia',
            title="Supervivencia según Tamaño de la Familia",
            labels={'Tamaño_Familia': 'Número de Miembros', 'Tasa_Supervivencia': 'Tasa de Supervivencia (%)'},
            color='Tasa_Supervivencia',
            color_continuous_scale='RdYlBu',
            text='Tasa_Supervivencia'
        )
        fig_family.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_family.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Supervivencia por Puerto (Reto 11)
        surv_embarked = df.groupby('Embarked')['Survived'].mean().reset_index()
        surv_embarked.columns = ['Puerto', 'Tasa_Supervivencia']
        surv_embarked['Tasa_Supervivencia'] = surv_embarked['Tasa_Supervivencia'] * 100
        puerto_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
        surv_embarked['Puerto_Nombre'] = surv_embarked['Puerto'].map(puerto_names)
        
        fig_embarked = px.bar(
            surv_embarked, x='Puerto_Nombre', y='Tasa_Supervivencia',
            title="Supervivencia según Puerto de Embarque",
            labels={'Puerto_Nombre': 'Puerto', 'Tasa_Supervivencia': 'Tasa de Supervivencia (%)'},
            color='Puerto_Nombre',
            color_discrete_sequence=['#34d399', '#60a5fa', '#fbbf24'],
            text='Tasa_Supervivencia'
        )
        fig_embarked.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_embarked.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_embarked, use_container_width=True)
    
    # Mapa de Calor de Correlaciones (Reto 12)
    st.subheader("Mapa de Calor de Correlaciones de Pearson")
    
    fig_heatmap = px.imshow(
        matriz_corr,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlación entre Variables",
        zmin=-1, zmax=1
    )
    fig_heatmap.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ===============================================================
# TAB 5: MAPA GEOESPACIAL (Reto 14)
# ===============================================================
with tab5:
    st.header("Mapa Geoespacial Interactivo - Ruta del Titanic")
    
    # Crear mapa con Folium
    coordenadas = {
        'Southampton': [50.9097, -1.4044],
        'Cherbourg': [49.6337, -1.6221],
        'Queenstown': [51.8503, -8.2943],
        'Hundimiento': [41.7269, -49.9482]
    }
    
    puerto_info = {
        'S': {'nombre': 'Southampton', 'stats': stats_puertos[stats_puertos['Embarked'] == 'S'].iloc[0] if len(stats_puertos[stats_puertos['Embarked'] == 'S']) > 0 else None},
        'C': {'nombre': 'Cherbourg', 'stats': stats_puertos[stats_puertos['Embarked'] == 'C'].iloc[0] if len(stats_puertos[stats_puertos['Embarked'] == 'C']) > 0 else None},
        'Q': {'nombre': 'Queenstown', 'stats': stats_puertos[stats_puertos['Embarked'] == 'Q'].iloc[0] if len(stats_puertos[stats_puertos['Embarked'] == 'Q']) > 0 else None}
    }
    
    mapa_titanic = folium.Map(
        location=[47.0, -25.0],
        zoom_start=4,
        tiles='cartodbpositron'
    )
    
    # Marcadores para puertos
    for codigo, info in puerto_info.items():
        if info['stats'] is not None:
            stats = info['stats']
            color_tasa = "green" if stats['Tasa_Supervivencia'] > 45 else "red"
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 220px; font-size: 12px; line-height: 1.4;">
                <h4 style="margin: 0 0 8px 0; color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 3px;">
                    Puerto: {info['nombre']} ({codigo})
                </h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 3px 0; color: #7f8c8d;"><b>Pasajeros:</b></td>
                        <td style="padding: 3px 0; text-align: right; font-weight: bold;">{stats['Total']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px 0; color: #7f8c8d;"><b>Supervivencia:</b></td>
                        <td style="padding: 3px 0; text-align: right; font-weight: bold; color: {color_tasa};">{stats['Tasa_Supervivencia']}%</td>
                    </tr>
                </table>
            </div>
            """
            folium.Marker(
                location=coordenadas[info['nombre']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Puerto: {info['nombre']}",
                icon=folium.Icon(color='blue', icon='ship', prefix='fa')
            ).add_to(mapa_titanic)
    
    # Marcador de hundimiento
    html_tragedia = """
    <div style="font-family: Arial, sans-serif; width: 200px; font-size: 12px;">
        <h4 style="margin: 0 0 5px 0; color: #c0392b; font-weight: bold; border-bottom: 2px solid #c0392b;">LUGAR DEL HUNDIMIENTO</h4>
        <p style="margin: 5px 0 0 0; line-height: 1.3;">
            El <b>RMS Titanic</b> colisionó contra un iceberg el 14 de abril de 1912 a las 23:40 y se hundió en la madrugada del 15 de abril de 1912.
        </p>
    </div>
    """
    folium.Marker(
        location=coordenadas['Hundimiento'],
        popup=folium.Popup(html_tragedia, max_width=250),
        tooltip="⚠️ Zona de Colisión y Hundimiento",
        icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
    ).add_to(mapa_titanic)
    
    # Ruta
    ruta_coordenadas = [
        coordenadas['Southampton'],
        coordenadas['Cherbourg'],
        coordenadas['Queenstown'],
        coordenadas['Hundimiento']
    ]
    folium.PolyLine(
        locations=ruta_coordenadas,
        color='#34495e',
        weight=3,
        opacity=0.8,
        dash_array='8, 8',
        tooltip="Ruta de navegación del RMS Titanic"
    ).add_to(mapa_titanic)
    
    # Minimapa
    minimap = plugins.MiniMap(toggle_display=True)
    mapa_titanic.add_child(minimap)
    
    st_folium(mapa_titanic, width=1200, height=600)
    
    # Tabla de stats
    st.subheader("Estadísticas por Puerto de Embarque")
    stats_display = stats_puertos.copy()
    stats_display.columns = ['Puerto', 'Total Pasajeros', 'Supervivientes', 'Tasa Supervivencia (%)']
    puerto_full = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    stats_display['Puerto'] = stats_display['Puerto'].map(puerto_full)
    st.dataframe(stats_display, use_container_width=True, hide_index=True)

# ===============================================================
# TAB 6: DASHBOARD EJECUTIVO (Reto 13)
# ===============================================================
with tab6:
    st.header("Dashboard Ejecutivo - Resumen de Hallazgos Clave")
    
    # Layout 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico 1: Supervivencia por Clase y Género
        surv_class_gender = df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
        surv_class_gender['Survived'] = surv_class_gender['Survived'] * 100
        
        fig1 = px.bar(
            surv_class_gender, x='Pclass', y='Survived', color='Sex',
            barmode='group',
            title="1. Supervivencia por Clase y Género",
            labels={'Pclass': 'Clase', 'Survived': 'Tasa Supervivencia (%)', 'Sex': 'Género'},
            color_discrete_map={'male': '#60a5fa', 'female': '#f472b6'}
        )
        fig1.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico 3: Distribución de Tarifas por Clase (sin outliers)
        fig3 = px.box(
            df_sin_outliers, x='Pclass', y='Fare', color='Pclass',
            title="3. Distribución de Tarifas por Clase (Sin Outliers)",
            labels={'Pclass': 'Clase', 'Fare': 'Tarifa ($)'},
            color_discrete_sequence=['#60a5fa', '#8b5cf6', '#ec4899']
        )
        fig3.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Gráfico 2: Supervivencia por Tamaño de Familia
        surv_family = df.groupby('FamilySize')['Survived'].mean().reset_index()
        surv_family['Survived'] = surv_family['Survived'] * 100
        
        fig2 = px.bar(
            surv_family, x='FamilySize', y='Survived',
            title="2. Supervivencia según Tamaño Familiar",
            labels={'FamilySize': 'Tamaño Familia', 'Survived': 'Tasa Supervivencia (%)'},
            color='Survived',
            color_continuous_scale='RdYlBu'
        )
        fig2.update_layout(height=350, template="plotly_white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico 4: Correlación de variables clave
        cols_clave = ['Survived', 'Pclass', 'Sex_Coded', 'Fare', 'FamilySize']
        corr_clave = df[cols_clave].corr()
        
        fig4 = px.imshow(
            corr_clave,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="4. Correlación de Variables Clave",
            zmin=-1, zmax=1
        )
        fig4.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)

# ===============================================================
# SIDEBAR - FILTROS INTERACTIVOS
# ===============================================================
st.sidebar.markdown("## 🎛️ Filtros Interactivos")
st.sidebar.markdown("---")

# Filtro por clase
clases_sel = st.sidebar.multiselect(
    "Clase de Ticket",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

# Filtro por género
generos_sel = st.sidebar.multiselect(
    "Género",
    options=['male', 'female'],
    default=['male', 'female'],
    format_func=lambda x: 'Masculino' if x == 'male' else 'Femenino'
)

# Filtro por puerto
puertos_sel = st.sidebar.multiselect(
    "Puerto de Embarque",
    options=['S', 'C', 'Q'],
    default=['S', 'C', 'Q'],
    format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x]
)

# Filtro por edad
edad_range = st.sidebar.slider(
    "Rango de Edad",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(0, 80)
)

# Aplicar filtros
df_filtered = df[
    (df['Pclass'].isin(clases_sel)) &
    (df['Sex'].isin(generos_sel)) &
    (df['Embarked'].isin(puertos_sel)) &
    (df['Age'] >= edad_range[0]) &
    (df['Age'] <= edad_range[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Pasajeros filtrados:** {len(df_filtered):,} / {len(df):,}")
st.sidebar.markdown(f"**Tasa de supervivencia (filtrado):** {df_filtered['Survived'].mean()*100:.1f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Dashboard creado con Streamlit + Plotly**")
st.sidebar.markdown("🚢 Datos: Titanic Dataset (Kaggle)")
