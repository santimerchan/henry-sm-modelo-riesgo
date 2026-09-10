"""App de Streamlit para visualizar el monitoreo de data drift.

Lee los archivos que genera `model_monitoring.py` (drift_report.json y
drift_history.csv) y los presenta en tres pestañas: métricas de la última
corrida, evolución en el tiempo, y recomendaciones automáticas.

Uso:
    streamlit run mlops_pipeline/src/app_monitoreo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Permitimos importar model_monitoring aunque Streamlit ejecute el archivo
# como script suelto: agregamos su carpeta al path de Python.
sys.path.append(str(Path(__file__).resolve().parent))

from model_monitoring import (  # noqa: E402
    ARCHIVO_HISTORIAL,
    ARCHIVO_REPORTE,
    CHI2_PVALUE_UMBRAL,
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    JS_UMBRAL,
    KS_UMBRAL,
    PSI_UMBRAL,
    RAIZ_PROYECTO,
    generar_batch_actual,
)

st.set_page_config(
    page_title="Monitoreo de Data Drift | Riesgo Crediticio",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_reporte() -> dict | None:
    """Lee el reporte de la última corrida de monitoreo.

    Returns:
        El reporte como diccionario, o None si todavía no se ha ejecutado el
        monitoreo.
    """
    ruta = RAIZ_PROYECTO / ARCHIVO_REPORTE
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_historial() -> pd.DataFrame:
    """Lee el historial acumulado de corridas de monitoreo.

    Returns:
        DataFrame con el historial, o un DataFrame vacío si el archivo no existe.
    """
    ruta = RAIZ_PROYECTO / ARCHIVO_HISTORIAL
    if not ruta.exists():
        return pd.DataFrame()
    df = pd.read_csv(ruta)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data
def cargar_distribuciones() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Carga el histórico y regenera el batch actual para poder graficarlos.

    Se reutiliza `generar_batch_actual` con la misma semilla que usa el script
    de monitoreo, así que las distribuciones que ves aquí son exactamente las
    que se midieron en el reporte.

    Returns:
        Tupla (df_historico, df_actual), o (None, None) si no está el CSV.
    """
    ruta = RAIZ_PROYECTO / "Base_de_datos.csv"
    if not ruta.exists():
        return None, None
    df_historico = pd.read_csv(ruta)
    df_actual = generar_batch_actual(df_historico)
    return df_historico, df_actual


# ---------------------------------------------------------------------------
# UTILIDADES DE PRESENTACIÓN
# ---------------------------------------------------------------------------
def semaforo(resultado: dict) -> tuple[str, str]:
    """Traduce las métricas de una variable a un semáforo de tres colores.

    Rojo si alguna métrica supera su umbral, amarillo si alguna está en zona de
    vigilancia (por encima del 60% del umbral) y verde si todo está tranquilo.

    Args:
        resultado: Diccionario de una variable dentro del reporte.

    Returns:
        Tupla (emoji, etiqueta) lista para mostrar.
    """
    if resultado.get("drift_detectado"):
        return "🔴", "Drift detectado"

    valores_en_vigilancia = []
    if resultado.get("psi") is not None:
        valores_en_vigilancia.append(resultado["psi"] > PSI_UMBRAL * 0.6)
    if resultado.get("ks_estadistico") is not None:
        valores_en_vigilancia.append(resultado["ks_estadistico"] > KS_UMBRAL * 0.6)
    if resultado.get("jensen_shannon") is not None:
        valores_en_vigilancia.append(resultado["jensen_shannon"] > JS_UMBRAL * 0.6)

    if any(valores_en_vigilancia):
        return "🟡", "En vigilancia"
    return "🟢", "Estable"


def formatear(valor: float | None, decimales: int = 4) -> str:
    """Formatea un número para la tabla, mostrando un guion si no aplica.

    Args:
        valor: Número a formatear (puede ser None).
        decimales: Cantidad de decimales.

    Returns:
        El número como texto, o '—' si es None.
    """
    return "—" if valor is None else f"{valor:.{decimales}f}"


# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.title("📊 Monitoreo de Data Drift — Modelo de Riesgo Crediticio")

reporte = cargar_reporte()
historial = cargar_historial()

if reporte is None:
    st.error(
        "Todavía no encuentro `drift_report.json`. Ejecuta primero el monitoreo con:\n\n"
        "```bash\npython mlops_pipeline/src/model_monitoring.py\n```"
    )
    st.stop()

st.caption(
    f"Última corrida: **{reporte['timestamp']}** · "
    f"{reporte['n_historico']:,} filas históricas vs {reporte['n_actual']:,} filas del batch actual"
)

columna_a, columna_b, columna_c, columna_d = st.columns(4)
columna_a.metric("Variables evaluadas", reporte["variables_evaluadas"])
columna_b.metric("Variables con drift", reporte["n_variables_con_drift"])
columna_c.metric(
    "Estado global",
    "Con drift" if reporte["drift_detectado_global"] else "Estable",
)
columna_d.metric("Corridas registradas", historial["timestamp"].nunique() if not historial.empty else 0)

st.info(
    "⚠️ El lote 'actual' es **simulado** a partir del histórico mientras no exista "
    "un log real de predicciones en producción. La lógica de las métricas y los "
    "umbrales sí es la definitiva.",
    icon="ℹ️",
)

tab_metricas, tab_temporal, tab_alertas = st.tabs(
    ["📈 Visualización de métricas", "🕒 Análisis temporal", "🚨 Recomendaciones y alertas"]
)

# ---------------------------------------------------------------------------
# TAB 1 — VISUALIZACIÓN DE MÉTRICAS
# ---------------------------------------------------------------------------
with tab_metricas:
    st.subheader("Distribución: histórico vs batch actual")

    resultados_por_variable = {r["variable"]: r for r in reporte["resultados"]}
    variable = st.selectbox(
        "Selecciona la variable que quieres inspeccionar:",
        options=list(resultados_por_variable.keys()),
    )

    df_historico, df_actual = cargar_distribuciones()

    if df_historico is None:
        st.warning("No encontré `Base_de_datos.csv`, así que no puedo graficar las distribuciones.")
    else:
        resultado = resultados_por_variable[variable]
        figura, eje = plt.subplots(figsize=(9, 4))

        if resultado["tipo"] == "numerica":
            serie_historica = pd.to_numeric(df_historico[variable], errors="coerce").dropna()
            serie_actual = pd.to_numeric(df_actual[variable], errors="coerce").dropna()

            eje.hist(serie_historica, bins=40, alpha=0.6, density=True, label="Histórico")
            eje.hist(serie_actual, bins=40, alpha=0.6, density=True, label="Batch actual")
            eje.set_ylabel("Densidad")
        else:
            proporcion_historica = df_historico[variable].astype(str).value_counts(normalize=True)
            proporcion_actual = df_actual[variable].astype(str).value_counts(normalize=True)

            comparacion = pd.DataFrame(
                {"Histórico": proporcion_historica, "Batch actual": proporcion_actual}
            ).fillna(0)
            # Mostramos solo las categorías con peso real para que el gráfico se lea.
            comparacion = comparacion.sort_values("Histórico", ascending=False).head(10)
            comparacion.plot(kind="bar", ax=eje)
            eje.set_ylabel("Proporción")

        eje.set_xlabel(variable)
        eje.set_title(f"{variable}: histórico vs batch actual")
        eje.legend()
        figura.tight_layout()
        st.pyplot(figura)
        plt.close(figura)

        emoji, etiqueta = semaforo(resultado)
        st.markdown(f"### {emoji} `{variable}` — {etiqueta}")

    st.divider()
    st.subheader("Tabla de métricas por variable")

    filas = []
    for resultado in reporte["resultados"]:
        emoji, etiqueta = semaforo(resultado)
        filas.append(
            {
                "Semáforo": emoji,
                "Variable": resultado["variable"],
                "Tipo": resultado["tipo"],
                "PSI": formatear(resultado["psi"]),
                "KS": formatear(resultado["ks_estadistico"]),
                "Jensen-Shannon": formatear(resultado["jensen_shannon"]),
                "Chi² p-value": formatear(resultado["chi2_pvalue"], 6),
                "Estado": etiqueta,
            }
        )

    st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)

    st.caption(
        f"Umbrales usados: PSI > {PSI_UMBRAL} · KS > {KS_UMBRAL} · "
        f"Jensen-Shannon > {JS_UMBRAL} · Chi² p-value < {CHI2_PVALUE_UMBRAL}. "
        "🟢 estable · 🟡 en vigilancia (por encima del 60% del umbral) · 🔴 drift detectado."
    )

# ---------------------------------------------------------------------------
# TAB 2 — ANÁLISIS TEMPORAL
# ---------------------------------------------------------------------------
with tab_temporal:
    st.subheader("Evolución de la métrica principal por variable")

    if historial.empty:
        st.warning(
            "Todavía no hay historial. Ejecuta `model_monitoring.py` al menos una vez "
            "para empezar a acumular corridas en `drift_history.csv`."
        )
    else:
        n_corridas = historial["timestamp"].nunique()
        st.caption(
            f"Historial con **{n_corridas}** corrida(s) registrada(s). "
            "La métrica principal es PSI para las numéricas y Jensen-Shannon para las categóricas."
        )

        if n_corridas < 2:
            st.info(
                "Con una sola corrida no se puede hablar de tendencia todavía. "
                "Vuelve a ejecutar el monitoreo para comparar contra esta primera medición.",
                icon="ℹ️",
            )

        variables_disponibles = sorted(historial["variable"].unique())
        seleccionadas = st.multiselect(
            "Variables a graficar:",
            options=variables_disponibles,
            default=variables_disponibles[: min(5, len(variables_disponibles))],
        )

        if seleccionadas:
            figura, eje = plt.subplots(figsize=(10, 4.5))
            for nombre in seleccionadas:
                serie = historial[historial["variable"] == nombre].sort_values("timestamp")
                metrica = serie["metrica_principal"].iloc[0]
                eje.plot(serie["timestamp"], serie[metrica], marker="o", label=f"{nombre} ({metrica})")

            eje.axhline(PSI_UMBRAL, linestyle="--", color="red", alpha=0.5, label=f"Umbral PSI ({PSI_UMBRAL})")
            eje.axhline(JS_UMBRAL, linestyle=":", color="orange", alpha=0.5, label=f"Umbral JS ({JS_UMBRAL})")
            eje.set_xlabel("Fecha de la corrida")
            eje.set_ylabel("Valor de la métrica principal")
            eje.set_title("Evolución del drift en el tiempo")
            eje.legend(fontsize=8, loc="best")
            figura.autofmt_xdate()
            figura.tight_layout()
            st.pyplot(figura)
            plt.close(figura)

        st.divider()
        st.subheader("Tendencia de las últimas corridas")
        st.caption(
            "Comparamos el promedio de la primera mitad de las corridas contra el de la "
            "segunda mitad. Si la diferencia es menor al 10%, la consideramos estable."
        )

        filas_tendencia = []
        for nombre in variables_disponibles:
            serie = historial[historial["variable"] == nombre].sort_values("timestamp")
            metrica = serie["metrica_principal"].iloc[0]
            valores = serie[metrica].dropna().to_numpy()

            if len(valores) < 2:
                tendencia, detalle = "Sin datos suficientes", "Se necesita más de una corrida."
            else:
                mitad = len(valores) // 2
                promedio_inicial = valores[:mitad].mean() if mitad > 0 else valores[0]
                promedio_final = valores[mitad:].mean()
                referencia = abs(promedio_inicial) if promedio_inicial != 0 else 1e-9
                variacion = (promedio_final - promedio_inicial) / referencia

                if variacion > 0.10:
                    tendencia = "📈 Sube"
                elif variacion < -0.10:
                    tendencia = "📉 Baja"
                else:
                    tendencia = "➡️ Estable"
                detalle = f"{promedio_inicial:.4f} → {promedio_final:.4f} ({variacion:+.1%})"

            filas_tendencia.append(
                {
                    "Variable": nombre,
                    "Métrica": metrica,
                    "Corridas": len(valores),
                    "Tendencia": tendencia,
                    "Detalle": detalle,
                }
            )

        st.dataframe(pd.DataFrame(filas_tendencia), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 — RECOMENDACIONES Y ALERTAS
# ---------------------------------------------------------------------------
with tab_alertas:
    st.subheader("Diagnóstico automático de la última corrida")

    variables_con_drift = reporte["variables_con_drift"]
    total_variables = reporte["variables_evaluadas"]
    n_con_drift = reporte["n_variables_con_drift"]
    proporcion = n_con_drift / total_variables if total_variables else 0

    if n_con_drift == 0:
        severidad, color_alerta = "Ninguna", "success"
    elif proporcion >= 0.30:
        severidad, color_alerta = "Alta", "error"
    else:
        severidad, color_alerta = "Media", "warning"

    st.metric("Severidad de la alerta", severidad)

    mensaje_principal = (
        f"Se detectó drift en **{n_con_drift} de {total_variables}** variables "
        f"({proporcion:.0%} del total)."
        if n_con_drift
        else "No se detectó drift. Las distribuciones se mantienen dentro de los umbrales definidos."
    )
    getattr(st, color_alerta)(mensaje_principal)

    st.divider()
    st.subheader("Recomendaciones")

    if n_con_drift == 0:
        st.markdown(
            "- ✅ No necesitas reentrenar el modelo por ahora.\n"
            "- 🔁 Mantén la corrida programada (semanal) para seguir acumulando historial.\n"
            "- 📌 Guarda esta corrida como línea base para comparar la próxima."
        )
    else:
        numericas_con_drift = [
            r["variable"] for r in reporte["resultados"]
            if r["drift_detectado"] and r["tipo"] == "numerica"
        ]
        categoricas_con_drift = [
            r["variable"] for r in reporte["resultados"]
            if r["drift_detectado"] and r["tipo"] == "categorica"
        ]

        st.markdown("**Variables que debes revisar primero:**")
        for resultado in reporte["resultados"]:
            if not resultado["drift_detectado"]:
                continue
            metrica = resultado["metrica_principal"]
            valor = resultado.get(metrica)
            st.markdown(
                f"- 🔴 `{resultado['variable']}` ({resultado['tipo']}) — "
                f"{metrica.upper()} = {formatear(valor)}"
            )

        st.markdown("**Qué hacer a continuación:**")
        acciones = [
            "🔍 Confirma con el equipo de datos si hubo un cambio en la fuente, en el ETL "
            "o en las reglas de negocio que explique el movimiento.",
        ]
        if numericas_con_drift:
            acciones.append(
                f"📐 Revisa la escala y los outliers de {', '.join(f'`{v}`' for v in numericas_con_drift)}: "
                "un cambio de unidad o un tope nuevo puede verse como drift sin serlo."
            )
        if categoricas_con_drift:
            acciones.append(
                f"🏷️ Verifica el catálogo de categorías de {', '.join(f'`{v}`' for v in categoricas_con_drift)}: "
                "una categoría nueva o renombrada rompe el encoding del modelo."
            )
        if severidad == "Alta":
            acciones.append(
                "♻️ **Reentrena el modelo** con datos recientes: con este nivel de drift el "
                "desempeño en producción muy probablemente ya se degradó."
            )
        else:
            acciones.append(
                "👀 Aún no es obligatorio reentrenar, pero deja estas variables en vigilancia "
                "y compara contra la próxima corrida."
            )
        acciones.append(
            "📊 Contrasta con las métricas de negocio (tasa de mora observada) antes de "
            "tomar la decisión final de reentrenamiento."
        )
        for accion in acciones:
            st.markdown(f"- {accion}")

    st.divider()
    st.subheader("Listado de alertas")

    alertas = []
    for resultado in reporte["resultados"]:
        emoji, etiqueta = semaforo(resultado)
        if etiqueta == "Estable":
            continue
        alertas.append(
            {
                "Severidad": "Alta" if etiqueta == "Drift detectado" else "Media",
                "Variable": resultado["variable"],
                "Tipo": resultado["tipo"],
                "Métrica principal": resultado["metrica_principal"],
                "Valor": formatear(resultado.get(resultado["metrica_principal"])),
                "Estado": f"{emoji} {etiqueta}",
            }
        )

    if alertas:
        st.dataframe(pd.DataFrame(alertas), width="stretch", hide_index=True)
    else:
        st.success("Sin alertas activas: ninguna variable superó ni se acercó a su umbral. 🟢")

    with st.expander("Ver configuración del monitoreo"):
        st.json(
            {
                "umbrales": reporte["umbrales"],
                "columnas_numericas": COLUMNAS_NUMERICAS,
                "columnas_categoricas": COLUMNAS_CATEGORICAS,
                "batch_actual_simulado": reporte["batch_actual_simulado"],
            }
        )
