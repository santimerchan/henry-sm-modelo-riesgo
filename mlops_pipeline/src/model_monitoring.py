"""Monitoreo de data drift para el modelo de riesgo crediticio.

Este módulo compara la distribución histórica de las variables (la que se usó
para entrenar el modelo) contra la distribución de un lote reciente, y reporta
si alguna variable se movió lo suficiente como para justificar una revisión o
un reentrenamiento.

Métricas implementadas:
    - KS (Kolmogorov-Smirnov)  -> variables numéricas
    - PSI (Population Stability Index) -> variables numéricas
    - Jensen-Shannon           -> numéricas (con bins) y categóricas
    - Chi-cuadrado             -> variables categóricas

Salidas que produce cada corrida (en la raíz del proyecto):
    - drift_report.json  : fotografía de la última corrida (se sobrescribe)
    - drift_history.csv  : una fila por corrida, para ver la evolución

Uso:
    python mlops_pipeline/src/model_monitoring.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

# ---------------------------------------------------------------------------
# UMBRALES DE DRIFT
# Son los umbrales acordados para este proyecto. Si los cambias, actualiza
# también el README y la app de monitoreo, porque el semáforo los usa.
# ---------------------------------------------------------------------------
PSI_UMBRAL = 0.25          # PSI > 0.25 -> cambio significativo en la distribución
KS_UMBRAL = 0.25           # estadístico KS > 0.25 -> cambio significativo
JS_UMBRAL = 0.3            # Jensen-Shannon > 0.3 -> cambio significativo
CHI2_PVALUE_UMBRAL = 0.05  # chi-cuadrado con p-value < 0.05 -> señal de cambio

# ---------------------------------------------------------------------------
# RUTAS Y CONFIGURACIÓN DEL DATASET
# ---------------------------------------------------------------------------
# Este archivo vive en mlops_pipeline/src/, así que la raíz del proyecto está
# dos niveles arriba. Resolvemos todo contra esa raíz para que el script
# funcione sin importar desde qué carpeta lo ejecutes.
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

ARCHIVO_REPORTE = "drift_report.json"
ARCHIVO_HISTORIAL = "drift_history.csv"

# Columnas reales de Base_de_datos.csv que vamos a monitorear.
# Se dejan fuera 'fecha_prestamo' (marca de tiempo, no es una feature) y
# 'Pago_atiempo' (es el target, no un input del modelo).
COLUMNAS_NUMERICAS: list[str] = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "saldo_total",
    "promedio_ingresos_datacredito",
]

COLUMNAS_CATEGORICAS: list[str] = [
    "tipo_laboral",
    "tendencia_ingresos",
    "tipo_credito",
]

# Categorías con una frecuencia menor a este porcentaje se agrupan en "OTROS".
# El dataset trae valores sucios en 'tendencia_ingresos' (números sueltos que
# aparecen una o dos veces); sin este agrupamiento el chi-cuadrado se vuelve
# inestable por celdas con frecuencia esperada casi cero.
FRECUENCIA_MINIMA_CATEGORIA = 0.01


def _resolver_ruta(path: str | Path) -> Path:
    """Convierte una ruta relativa en absoluta tomando como base la raíz del proyecto.

    Args:
        path: Ruta relativa o absoluta.

    Returns:
        La ruta absoluta equivalente.
    """
    ruta = Path(path)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos_historicos(path: str) -> pd.DataFrame:
    """Carga el dataset histórico que sirve como distribución de referencia.

    Esta es la distribución contra la que se compara todo: representa los datos
    con los que el modelo fue entrenado y, por lo tanto, el mundo que el modelo
    "conoce".

    Args:
        path: Ruta al CSV histórico. Si es relativa, se resuelve contra la raíz
            del proyecto.

    Returns:
        DataFrame con los datos históricos.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    ruta = _resolver_ruta(path)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encontré el archivo de datos históricos en: {ruta}. "
            "Verifica que Base_de_datos.csv esté en la raíz del proyecto."
        )
    df = pd.read_csv(ruta)
    print(f"Datos históricos cargados: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


# ---------------------------------------------------------------------------
# 2. GENERACIÓN DEL BATCH ACTUAL (SIMULADO)
# ---------------------------------------------------------------------------
# ===========================================================================
# TODO: reemplazar esta simulación por el log real de predicciones en
# producción (inputs + predicción del modelo) en cuanto ese pipeline exista.
# ===========================================================================
def generar_batch_actual(
    df_historico: pd.DataFrame,
    frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Simula el lote de datos "actual" que llegaría desde producción.

    Mientras no exista un log real de predicciones, construimos el lote actual
    tomando una muestra del histórico y aplicándole una perturbación controlada
    y reproducible. La perturbación es intencional: queremos que el monitoreo
    encuentre drift para poder mostrar cómo se ve una alerta real.

    Perturbaciones aplicadas:
        - Numéricas: se suma ruido gaussiano con desviación igual al 10% de la
          desviación estándar de cada variable.
        - Desplazamiento de media (drift fuerte y deliberado):
            * 'puntaje': baja 0.8 desviaciones estándar (simula una cartera que
              se deterioró).
            * 'salario_cliente': sube 0.6 desviaciones estándar (simula un
              cambio en el perfil de clientes que está entrando).
        - Categóricas: se reponderan las proporciones para que las categorías
          dejen de aparecer con la misma frecuencia que en el histórico.

    Args:
        df_historico: DataFrame de referencia del cual se toma la muestra.
        frac: Fracción del histórico a muestrear. Por defecto 0.3.
        seed: Semilla para que la simulación sea reproducible en cualquier máquina.

    Returns:
        DataFrame con el lote "actual" perturbado.
    """
    rng = np.random.default_rng(seed)
    df_actual = df_historico.sample(frac=frac, random_state=seed).copy()

    # --- Perturbación de variables numéricas -------------------------------
    desplazamientos_de_media = {
        "puntaje": -0.8,
        "salario_cliente": 0.6,
    }

    for columna in COLUMNAS_NUMERICAS:
        if columna not in df_actual.columns:
            continue
        desviacion = df_historico[columna].std()
        if not np.isfinite(desviacion) or desviacion == 0:
            continue

        ruido = rng.normal(loc=0.0, scale=0.10 * desviacion, size=len(df_actual))
        desplazamiento = desplazamientos_de_media.get(columna, 0.0) * desviacion
        df_actual[columna] = df_actual[columna] + ruido + desplazamiento

    # --- Perturbación de variables categóricas -----------------------------
    # Remuestreamos cada categórica con probabilidades distintas a las del
    # histórico: la categoría más frecuente pierde peso y las demás lo ganan.
    for columna in COLUMNAS_CATEGORICAS:
        if columna not in df_actual.columns:
            continue
        proporciones = df_historico[columna].value_counts(normalize=True, dropna=False)
        if len(proporciones) < 2:
            continue

        pesos = proporciones.to_numpy(dtype=float).copy()
        pesos[0] = pesos[0] * 0.55          # la categoría dominante pierde peso
        pesos[1:] = pesos[1:] * 1.80        # el resto gana peso
        pesos = pesos / pesos.sum()

        df_actual[columna] = rng.choice(
            proporciones.index.to_numpy(), size=len(df_actual), p=pesos
        )

    print(
        f"Batch actual simulado: {df_actual.shape[0]:,} filas "
        f"(muestra del {frac:.0%} del histórico + perturbación controlada)"
    )
    return df_actual


# ---------------------------------------------------------------------------
# 3. MÉTRICAS DE DRIFT
# ---------------------------------------------------------------------------
def calcular_ks(serie_historica: pd.Series, serie_actual: pd.Series) -> tuple[float, float]:
    """Calcula la prueba de Kolmogorov-Smirnov entre dos muestras numéricas.

    KS mide la máxima distancia vertical entre las dos funciones de distribución
    acumulada. Entre más grande el estadístico, más se separaron las curvas.

    Args:
        serie_historica: Valores de la variable en el histórico.
        serie_actual: Valores de la variable en el lote actual.

    Returns:
        Tupla (estadístico, p_value). Hay drift si el estadístico supera KS_UMBRAL.
    """
    historico = pd.to_numeric(serie_historica, errors="coerce").dropna()
    actual = pd.to_numeric(serie_actual, errors="coerce").dropna()

    if len(historico) < 2 or len(actual) < 2:
        return float("nan"), float("nan")

    resultado = ks_2samp(historico, actual)
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_psi(
    serie_historica: pd.Series,
    serie_actual: pd.Series,
    bins: int = 10,
) -> float:
    """Calcula el Population Stability Index (PSI) entre dos muestras numéricas.

    El PSI parte el histórico en bins por cuantiles y compara qué porcentaje de
    la población cae en cada bin en cada período:

        PSI = suma( (%actual - %historico) * ln(%actual / %historico) )

    Lectura habitual en la industria:
        PSI < 0.10  -> población estable
        0.10 - 0.25 -> cambio moderado, vale la pena vigilar
        PSI > 0.25  -> cambio significativo (nuestro umbral de alerta)

    Args:
        serie_historica: Valores de la variable en el histórico.
        serie_actual: Valores de la variable en el lote actual.
        bins: Número de cuantiles con los que se parte el histórico.

    Returns:
        El valor de PSI. Hay drift si supera PSI_UMBRAL.
    """
    historico = pd.to_numeric(serie_historica, errors="coerce").dropna()
    actual = pd.to_numeric(serie_actual, errors="coerce").dropna()

    if len(historico) < 2 or len(actual) < 2:
        return float("nan")

    # Los cortes se definen SOBRE EL HISTÓRICO: la referencia manda.
    cortes = np.unique(np.quantile(historico, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:
        return 0.0

    # Abrimos los extremos para que ningún valor nuevo se quede por fuera.
    cortes[0] = -np.inf
    cortes[-1] = np.inf

    prop_historico = np.histogram(historico, bins=cortes)[0] / len(historico)
    prop_actual = np.histogram(actual, bins=cortes)[0] / len(actual)

    # Reemplazamos ceros por un valor mínimo para evitar divisiones por cero
    # y logaritmos infinitos cuando un bin queda vacío.
    epsilon = 1e-6
    prop_historico = np.where(prop_historico == 0, epsilon, prop_historico)
    prop_actual = np.where(prop_actual == 0, epsilon, prop_actual)

    psi = np.sum((prop_actual - prop_historico) * np.log(prop_actual / prop_historico))
    return float(psi)


def calcular_jensen_shannon(
    serie_historica: pd.Series,
    serie_actual: pd.Series,
    bins: int = 10,
) -> float:
    """Calcula la distancia de Jensen-Shannon entre dos distribuciones.

    Sirve tanto para numéricas como para categóricas:
        - Numéricas: se discretizan en `bins` usando cuantiles del histórico.
        - Categóricas: cada categoría es un bin.

    En ambos casos se normalizan las frecuencias a probabilidades y se usa
    `scipy.spatial.distance.jensenshannon`, que devuelve un valor acotado entre
    0 (distribuciones idénticas) y 1 (sin traslape).

    Args:
        serie_historica: Valores de la variable en el histórico.
        serie_actual: Valores de la variable en el lote actual.
        bins: Número de bins a usar si la variable es numérica.

    Returns:
        La distancia de Jensen-Shannon. Hay drift si supera JS_UMBRAL.
    """
    es_numerica = pd.api.types.is_numeric_dtype(
        serie_historica
    ) and pd.api.types.is_numeric_dtype(serie_actual)

    if es_numerica:
        historico = pd.to_numeric(serie_historica, errors="coerce").dropna()
        actual = pd.to_numeric(serie_actual, errors="coerce").dropna()
        if len(historico) < 2 or len(actual) < 2:
            return float("nan")

        cortes = np.unique(np.quantile(historico, np.linspace(0, 1, bins + 1)))
        if len(cortes) < 3:
            return 0.0
        cortes[0] = -np.inf
        cortes[-1] = np.inf

        conteo_historico = np.histogram(historico, bins=cortes)[0]
        conteo_actual = np.histogram(actual, bins=cortes)[0]
    else:
        historico = _normalizar_categorica(serie_historica)
        actual = _normalizar_categorica(serie_actual)
        # Alineamos ambas series al mismo universo de categorías; si una
        # categoría no aparece en un período, cuenta como 0.
        categorias = sorted(set(historico.unique()) | set(actual.unique()))
        conteo_historico = historico.value_counts().reindex(categorias, fill_value=0).to_numpy()
        conteo_actual = actual.value_counts().reindex(categorias, fill_value=0).to_numpy()

    total_historico = conteo_historico.sum()
    total_actual = conteo_actual.sum()
    if total_historico == 0 or total_actual == 0:
        return float("nan")

    prob_historico = conteo_historico / total_historico
    prob_actual = conteo_actual / total_actual

    distancia = jensenshannon(prob_historico, prob_actual, base=2)
    return float(distancia) if np.isfinite(distancia) else 0.0


def calcular_chi_cuadrado(
    serie_historica: pd.Series,
    serie_actual: pd.Series,
) -> tuple[float, float]:
    """Aplica la prueba de chi-cuadrado de independencia sobre una variable categórica.

    Se arma una tabla de contingencia de 2 filas (histórico vs actual) por N
    columnas (una por categoría) y se contrasta la hipótesis nula de que la
    distribución de categorías es la misma en ambos períodos.

    Args:
        serie_historica: Valores de la variable en el histórico.
        serie_actual: Valores de la variable en el lote actual.

    Returns:
        Tupla (estadístico, p_value). Hay drift si el p_value es menor que
        CHI2_PVALUE_UMBRAL, es decir, si se rechaza la hipótesis de "misma
        distribución".
    """
    historico = _normalizar_categorica(serie_historica)
    actual = _normalizar_categorica(serie_actual)

    categorias = sorted(set(historico.unique()) | set(actual.unique()))
    if len(categorias) < 2:
        return float("nan"), float("nan")

    tabla = np.vstack(
        [
            historico.value_counts().reindex(categorias, fill_value=0).to_numpy(),
            actual.value_counts().reindex(categorias, fill_value=0).to_numpy(),
        ]
    )

    # Descartamos categorías que quedaron en cero en los dos períodos: aportan
    # una columna vacía que chi2_contingency no puede procesar.
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2:
        return float("nan"), float("nan")

    estadistico, p_value, _, _ = chi2_contingency(tabla)
    return float(estadistico), float(p_value)


def _normalizar_categorica(serie: pd.Series) -> pd.Series:
    """Deja una serie categórica lista para comparar entre períodos.

    Hace tres cosas: convierte todo a texto (para que `tipo_credito`, que viene
    como número, se trate como categoría), marca los nulos como una categoría
    propia llamada 'SIN_DATO' —porque que un campo deje de llegar también es
    drift— y agrupa en 'OTROS' las categorías con frecuencia menor a
    FRECUENCIA_MINIMA_CATEGORIA.

    Args:
        serie: Serie categórica cruda.

    Returns:
        Serie de texto, sin nulos y con las categorías raras agrupadas.
    """
    normalizada = serie.astype("object").where(serie.notna(), "SIN_DATO").astype(str)
    proporciones = normalizada.value_counts(normalize=True)
    categorias_raras = proporciones[proporciones < FRECUENCIA_MINIMA_CATEGORIA].index
    return normalizada.where(~normalizada.isin(categorias_raras), "OTROS")


# ---------------------------------------------------------------------------
# 4. EVALUACIÓN POR VARIABLE
# ---------------------------------------------------------------------------
def evaluar_variable(
    nombre_columna: str,
    serie_historica: pd.Series,
    serie_actual: pd.Series,
    es_categorica: bool,
) -> dict:
    """Evalúa el drift de una sola variable aplicando las métricas que le corresponden.

    A una variable numérica se le aplican KS, PSI y Jensen-Shannon. A una
    categórica se le aplican chi-cuadrado y Jensen-Shannon. Basta con que una
    métrica supere su umbral para marcar la variable con drift.

    Args:
        nombre_columna: Nombre de la variable evaluada.
        serie_historica: Valores en el histórico.
        serie_actual: Valores en el lote actual.
        es_categorica: True si la variable debe tratarse como categórica.

    Returns:
        Diccionario con el nombre, el tipo, cada métrica calculada y la bandera
        `drift_detectado`.
    """
    resultado: dict = {
        "variable": nombre_columna,
        "tipo": "categorica" if es_categorica else "numerica",
        "n_historico": int(serie_historica.notna().sum()),
        "n_actual": int(serie_actual.notna().sum()),
        "ks_estadistico": None,
        "ks_pvalue": None,
        "psi": None,
        "jensen_shannon": None,
        "chi2_estadistico": None,
        "chi2_pvalue": None,
    }

    senales: list[bool] = []

    if es_categorica:
        chi2_estadistico, chi2_pvalue = calcular_chi_cuadrado(serie_historica, serie_actual)
        js = calcular_jensen_shannon(serie_historica, serie_actual)

        resultado["chi2_estadistico"] = _limpiar_nan(chi2_estadistico)
        resultado["chi2_pvalue"] = _limpiar_nan(chi2_pvalue)
        resultado["jensen_shannon"] = _limpiar_nan(js)

        if np.isfinite(chi2_pvalue):
            senales.append(chi2_pvalue < CHI2_PVALUE_UMBRAL)
        if np.isfinite(js):
            senales.append(js > JS_UMBRAL)
        resultado["metrica_principal"] = "jensen_shannon"
    else:
        ks_estadistico, ks_pvalue = calcular_ks(serie_historica, serie_actual)
        psi = calcular_psi(serie_historica, serie_actual)
        js = calcular_jensen_shannon(serie_historica, serie_actual)

        resultado["ks_estadistico"] = _limpiar_nan(ks_estadistico)
        resultado["ks_pvalue"] = _limpiar_nan(ks_pvalue)
        resultado["psi"] = _limpiar_nan(psi)
        resultado["jensen_shannon"] = _limpiar_nan(js)

        if np.isfinite(ks_estadistico):
            senales.append(ks_estadistico > KS_UMBRAL)
        if np.isfinite(psi):
            senales.append(psi > PSI_UMBRAL)
        if np.isfinite(js):
            senales.append(js > JS_UMBRAL)
        resultado["metrica_principal"] = "psi"

    resultado["drift_detectado"] = bool(any(senales))
    return resultado


def _limpiar_nan(valor: float) -> Optional[float]:
    """Convierte NaN e infinitos en None para que el JSON quede válido.

    Args:
        valor: Número a limpiar.

    Returns:
        El mismo número redondeado a 6 decimales, o None si no es finito.
    """
    if valor is None or not np.isfinite(valor):
        return None
    return round(float(valor), 6)


# ---------------------------------------------------------------------------
# 5. ORQUESTADOR
# ---------------------------------------------------------------------------
def ejecutar_monitoreo(
    path_datos: str = "Base_de_datos.csv",
    columnas_categoricas: Optional[list[str]] = None,
    columnas_numericas: Optional[list[str]] = None,
) -> dict:
    """Ejecuta el ciclo completo de monitoreo y guarda los resultados en disco.

    En producción esta función no se corre a mano: se programa con una
    periodicidad definida (por ejemplo, semanal, mediante un scheduler tipo
    cron, Airflow o un job de Jenkins) para que cada semana quede una fila nueva
    en el historial y el equipo pueda ver la tendencia sin esperar a que el
    modelo falle.

    Pasos que ejecuta:
        1. Carga el histórico de referencia.
        2. Genera el lote actual (hoy simulado, ver `generar_batch_actual`).
        3. Evalúa cada variable con las métricas que le corresponden.
        4. Arma el reporte con timestamp y la bandera global de drift.
        5. Guarda `drift_report.json` (sobrescribe) y agrega una fila a
           `drift_history.csv` (lo crea con encabezados si no existe).

    Args:
        path_datos: Ruta al CSV histórico.
        columnas_categoricas: Variables a tratar como categóricas. Si es None,
            usa COLUMNAS_CATEGORICAS.
        columnas_numericas: Variables a tratar como numéricas. Si es None, usa
            COLUMNAS_NUMERICAS.

    Returns:
        El reporte completo de la corrida como diccionario.
    """
    columnas_categoricas = (
        COLUMNAS_CATEGORICAS if columnas_categoricas is None else columnas_categoricas
    )
    columnas_numericas = (
        COLUMNAS_NUMERICAS if columnas_numericas is None else columnas_numericas
    )

    df_historico = cargar_datos_historicos(path_datos)
    df_actual = generar_batch_actual(df_historico)

    resultados: list[dict] = []

    for columna in columnas_numericas:
        if columna not in df_historico.columns:
            print(f"Aviso: la columna numérica '{columna}' no está en el dataset. La omito.")
            continue
        resultados.append(
            evaluar_variable(columna, df_historico[columna], df_actual[columna], es_categorica=False)
        )

    for columna in columnas_categoricas:
        if columna not in df_historico.columns:
            print(f"Aviso: la columna categórica '{columna}' no está en el dataset. La omito.")
            continue
        resultados.append(
            evaluar_variable(columna, df_historico[columna], df_actual[columna], es_categorica=True)
        )

    variables_con_drift = [r["variable"] for r in resultados if r["drift_detectado"]]

    reporte = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "archivo_datos": str(_resolver_ruta(path_datos)),
        "n_historico": int(len(df_historico)),
        "n_actual": int(len(df_actual)),
        "batch_actual_simulado": True,
        "umbrales": {
            "PSI_UMBRAL": PSI_UMBRAL,
            "KS_UMBRAL": KS_UMBRAL,
            "JS_UMBRAL": JS_UMBRAL,
            "CHI2_PVALUE_UMBRAL": CHI2_PVALUE_UMBRAL,
        },
        "variables_evaluadas": len(resultados),
        "variables_con_drift": variables_con_drift,
        "n_variables_con_drift": len(variables_con_drift),
        "drift_detectado_global": bool(variables_con_drift),
        "resultados": resultados,
    }

    _guardar_reporte(reporte)
    _agregar_a_historial(reporte)

    return reporte


def _guardar_reporte(reporte: dict, archivo: str = ARCHIVO_REPORTE) -> Path:
    """Guarda el reporte de la corrida en un JSON, sobrescribiendo el anterior.

    Args:
        reporte: Reporte devuelto por `ejecutar_monitoreo`.
        archivo: Nombre del archivo de salida.

    Returns:
        La ruta donde quedó guardado el reporte.
    """
    ruta = _resolver_ruta(archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en: {ruta}")
    return ruta


def _agregar_a_historial(reporte: dict, archivo: str = ARCHIVO_HISTORIAL) -> Path:
    """Agrega una fila por variable al historial acumulado de corridas.

    Si el archivo no existe lo crea con encabezados; si ya existe, agrega las
    filas al final sin repetir los encabezados. Este historial es el insumo del
    análisis temporal de la app de monitoreo.

    Args:
        reporte: Reporte devuelto por `ejecutar_monitoreo`.
        archivo: Nombre del archivo de historial.

    Returns:
        La ruta donde quedó guardado el historial.
    """
    ruta = _resolver_ruta(archivo)

    filas = []
    for resultado in reporte["resultados"]:
        filas.append(
            {
                "timestamp": reporte["timestamp"],
                "variable": resultado["variable"],
                "tipo": resultado["tipo"],
                "metrica_principal": resultado["metrica_principal"],
                "ks_estadistico": resultado["ks_estadistico"],
                "ks_pvalue": resultado["ks_pvalue"],
                "psi": resultado["psi"],
                "jensen_shannon": resultado["jensen_shannon"],
                "chi2_estadistico": resultado["chi2_estadistico"],
                "chi2_pvalue": resultado["chi2_pvalue"],
                "drift_detectado": resultado["drift_detectado"],
                "drift_detectado_global": reporte["drift_detectado_global"],
            }
        )

    df_nuevo = pd.DataFrame(filas)
    existe = ruta.exists()
    df_nuevo.to_csv(ruta, mode="a" if existe else "w", header=not existe, index=False)

    print(f"Historial actualizado en: {ruta} ({len(filas)} filas agregadas)")
    return ruta


# ---------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
def _imprimir_resumen(reporte: dict) -> None:
    """Imprime en consola un resumen legible de la corrida.

    Args:
        reporte: Reporte devuelto por `ejecutar_monitoreo`.
    """
    print("\n" + "=" * 78)
    print("REPORTE DE MONITOREO DE DATA DRIFT")
    print("=" * 78)
    print(f"Fecha de la corrida : {reporte['timestamp']}")
    print(f"Filas históricas    : {reporte['n_historico']:,}")
    print(f"Filas del batch     : {reporte['n_actual']:,} (simulado)")
    print(
        f"Umbrales            : PSI>{PSI_UMBRAL} | KS>{KS_UMBRAL} | "
        f"JS>{JS_UMBRAL} | chi2 p<{CHI2_PVALUE_UMBRAL}"
    )
    print("-" * 78)
    print(f"{'Variable':<32}{'Tipo':<12}{'PSI':>9}{'KS':>9}{'JS':>9}{'Drift':>8}")
    print("-" * 78)

    for resultado in reporte["resultados"]:
        psi = resultado["psi"]
        ks = resultado["ks_estadistico"]
        js = resultado["jensen_shannon"]
        print(
            f"{resultado['variable']:<32}"
            f"{resultado['tipo']:<12}"
            f"{(f'{psi:.4f}' if psi is not None else '-'):>9}"
            f"{(f'{ks:.4f}' if ks is not None else '-'):>9}"
            f"{(f'{js:.4f}' if js is not None else '-'):>9}"
            f"{('SÍ' if resultado['drift_detectado'] else 'no'):>8}"
        )

    print("-" * 78)
    if reporte["drift_detectado_global"]:
        print(
            f"RESULTADO: se detectó drift en {reporte['n_variables_con_drift']} "
            f"de {reporte['variables_evaluadas']} variables."
        )
        print("Variables afectadas: " + ", ".join(reporte["variables_con_drift"]))
        print("Recomendación: revisa esas variables y evalúa un reentrenamiento del modelo.")
    else:
        print("RESULTADO: no se detectó drift. Las distribuciones se mantienen estables.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    reporte_actual = ejecutar_monitoreo()
    _imprimir_resumen(reporte_actual)
