# 🎓 Proyecto Integrador: Modelo de Riesgo Crediticio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Licencia MIT](https://img.shields.io/badge/Licencia-MIT-green)]()
[![Estado: En Desarrollo](https://img.shields.io/badge/Estado-En%20Desarrollo-yellow)]()

## 📋 Descripción del Proyecto

Este proyecto implementa un **pipeline MLOps completo** para un modelo de predicción de riesgo crediticio.

**Caso de Negocio:** Crear un modelo de machine learning que prediga la probabilidad de que un cliente pague o no su crédito a tiempo, permitiendo a la entidad financiera tomar decisiones informadas en la aprobación de créditos.

---

## 📁 Estructura del Proyecto

```
henry-sm-modelo-riesgo/
│
├── mlops_pipeline/                    # 📦 Pipeline de ML (el corazón del proyecto)
│   └── src/                           # 📄 Código fuente
│       ├── Cargar_datos.ipynb         # 1️⃣ Cargar datos desde CSV
│       ├── comprension_eda.ipynb      # 2️⃣ Análisis Exploratorio de Datos
│       ├── ft_engineering.py          # 3️⃣ Feature Engineering
│       ├── model_training_evaluation.py # 4️⃣ Entrenar y evaluar modelos
│       ├── model_deploy.py            # 5️⃣ Desplegar modelo
│       ├── model_monitoring.py        # 6️⃣ Monitorear data drift en producción
│       └── app_monitoreo.py           # 7️⃣ App de Streamlit para visualizar el drift
│
├── Base_de_datos.csv                  # 📊 Dataset principal (10,763 registros)
├── drift_report.json                  # 📈 Reporte de la última corrida de monitoreo
├── drift_history.csv                  # 🕒 Historial acumulado de corridas
├── Base_de_datos.xlsx                 # 📑 Datos originales en Excel
├── requirements.txt                   # 📦 Dependencias de Python
├── .gitignore                         # 🚫 Archivos a ignorar en Git
└── README.md                          # 📖 Este archivo
```

---

## 🚀 Inicio Rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/santimerchan/henry-sm-modelo-riesgo.git
cd henry-sm-modelo-riesgo
```

### 2. Crear entorno virtual

```bash
# En macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar Jupyter Notebook

```bash
# Opción 1: Desde terminal
jupyter notebook

# Opción 2: Desde Cursor IDE (recomendado)
# - Abre el archivo: mlops_pipeline/src/Cargar_datos.ipynb
# - Selecciona el kernel de Python de tu venv
# - Ejecuta las celdas (Shift + Enter)
```

---

## 📊 Dataset

| Propiedad | Valor |
|-----------|-------|
| **Filas** | 10,763 registros |
| **Columnas** | 23 variables |
| **Variable Objetivo** | `Pago_atiempo` (Binaria: Sí/No) |
| **Formato** | CSV y XLSX |

### Columnas principales:

- `tipo_credito`: Tipo de crédito solicitado
- `capital_prestado`: Monto del préstamo
- `plazo_meses`: Duración del crédito
- `edad_cliente`: Edad del solicitante
- `salario_cliente`: Ingreso mensual
- `puntaje`: Score crediticio
- `Pago_atiempo`: **Target** - ¿Pagó a tiempo? (Sí/No)
- ... y 15 más

---

## 💼 Caso de negocio

La entidad financiera aprueba créditos todos los días y necesita saber, **antes de
desembolsar**, qué tan probable es que el cliente pague a tiempo. El modelo predice
la variable `Pago_atiempo` a partir del perfil del solicitante (capital pedido, plazo,
edad, salario, puntaje interno, puntaje de Datacrédito, obligaciones vigentes, etc.).

El impacto es directo en dos frentes:

- **Pérdida esperada:** aprobar clientes que no van a pagar se traduce en cartera vencida
  y en provisiones que golpean el estado de resultados.
- **Costo de oportunidad:** negar clientes buenos por un modelo mal calibrado significa
  dejar de colocar créditos rentables frente a la competencia.

El dataset está desbalanceado (10.252 pagos a tiempo contra 511 incumplimientos, cerca
del 4,7% de mora), así que el desempeño se evalúa con precision, recall, F1 y ROC-AUC,
nunca con accuracy sola.

---

## 🔍 ¿Por qué monitorear data drift?

Un modelo de riesgo se entrena sobre una fotografía del pasado, pero la realidad no se
queda quieta: cambia la política de originación, entra un segmento de clientes distinto,
se ajusta el scoring del buró, o simplemente el ciclo económico se mueve. Cuando la
distribución de las variables de entrada se aleja de la que el modelo vio en
entrenamiento, hablamos de **data drift**.

Lo peligroso del drift es que **no lanza un error**. El pipeline sigue corriendo, la API
sigue respondiendo, y el modelo sigue entregando probabilidades con toda confianza —
solo que cada vez peor. En un modelo de crédito eso se descubre meses después, cuando
la mora real ya subió y el daño está hecho.

Monitorear drift es el sistema de alarma temprana: nos avisa que las entradas cambiaron
**antes** de que el negocio sienta el deterioro, y nos permite decidir a tiempo si hay
que investigar la fuente de datos o reentrenar el modelo.

---

## 🏗️ Arquitectura del monitoreo

Todo el monitoreo vive en `mlops_pipeline/src/model_monitoring.py`. El script compara
dos distribuciones —el **histórico** (`Base_de_datos.csv`, lo que el modelo conoce)
contra un **batch actual** (lo que está llegando ahora)— variable por variable.

### Métricas y umbrales

Los umbrales están definidos como constantes en mayúsculas al inicio del script:

| Métrica | Constante | Umbral | Aplica a | Se marca drift si… |
|---------|-----------|--------|----------|--------------------|
| PSI (Population Stability Index) | `PSI_UMBRAL` | `0.25` | Numéricas | PSI > 0.25 |
| KS (Kolmogorov-Smirnov) | `KS_UMBRAL` | `0.25` | Numéricas | estadístico > 0.25 |
| Jensen-Shannon | `JS_UMBRAL` | `0.3` | Numéricas y categóricas | distancia > 0.3 |
| Chi-cuadrado | `CHI2_PVALUE_UMBRAL` | `0.05` | Categóricas | p-value < 0.05 |

Una variable numérica se evalúa con PSI, KS y Jensen-Shannon; una categórica con
chi-cuadrado y Jensen-Shannon. Basta con que **una** métrica supere su umbral para
marcar la variable con `drift_detectado = True`.

### Variables monitoreadas

- **Numéricas (11):** `capital_prestado`, `plazo_meses`, `edad_cliente`, `salario_cliente`,
  `total_otros_prestamos`, `cuota_pactada`, `puntaje`, `puntaje_datacredito`,
  `cant_creditosvigentes`, `saldo_total`, `promedio_ingresos_datacredito`.
- **Categóricas (3):** `tipo_laboral`, `tendencia_ingresos`, `tipo_credito`.

Quedan fuera `fecha_prestamo` (es una marca de tiempo, no una feature) y `Pago_atiempo`
(es el target, no un input del modelo).

### Funciones del script

| Función | Qué hace |
|---------|----------|
| `cargar_datos_historicos(path)` | Carga el CSV de referencia |
| `generar_batch_actual(df, frac, seed)` | Simula el lote actual con perturbación controlada |
| `calcular_ks(...)` | Prueba de Kolmogorov-Smirnov, devuelve (estadístico, p-value) |
| `calcular_psi(...)` | PSI manual con bins por cuantiles del histórico |
| `calcular_jensen_shannon(...)` | Distancia JS sobre distribuciones discretizadas |
| `calcular_chi_cuadrado(...)` | Chi-cuadrado sobre tabla de contingencia |
| `evaluar_variable(...)` | Aplica las métricas que le corresponden al tipo de variable |
| `ejecutar_monitoreo(...)` | Orquesta todo y escribe los archivos de salida |

### Salidas que produce cada corrida

- **`drift_report.json`** — fotografía de la última corrida (se sobrescribe cada vez).
- **`drift_history.csv`** — una fila por variable y por corrida; se crea con encabezados
  la primera vez y luego crece en modo *append*. Es el insumo del análisis temporal.

En producción, `ejecutar_monitoreo()` no se corre a mano: se programa con periodicidad
definida (por ejemplo, **semanal** vía cron, Airflow o un job de Jenkins) para que el
historial se llene solo y la tendencia sea visible sin esperar a que el modelo falle.

---

## ▶️ Cómo ejecutar el monitoreo

### 1. Script de monitoreo

```bash
source venv/bin/activate
python mlops_pipeline/src/model_monitoring.py
```

Imprime un resumen en consola y deja `drift_report.json` y `drift_history.csv` en la
raíz del proyecto.

### 2. App de visualización (Streamlit)

```bash
streamlit run mlops_pipeline/src/app_monitoreo.py
```

Se abre en `http://localhost:8501` con tres pestañas:

- **📈 Visualización de métricas** — selector de variable, comparación gráfica de
  histórico vs actual (histograma para numéricas, barras de proporción para categóricas),
  tabla completa de métricas y semáforo por variable (🟢 estable · 🟡 en vigilancia ·
  🔴 drift detectado).
- **🕒 Análisis temporal** — evolución de la métrica principal de cada variable a lo largo
  de las corridas registradas en `drift_history.csv`, con la tendencia calculada
  comparando el promedio de la primera mitad de las corridas contra el de la segunda.
- **🚨 Recomendaciones y alertas** — diagnóstico automático de la última corrida, qué
  variables revisar, si conviene reentrenar, y el listado de alertas con severidad
  (alta / media / ninguna).

> La app necesita que `model_monitoring.py` se haya ejecutado al menos una vez; si no
> encuentra `drift_report.json` te lo dice explícitamente en pantalla.

---

## 📉 Hallazgos de la última corrida

Corrida del **2026-09-09**, comparando 10.763 filas históricas contra un batch actual
de 3.229 filas. Resultado: **drift en 8 de 14 variables** (57%), severidad **alta**.

### Variables con drift

| Variable | Tipo | PSI | KS | Jensen-Shannon | Chi² p-value |
|----------|------|-----|-----|----------------|--------------|
| `puntaje` | numérica | **14.4900** | **0.9284** | **0.8738** | — |
| `salario_cliente` | numérica | **12.4723** | **0.9931** | **0.8706** | — |
| `total_otros_prestamos` | numérica | **2.7906** | **0.4528** | **0.6481** | — |
| `plazo_meses` | numérica | **0.3768** | 0.1593 | 0.2546 | — |
| `promedio_ingresos_datacredito` | numérica | **0.2846** | 0.1191 | 0.2195 | — |
| `tipo_laboral` | categórica | — | — | 0.2542 | **< 0.0001** |
| `tendencia_ingresos` | categórica | — | — | 0.2442 | **< 0.0001** |
| `tipo_credito` | categórica | — | — | 0.2349 | **< 0.0001** |

### Variables estables

| Variable | PSI | KS | Jensen-Shannon |
|----------|-----|-----|----------------|
| `capital_prestado` | 0.0053 | 0.0170 | 0.0310 |
| `edad_cliente` | 0.0062 | 0.0217 | 0.0334 |
| `cuota_pactada` | 0.0044 | 0.0223 | 0.0281 |
| `puntaje_datacredito` | 0.0056 | 0.0180 | 0.0318 |
| `cant_creditosvigentes` | 0.0270 | 0.0665 | 0.0696 |
| `saldo_total` | 0.1870 | 0.1756 | 0.1795 |

### Lectura de los resultados

- `puntaje` y `salario_cliente` son los casos más extremos, y era lo esperado: la
  simulación les aplica un desplazamiento deliberado de media (−0.8 y +0.6 desviaciones
  estándar respectivamente) precisamente para tener un ejemplo claro de drift severo.
- `plazo_meses` y `total_otros_prestamos` disparan PSI alto aunque su KS sea moderado.
  La razón es que son variables discretas concentradas en pocos valores: cualquier ruido
  las dispersa y el PSI, que compara proporciones por bin, lo castiga fuerte. Es un buen
  recordatorio de que **PSI y KS no siempre coinciden** y de que hay que mirar las dos.
- Las tres categóricas dan chi-cuadrado significativo (p-value prácticamente 0) pero
  Jensen-Shannon por debajo de 0.3. Con muestras grandes el chi-cuadrado detecta
  diferencias muy pequeñas; JS mide la magnitud real del cambio. Aquí el chi-cuadrado
  dice "el cambio es real", y JS dice "pero es moderado".
- `saldo_total` (PSI 0.1870) queda en 🟡 vigilancia: no cruzó el umbral, pero está lo
  suficientemente cerca como para revisarlo en la próxima corrida.

**Recomendación:** con este nivel de drift correspondería investigar la fuente de
`puntaje` y `salario_cliente` antes que nada, y planificar un reentrenamiento con datos
recientes.

---

## ⚠️ Nota importante: el batch actual es simulado

Todavía **no existe un log real de predicciones en producción**, así que el "batch
actual" se construye tomando una muestra del 30% del histórico y aplicándole una
perturbación controlada y reproducible (`seed=42`):

- **Numéricas:** ruido gaussiano con desviación igual al 10% de la desviación estándar
  de cada variable, más un desplazamiento intencional de media en `puntaje` y
  `salario_cliente` para que el drift sea visible.
- **Categóricas:** se reponderan las proporciones (la categoría dominante pierde peso y
  las demás lo ganan).

La función `generar_batch_actual()` lleva un `# TODO` explícito arriba: debe reemplazarse
por el log real de predicciones (inputs + predicción del modelo) en cuanto ese pipeline
exista. **La lógica de las métricas, los umbrales y toda la infraestructura de reporte
son las definitivas** — lo único provisional es de dónde salen los datos del batch actual.

---

## 🔀 Flujo de ramas y estado de este avance

```
developer  ──PR #1──►  certification  ──PR #2──►  main
(desarrollo)           (pruebas/QA)              (producción)
```

- **`developer`** — donde se escribe el código. Aquí se hizo el commit de este avance.
- **`certification`** — ambiente de pruebas. Nada llega aquí sin pasar por un Pull Request
  revisado desde `developer`.
- **`main`** — producción. Solo recibe lo que ya fue certificado.

**Este avance corresponde al primer Pull Request: `developer` → `certification`.**
No se abre PR hacia `main` en este punto; la promoción a producción es un paso posterior
y separado, que ocurre solo después de que el código haya sido validado en certification.

---

## 🎯 Fases del Proyecto

### ✅ Fase 1: Exploración (V1.0.0 - V1.1.0)

| Versión | Hito | Estado |
|---------|------|--------|
| **V1.0.0** | Estructura base + setup | ✅ |
| **V1.0.1** | Cargar datos | ⏳ En Progreso |
| **V1.1.0** | EDA (Análisis exploratorio) | ⏳ En Progreso |

### 🔄 Fase 2: Entrenamiento (V1.1.1 - V2.0.0)

| Versión | Hito | Estado |
|---------|------|--------|
| **V1.1.1** | Feature Engineering | ⏳ Por hacer |
| **V1.2.0** | Entrenar modelos | ⏳ Por hacer |
| **V2.0.0** | Monitoreo de drift + app Streamlit | ✅ |

### 🚀 Fase 3: Producción (V2.1.0+)

| Versión | Hito | Estado |
|---------|------|--------|
| **V2.1.0** | FastAPI + Docker | ⏳ Por hacer |
| **V2.1.1** | CI/CD + Validación | ⏳ Por hacer |

---

## 🌳 Gestión de Ramas (Git Workflow)

Este proyecto usa un **Git flow profesional** con 3 ramas principales:

```
main (producción)
  ↑
  ├─ certification (staging/testing)
  │   ↑
  │   └─ developer (desarrollo)
```

### Rama `main` 🟦
- **Contenido:** Código estable en producción
- **Quién puede hacer push:** Solo después de aprobación
- **Versiones:** Tags semánticos (v1.0.0, v1.0.1, etc.)

### Rama `certification` 🟨
- **Contenido:** Código probado, listo para producción
- **Quién puede hacer push:** Después de pasar pruebas
- **Propósito:** Testing/staging

### Rama `developer` 🟩
- **Contenido:** Código en desarrollo
- **Quién puede hacer push:** Todos los desarrolladores
- **Propósito:** Integración continua

---

## 📝 Convenciones de Código

### Mensajes de Commit

Usa el formato: `<tipo>: <descripción breve>`

```bash
# ✅ Bien
git commit -m "feat: Agregar análisis bivariable en EDA"

# ✅ Bien
git commit -m "fix: Corregir valor nulo en columna edad"

# ❌ Mal
git commit -m "arregle cosas"
```

### Tipos de commit:
- `feat`: Nueva característica
- `fix`: Corrección de bugs
- `docs`: Documentación
- `refactor`: Refactorización de código
- `test`: Pruebas
- `chore`: Cambios en configuración

---

## 🛠️ Tecnologías Utilizadas

| Área | Herramientas |
|------|--------------|
| **Lenguaje** | Python 3.10+ |
| **Datos** | pandas, numpy |
| **ML/AI** | scikit-learn, xgboost |
| **Notebooks** | Jupyter, JupyterLab |
| **Visualización** | seaborn, matplotlib |
| **Monitoreo** | scipy (KS, chi², Jensen-Shannon), Streamlit |
| **MLOps** | (próximamente) |
| **API** | FastAPI, uvicorn |
| **Contenedores** | Docker |

---

## 📚 Documentación

- [GUIA_DESARROLLO_PI.md](./GUIA_DESARROLLO_PI.md) — Guía completa para instructores (comando por comando)
- [requirements.txt](./requirements.txt) — Todas las dependencias

---

## 👨‍💻 Autor

**Estudiante:** Santiago Merchan  
**Bootcamp:** Henry - Módulo 5 (MLOps)  
**Email:** santiago@cotidiania.co

---

## 📄 Licencia

MIT License - Ver detalles en el repositorio

---

## 🤝 Contribuciones

Este es un proyecto educativo. Para cambios:

1. Crea una rama: `git checkout -b feature/tu-feature`
2. Hace cambios y commits
3. Push a tu rama: `git push origin feature/tu-feature`
4. Abre un Pull Request

---

**Última actualización:** agosto 2026  
**Versión del documento:** V1.0.0
