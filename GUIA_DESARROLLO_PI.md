# 📚 GUÍA COMPLETA: DESARROLLO DEL PROYECTO INTEGRADOR (PI)
## Modelo de Riesgo Crediticio - Rama por Rama, Versión por Versión

**Documento para Instructores:** Esta guía detalla cada comando, cada paso, y cada decisión arquitectónica necesaria para enseñar a los estudiantes cómo construir un proyecto MLOps profesional.

---

## 📋 TABLA DE CONTENIDOS

1. [Fase 0: Configuración Inicial](#fase-0-configuración-inicial)
2. [Fase 1: V1.0.0 - Estructura de Carpetas](#fase-1-v100---estructura-de-carpetas)
3. [Fase 2: V1.0.1 - Carga de Datos](#fase-2-v101---carga-de-datos)
4. [Fase 3: V1.1.0 - Exploración y Análisis de Datos (EDA)](#fase-3-v110---exploración-y-análisis-de-datos-eda)
5. [Comandos Git: Branching y Merging](#comandos-git-branching-y-merging)
6. [Flujo de Trabajo Completo (Resumen)](#flujo-de-trabajo-completo-resumen)

---

## FASE 0: Configuración Inicial

### 🎯 Objetivo
Preparar el repositorio con las tres ramas requeridas y la estructura base necesaria.

### 0.1 Verificar que estamos en la rama main

```bash
cd /Users/smerchan/Desktop/Personal/Henry/DS/PI
git status
git branch -a
```

**Salida esperada:**
```
* main
  remotes/origin/main
```

**¿Qué significa?**
- El `*` indica que estamos en la rama `main`
- Es la rama predeterminada y la más importante (será nuestro "release" final)

### 0.2 Crear la rama `developer` (rama de desarrollo)

```bash
git branch developer
```

**Explicación pedagógica:**
- **main:** rama de producción (código estable, versiones finales)
- **developer:** rama de desarrollo (donde trabajamos nosotros, integramos features)
- **certification:** rama intermedia (código que pasó pruebas, listo para producción)

### 0.3 Crear la rama `certification` (rama de certificación/staging)

```bash
git branch certification
```

### 0.4 Verificar que se crearon las ramas

```bash
git branch -a
```

**Salida esperada:**
```
  certification
  developer
* main
  remotes/origin/main
```

### 0.5 Cambiar a la rama `developer` (donde trabajaremos)

```bash
git checkout developer
```

O con la sintaxis moderna (Git 2.23+):
```bash
git switch developer
```

**Verificación:**
```bash
git status
```

Debe mostrar: `On branch developer`

### 0.6 Subir las ramas al repositorio remoto (GitHub)

```bash
git push -u origin developer
git push -u origin certification
```

**¿Qué hace `-u`?**
- `-u` = `--set-upstream`
- Vincula tu rama local con la rama remota
- La próxima vez que hagas `git push`, no necesitarás especificar `origin developer`

---

## FASE 1: V1.0.0 - Estructura de Carpetas

### 🎯 Objetivo
Crear la estructura de carpetas **exacta** que requiere Jenkins y la consigna. Esta estructura debe ser idéntica en las tres ramas.

### 1.1 Crear la estructura de directorios

**¡CRÍTICO!** Esta estructura es **rigurosamente específica** y NO debe cambiar.

```bash
# Desde el directorio raíz del repo (PI/)
mkdir -p mlops_pipeline/src
```

**Qué creamos:**
- `mlops_pipeline/` — carpeta principal de la pipeline
- `mlops_pipeline/src/` — carpeta de código fuente (source)

### 1.2 Crear archivos de soporte a nivel raíz

```bash
# Crear los archivos en la raíz del repo
touch README.md
touch requirements.txt
touch .gitignore
```

**Archivos creados:**
| Archivo | Propósito |
|---------|-----------|
| `README.md` | Documentación del proyecto (caso de negocio, instrucciones) |
| `requirements.txt` | Dependencias de Python (pip packages) |
| `.gitignore` | Archivos/carpetas a ignorar en Git |

### 1.3 Crear notebooks stub en `mlops_pipeline/src/`

```bash
# Crear los 4 notebooks que usaremos inicialmente
touch mlops_pipeline/src/Cargar_datos.ipynb
touch mlops_pipeline/src/comprension_eda.ipynb
```

**Aún no crearemos:**
- `ft_engineering.py` (Feature Engineering - próxima fase)
- `model_training_evaluation.py` (Entrenamiento - próxima fase)
- `model_deploy.py` (Despliegue - fase posterior)
- `model_monitoring.py` (Monitoreo - fase posterior)

### 1.4 Crear `.gitignore` con patrones estándar

```bash
cat > .gitignore << 'EOF'
# Archivos del sistema
.DS_Store
Thumbs.db
*.log

# Entornos virtuales
venv/
env/
ENV/
.venv

# Caché de Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Datos (grandes, no deberían estar en Git)
*.csv
*.xlsx
*.xls
data/
output/

# MLflow / Experimentos
mlruns/
.mlflow/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.env
.env.local
EOF
```

**Explicación de cada sección:**

| Sección | Archivos Ignorados | Por Qué |
|---------|-------------------|--------|
| Sistema | `.DS_Store`, `Thumbs.db` | Archivos del SO que no queremos versionar |
| Python | `__pycache__/`, `.pyc` | Caché compilada, ocupa espacio innecesario |
| Jupyter | `.ipynb_checkpoints/` | Puntos de restauración que Jupyter crea automáticamente |
| Datos | `*.csv`, `*.xlsx` | **IMPORTANTE:** datos sensibles no deben estar en GitHub |
| MLflow | `mlruns/` | Artefactos de experimentos (se regeneran) |
| IDE | `.vscode/`, `.idea/` | Configuración personal de editores |

### 1.5 Crear `requirements.txt` inicial

```bash
cat > requirements.txt << 'EOF'
# Data manipulation and analysis
pandas==2.1.4
numpy==1.26.4

# Jupyter
jupyter==1.0.0
jupyterlab==4.1.0

# Machine Learning & Data Science
scikit-learn==1.4.2
pycaret==3.3.2

# MLOps & Experiment Tracking
mlflow==2.16.2

# Visualization
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.18.0

# Utilities
python-dotenv==1.0.0
EOF
```

**¿Por qué estos packages?**

| Package | Versión | Uso |
|---------|---------|-----|
| `pandas` | 2.1.4 | Manipulación de datos |
| `numpy` | 1.26.4 | Operaciones numéricas |
| `jupyter` | 1.0.0 | Ejecutar notebooks |
| `scikit-learn` | 1.4.2 | ML algorithms |
| `pycaret` | 3.3.2 | Automatización de ML |
| `mlflow` | 2.16.2 | Experiment tracking |
| `matplotlib/seaborn` | 3.8.4 / 0.13.2 | Visualización |
| `plotly` | 5.18.0 | Gráficos interactivos |

### 1.6 Crear `README.md` base

```bash
cat > README.md << 'EOF'
# Modelo de Riesgo Crediticio - MLOps Pipeline

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline MLOps completo para un modelo de predicción de riesgo crediticio.

**Caso de Negocio:** 
Reducir el riesgo de pérdida crediticia mediante la predicción temprana de posibles defaults.

## 📁 Estructura del Proyecto

```
.
├── mlops_pipeline/          # Pipeline de ML
│   └── src/                 # Código fuente
│       ├── Cargar_datos.ipynb
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.py
│       ├── model_training_evaluation.py
│       ├── model_deploy.py
│       └── model_monitoring.py
├── Base_de_datos.csv        # Dataset
├── requirements.txt         # Dependencias
├── .gitignore              # Archivos ignorados
└── README.md               # Este archivo
```

## 🚀 Instalación

1. Clonar repositorio
2. Crear entorno virtual: `python -m venv venv`
3. Activar: `source venv/bin/activate` (macOS/Linux)
4. Instalar: `pip install -r requirements.txt`

## 📚 Fases del Proyecto

### V1.0.0 - Estructura Base ✅
- Carpetas organizadas
- requirements.txt
- .gitignore

### V1.0.1 - Carga de Datos (en progreso)
- Cargar datos desde CSV
- Validación básica

### V1.1.0 - Exploración de Datos (próximo)
- Análisis Exploratorio de Datos (EDA)
- Visualizaciones

## 👥 Equipo

Estudiantes del Bootcamp Henry - Módulo 5 (MLOps)

## 📅 Ramas

- `main` — Versiones stable (releases)
- `developer` — Desarrollo activo
- `certification` — Pre-producción (testing)
EOF
```

### 1.7 Verificar estructura creada

```bash
tree -L 3 -a
```

O si `tree` no está disponible:
```bash
find . -type f -o -type d | grep -v .git | sort
```

**Estructura esperada:**
```
.
├── .git/
├── .gitignore
├── README.md
├── requirements.txt
└── mlops_pipeline/
    └── src/
        ├── Cargar_datos.ipynb
        └── comprension_eda.ipynb
```

### 1.8 Realizar primer commit (V1.0.0)

```bash
# Verificar qué archivos están sin stagear
git status

# Agregar todos los archivos nuevos al staging area
git add .

# Verificar nuevamente
git status

# Hacer commit con mensaje descriptivo
git commit -m "V1.0.0: Estructura base del proyecto MLOps

- Crear estructura de carpetas (mlops_pipeline/src/)
- Inicializar requirements.txt con dependencias base
- Crear .gitignore estándar para proyectos ML
- Crear README.md con documentación inicial

Rama: developer
Versión: V1.0.0"
```

**Explicación del mensaje de commit:**
- **Primera línea:** resumen breve (< 50 caracteres)
- **Línea en blanco:** separador
- **Viñetas:** cambios detallados
- **Metadatos:** rama y versión

### 1.9 Subir cambios a GitHub

```bash
git push origin developer
```

**Verificación en GitHub:**
1. Ve a https://github.com/santimerchan/henry-sm-modelo-riesgo
2. Cambia a la rama `developer` (dropdown arriba)
3. Deberías ver los archivos que acabas de subir

---

## FASE 2: V1.0.1 - Carga de Datos

### 🎯 Objetivo
Crear el notebook `Cargar_datos.ipynb` que cargue, valide y prepare el dataset inicial.

### 2.1 Nota sobre la estructura de datos

En producción real:
- Los datos vienen de un **Data Warehouse (DWH)** o **Datalake** (Snowflake, BigQuery, etc.)
- Habría conexiones SQL, APIs de datos, ETL pipelines

**Para este proyecto (educativo):**
- Usamos un CSV local (`Base_de_datos.csv`)
- El notebook simula cómo se cargarían datos en producción

### 2.2 Asegúrate que tienes el archivo Base_de_datos.csv

```bash
# Verificar si existe
ls -lh Base_de_datos.csv

# Si no existe, el usuario lo copiará manualmente
```

**El archivo debe estar en la raíz del repo**
```
/Users/smerchan/Desktop/Personal/Henry/DS/PI/Base_de_datos.csv
```

### 2.3 Crear el notebook `Cargar_datos.ipynb`

Como los notebooks `.ipynb` son JSON, usaremos una estructura estándar. Puedes crearla de varias formas:

#### Opción A: Crear el notebook manualmente en Jupyter

```bash
# Activar entorno virtual
source venv/bin/activate

# Iniciar Jupyter
jupyter notebook

# 1. Crear nuevo notebook: New > Python 3
# 2. Guardar como: mlops_pipeline/src/Cargar_datos.ipynb
# 3. Escribir las celdas (ver sección 2.4)
```

#### Opción B: Crear el notebook con nbformat (programático)

```python
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

# Celda 1: Título
nb.cells.append(new_markdown_cell("""
# Cargar Datos - Proyecto Riesgo Crediticio

**Objetivo:** Cargar y validar el dataset de riesgo crediticio desde CSV.

**Nota:** En producción, estos datos vendrían de un DWH/Datalake.
"""))

# Celda 2: Imports
nb.cells.append(new_code_cell("""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports exitosos")
"""))

# Celda 3: Cargar datos
nb.cells.append(new_code_cell("""
# Ruta al archivo
base_path = Path.cwd().parent.parent
csv_path = base_path / "Base_de_datos.csv"

print(f"Buscando archivo en: {csv_path}")
assert csv_path.exists(), f"Archivo no encontrado: {csv_path}"

# Cargar datos
df = pd.read_csv(csv_path)
print(f"✅ Datos cargados: {df.shape}")
print(df.head())
"""))

# Celda 4: Validación básica
nb.cells.append(new_code_cell("""
# Información del dataset
print("Información del Dataset:")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
print(f"\\nNulos por columna:\\n{df.isnull().sum()}")
print(f"\\nTipos de datos:\\n{df.dtypes}")
"""))

# Guardar notebook
nbf.write(nb, open("mlops_pipeline/src/Cargar_datos.ipynb", "w"))
print("✅ Notebook creado")
```

### 2.4 Contenido del Notebook `Cargar_datos.ipynb` (Celdas)

#### Celda 1: Markdown - Introducción

```markdown
# 1. Cargar Datos

**Objetivo:** Cargar y validar el dataset de riesgo crediticio.

En producción, estos datos vendrían de:
- Data Warehouse (DWH)
- Datalake (ADLS, S3)
- API de datos
- SQL query

**Para este proyecto:** Usamos un CSV local como simulación.
```

#### Celda 2: Python - Imports

```python
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente")
```

#### Celda 3: Python - Cargar datos

```python
# Definir ruta al archivo CSV
# Los notebooks están en: mlops_pipeline/src/
# El CSV está en: raíz del proyecto
# Por eso subimos 2 niveles (../..)

csv_path = Path.cwd().parent.parent / "Base_de_datos.csv"

print(f"📂 Buscando archivo en: {csv_path}")
print(f"   Existe: {csv_path.exists()}")

# Cargar CSV
df = pd.read_csv(csv_path)

print(f"\n✅ Dataset cargado exitosamente")
print(f"   Filas: {df.shape[0]:,}")
print(f"   Columnas: {df.shape[1]}")
print(f"\n📊 Primeras filas:")
print(df.head())
```

#### Celda 4: Python - Información del dataset

```python
print("=" * 60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 60)

print(f"\n1️⃣ DIMENSIONES")
print(f"   Filas: {df.shape[0]:,}")
print(f"   Columnas: {df.shape[1]}")

print(f"\n2️⃣ VALORES NULOS")
nulos = df.isnull().sum()
if nulos.sum() == 0:
    print("   ✅ No hay valores nulos")
else:
    print(f"   ⚠️  Hay {nulos.sum()} nulos totales:")
    print(nulos[nulos > 0])

print(f"\n3️⃣ TIPOS DE DATOS")
print(df.dtypes)

print(f"\n4️⃣ DUPLICADOS")
duplicados = df.duplicated().sum()
print(f"   Filas duplicadas: {duplicados}")

print(f"\n5️⃣ COLUMNAS")
print(f"   {', '.join(df.columns)}")
```

#### Celda 5: Python - Resumen estadístico

```python
print("\n📈 RESUMEN ESTADÍSTICO")
print(df.describe())

print("\n📊 TIPOS DE VARIABLES")
numericas = df.select_dtypes(include=[np.number]).columns.tolist()
categoricas = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numéricas: {len(numericas)} → {numericas}")
print(f"Categóricas: {len(categoricas)} → {categoricas}")
```

### 2.5 Verificar el notebook

```bash
# Ir a la carpeta del proyecto
cd /Users/smerchan/Desktop/Personal/Henry/DS/PI

# Abrir Jupyter
jupyter notebook mlops_pipeline/src/Cargar_datos.ipynb

# Ejecutar todas las celdas: Kernel > Restart & Run All
```

**Resultado esperado:**
- ✅ Dataset cargado sin errores
- ✅ Información de filas/columnas mostrada
- ✅ Resumen estadístico visible

### 2.6 Commit V1.0.1

```bash
git add mlops_pipeline/src/Cargar_datos.ipynb

git commit -m "V1.0.1: Implementar carga de datos

- Crear notebook Cargar_datos.ipynb
- Cargar dataset desde CSV
- Validar estructura y tipos de datos
- Mostrar resumen estadístico

Rama: developer
Versión: V1.0.1"

git push origin developer
```

---

## FASE 3: V1.1.0 - Exploración y Análisis de Datos (EDA)

### 🎯 Objetivo
Crear un análisis exploratorio exhaustivo del dataset que incluya:
- Análisis Univariable (cada variable por sí sola)
- Análisis Bivariable (relación con variable objetivo)
- Análisis Multivariable (correlaciones entre todas)

### 3.1 Estructura del Notebook `comprension_eda.ipynb`

Este notebook se divide en 5 secciones principales:

#### Sección 0: Configuración y Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurar estilos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Entorno configurado para EDA")
```

#### Sección 1: Exploración Inicial de Datos

**Objetivo:** Entender qué tenemos.

```python
# Cargar datos
csv_path = Path.cwd().parent.parent / "Base_de_datos.csv"
df = pd.read_csv(csv_path)

print("=" * 70)
print("EXPLORACIÓN INICIAL")
print("=" * 70)

# 1.1 Dimensiones
print(f"\n1️⃣ DIMENSIONES DEL DATASET")
print(f"   Filas: {df.shape[0]:,}")
print(f"   Columnas: {df.shape[1]}")

# 1.2 Información general
print(f"\n2️⃣ INFORMACIÓN GENERAL")
print(f"   Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   Tipos de datos:")
for dtype, count in df.dtypes.value_counts().items():
    print(f"      {dtype}: {count} columnas")

# 1.3 Valores nulos
print(f"\n3️⃣ ANÁLISIS DE NULOS")
nulos_totales = df.isnull().sum()
if nulos_totales.sum() == 0:
    print("   ✅ No hay valores nulos")
else:
    print(f"   ⚠️  Hay valores nulos:")
    for col in nulos_totales[nulos_totales > 0].index:
        pct = (nulos_totales[col] / len(df)) * 100
        print(f"      {col}: {nulos_totales[col]} ({pct:.2f}%)")

# 1.4 Valores duplicados
duplicados = df.duplicated().sum()
print(f"\n4️⃣ DUPLICADOS")
print(f"   Filas duplicadas: {duplicados}")

# 1.5 Separar variables
numericas = df.select_dtypes(include=[np.number]).columns.tolist()
categoricas = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n5️⃣ CLASIFICACIÓN DE VARIABLES")
print(f"   Numéricas ({len(numericas)}): {numericas}")
print(f"   Categóricas ({len(categoricas)}): {categoricas}")
```

#### Sección 2: Análisis Univariable

**Objetivo:** Entender cada variable individualmente.

```python
print("\n" + "=" * 70)
print("ANÁLISIS UNIVARIABLE")
print("=" * 70)

# 2.1 Para variables numéricas
print("\n📊 VARIABLES NUMÉRICAS - ESTADÍSTICOS")
print(df[numericas].describe())

# 2.2 Detectar outliers
print("\n🎯 DETECCIÓN DE OUTLIERS (usando IQR)")
for col in numericas:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        pct = (len(outliers) / len(df)) * 100
        print(f"   {col}: {len(outliers)} outliers ({pct:.2f}%)")

# 2.3 Distribución de variables categóricas
print("\n📋 VARIABLES CATEGÓRICAS - CONTEOS")
for col in categoricas:
    print(f"\n   {col}:")
    print(df[col].value_counts())

# 2.4 Visualización univariable
fig, axes = plt.subplots(len(numericas), 1, figsize=(12, 4*len(numericas)))
if len(numericas) == 1:
    axes = [axes]

for idx, col in enumerate(numericas):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribución: {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Sección 3: Análisis Bivariable

**Objetivo:** Relación entre cada variable y la variable objetivo.

**Importante:** Primero debes identificar cuál es tu variable objetivo (target):
- En riesgo crediticio: generalmente es "default", "riesgo", "churn", etc.
- Ajusta la siguiente sección según tu dataset

```python
print("\n" + "=" * 70)
print("ANÁLISIS BIVARIABLE")
print("=" * 70)

# ⚠️  AJUSTA ESTO SEGÚN TU DATASET
# Asume que la variable objetivo se llama "default" o "target"
# Si es diferente, cámbialo
TARGET = "default"  # <-- CAMBIAR SI ES NECESARIO

if TARGET not in df.columns:
    print(f"⚠️  AVISO: No se encontró columna '{TARGET}'")
    print(f"   Columnas disponibles: {df.columns.tolist()}")
else:
    # 3.1 Balanceo de clases (si es binaria)
    if df[TARGET].dtype == 'object' or df[TARGET].nunique() <= 5:
        print(f"\n📊 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO: {TARGET}")
        print(df[TARGET].value_counts())
        print("\nProporción:")
        print(df[TARGET].value_counts(normalize=True) * 100)
        
        # Visualizar
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        df[TARGET].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title(f'Distribución de {TARGET}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Conteo')
        ax.set_xlabel(TARGET)
        plt.tight_layout()
        plt.show()
    
    # 3.2 Relación: numéricas vs objetivo
    print(f"\n📈 VARIABLES NUMÉRICAS vs {TARGET}")
    for col in numericas:
        if col != TARGET:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            
            # Boxplot: visualizar distribución por clase
            df.boxplot(column=col, by=TARGET, ax=ax)
            ax.set_title(f'{col} vs {TARGET}', fontsize=12, fontweight='bold')
            ax.set_ylabel(col)
            ax.set_xlabel(TARGET)
            plt.suptitle('')  # Quitar título por defecto
            plt.tight_layout()
            plt.show()
    
    # 3.3 Relación: categóricas vs objetivo
    print(f"\n📊 VARIABLES CATEGÓRICAS vs {TARGET}")
    for col in categoricas:
        if col != TARGET:
            print(f"\n   {col} vs {TARGET}:")
            tabla = pd.crosstab(df[col], df[TARGET], margins=True)
            print(tabla)
            
            # Visualizar
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            pd.crosstab(df[col], df[TARGET]).plot(kind='bar', ax=ax)
            ax.set_title(f'{col} vs {TARGET}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conteo')
            ax.set_xlabel(col)
            plt.legend(title=TARGET)
            plt.tight_layout()
            plt.show()
```

#### Sección 4: Análisis Multivariable

**Objetivo:** Entender correlaciones y relaciones complejas.

```python
print("\n" + "=" * 70)
print("ANÁLISIS MULTIVARIABLE")
print("=" * 70)

# 4.1 Matriz de correlación (solo para numéricas)
if len(numericas) > 1:
    print("\n📊 MATRIZ DE CORRELACIÓN")
    corr_matrix = df[numericas].corr()
    print(corr_matrix)
    
    # Visualizar
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlación'})
    ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Identificar correlaciones altas
    print("\n🔗 CORRELACIONES ALTAS (>0.8)")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                print(f"   {col1} ↔ {col2}: {corr_val:.3f}")

# 4.2 Pairplot (relación entre todas las variables)
if len(numericas) <= 5:  # Solo si hay pocas variables
    print("\n📈 PAIRPLOT (Relación entre variables)")
    if TARGET in df.columns and df[TARGET].dtype == 'object':
        pairplot = sns.pairplot(df[numericas + [TARGET]], hue=TARGET, diag_kind='hist')
    else:
        pairplot = sns.pairplot(df[numericas], diag_kind='hist')
    plt.show()

# 4.3 Análisis de variables derivadas
print("\n✨ VARIABLES DERIVADAS POTENCIALES")
print("   Considerar crear:")
print("   - Ratios (ej: debt_to_income = deuda / ingresos)")
print("   - Polinomios (ej: edad²)")
print("   - Interacciones (ej: edad × ingresos)")
print("   - Binning (ej: agrupar edades en rangos)")
```

#### Sección 5: Conclusiones y Próximos Pasos

```python
print("\n" + "=" * 70)
print("CONCLUSIONES Y PRÓXIMOS PASOS")
print("=" * 70)

print("""
📝 HALLAZGOS CLAVE:
   1. Dataset tiene X filas y Y columnas
   2. No hay valores nulos / Hay nulos en: [listar]
   3. Distribución de clases: [balanceado/desbalanceado]
   4. Correlaciones importantes: [listar]
   5. Outliers detectados en: [listar]

🔧 PRÓXIMOS PASOS:
   1. Feature Engineering (crear nuevas variables)
   2. Manejo de valores faltantes
   3. Tratamiento de outliers
   4. Escalado de variables
   5. Selección de features relevantes
   6. División train/test
   7. Entrenamiento de modelos
""")
```

### 3.2 Crear el notebook completo

Puedes crear el notebook manualmente en Jupyter o programáticamente. Te recomiendo:

1. Abre Jupyter: `jupyter notebook`
2. Ve a `mlops_pipeline/src/`
3. Crea nuevo notebook: `comprension_eda.ipynb`
4. Copia las celdas de la Sección 3.1

### 3.3 Ejecutar y validar

```bash
# En Jupyter, ejecuta todas las celdas:
# Kernel > Restart & Run All

# O desde terminal:
jupyter nbconvert --to notebook --execute mlops_pipeline/src/comprension_eda.ipynb
```

### 3.4 Commit V1.1.0

```bash
git add mlops_pipeline/src/comprension_eda.ipynb

git commit -m "V1.1.0: Análisis exploratorio de datos (EDA)

- Exploración inicial de datos
- Análisis univariable (descripción, distribuciones, outliers)
- Análisis bivariable (relación con variable objetivo)
- Análisis multivariable (correlaciones, pairplot)
- Identificación de variables derivadas potenciales
- Conclusiones y próximos pasos

Rama: developer
Versión: V1.1.0"

git push origin developer
```

---

## Comandos Git: Branching y Merging

### 🎯 Flujo completo de Git para el PI

#### Parte 1: Desarrollar en `developer`

```bash
# Estar en developer
git checkout developer

# Crear feature branch (opcional, para features grandes)
git checkout -b feature/cargar-datos

# Hacer cambios, commits, etc.
git add .
git commit -m "..."
git push origin feature/cargar-datos

# Hacer Pull Request en GitHub (si trabajas en equipo)
# O simplemente mergear localmente:
git checkout developer
git merge feature/cargar-datos
git push origin developer
```

#### Parte 2: Preparar para `certification` (pruebas)

```bash
# Cambiar a certification
git checkout certification

# Traer cambios de developer
git merge developer

# Hacer pruebas/certificación

# Subir
git push origin certification
```

#### Parte 3: Release a `main` (producción)

```bash
# Cambiar a main
git checkout main

# Traer cambios de certification
git merge certification

# (Opcional) Crear tag para versión
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin main --tags

# Volver a developer para seguir trabajando
git checkout developer
```

### ⚠️ Regla de Oro

**NUNCA hagas push directo a `main`**

Siempre:
1. Desarrollar en `developer`
2. Probar en `certification`
3. Mergear a `main` solo cuando esté listo

---

## Flujo de Trabajo Completo (Resumen)

### Timeline del Proyecto Integrador

| Fase | Rama | Versión | Entrega | Focus |
|------|------|---------|---------|-------|
| 1 | developer | V1.0.0 | Clase 1 | Estructura + requirements |
| 2 | developer | V1.0.1 | Clase 2 | Cargar datos |
| 3 | developer | V1.1.0 | Clase 2 | EDA |
| **MERGE** | main | V1.0.1 | Clase 3 | Aprobar compañero |
| 4 | developer | V1.1.1 | Clase 4 | Feature Engineering |
| 5 | developer | V1.2.0 | Clase 4 | Entrenamiento modelo |
| 6 | developer | V2.0.0 | Clase 7 | Monitoreo + Streamlit |
| 7 | developer | V2.1.0 | Clase 9 | Docker + FastAPI |

### Checklist para cada Release

```markdown
## ✅ Pre-release Checklist

- [ ] Código escrito en `developer`
- [ ] Todos los tests pasan
- [ ] Notebooks ejecutan sin errores
- [ ] Documentación actualizada
- [ ] `requirements.txt` actualizado
- [ ] `.gitignore` completo
- [ ] Commits con mensajes claros
- [ ] Sin archivos sensibles (contraseñas, datos)
- [ ] Merge a `main` aprobado por compañero
- [ ] Tag de versión creado (ej: `v1.0.1`)

## 📋 Cambios incluidos en esta versión

- [ ] Cambio 1
- [ ] Cambio 2
- [ ] Cambio 3
```

---

## 📝 Notas para el Instructor

### Puntos pedagógicos clave

1. **Estructura de carpetas:** Enfatiza que Jenkins valida esta estructura
2. **Versionado semántico:** 
   - `MAJOR.MINOR.PATCH` (ej: `1.0.1`)
   - Major: cambios que rompen compatibilidad
   - Minor: nuevas funcionalidades
   - Patch: bug fixes

3. **Ramas:**
   - `main` es sagrada → solo código estable
   - `developer` es el área de trabajo
   - `certification` es el test pre-producción

4. **Commits:** Enseña a escribir mensajes descriptivos
   - "arreglé un bug" ❌
   - "Corregir cálculo de AUC en predicción holdout" ✅

5. **EDA:** Es el paso CRÍTICO
   - Garbage in = Garbage out
   - Una buena EDA evita problemas después
   - Las visualizaciones hablan más que los números

### Tiempos sugeridos por sesión

| Sesión | Tema | Tiempo |
|--------|------|--------|
| 1 | Git setup + V1.0.0 | 30 min |
| 2 | V1.0.1 + V1.1.0 | 60 min |
| 3 | Merge a main + Code review | 30 min |
| 4+ | Feature Engineering + Training | 60+ min |

### Errores comunes de estudiantes

1. **"Olvidé cambiar de rama"** → hacer commits en main
   - Solución: `git reflog`, `git reset --hard`

2. **"Subí datos sensibles"** → no usar `.gitignore` correctamente
   - Solución: `git rm --cached`, `git filter-branch`

3. **"La estructura cambió"** → Jenkins lo rechaza
   - Solución: NUNCA mover carpetas de `mlops_pipeline/src/`

4. **"Jupythe no guarda"** → trabajar con notebooks pero sin commitear `.ipynb`
   - Solución: usar nbdime, o convertir a `.py` en producción

---

## 📚 Referencias

- [Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/)
- [Semantic Versioning](https://semver.org/)
- [Python Notebook Best Practices](https://nbformat.readthedocs.io/)
- [EDA Guide](https://en.wikipedia.org/wiki/Exploratory_data_analysis)

---

**Última actualización:** agosto 2026  
**Versión del documento:** 1.0.0  
**Para:** Instructores del Bootcamp Henry - Módulo 5

EOF
```

Perfecto. Ahora, tengo algunas **preguntas importantes** antes de continuar:

## ❓ PREGUNTAS

1. **¿Puedes copiar manualmente estos archivos a la carpeta PI?**
   ```bash
   cp /Users/smerchan/Downloads/Base_de_datos.xlsx ~/Desktop/Personal/Henry/DS/PI/
   cp /Users/smerchan/Downloads/requirements.txt ~/Desktop/Personal/Henry/DS/PI/
   cp /Users/smerchan/Downloads/set_up.bat ~/Desktop/Personal/Henry/DS/PI/
   ```

2. **¿El archivo Excel (`Base_de_datos.xlsx`) necesita conversión a CSV o lo manejamos como Excel?**

3. **¿Ya tienes un entorno virtual configurado, o empezamos a crear uno desde cero?**

Mientras responde, **acabo de crear una Guía Markdown completa** en:
```
/Users/smerchan/Desktop/Personal/Henry/DS/PI/GUIA_DESARROLLO_PI.md
```

Esa guía tiene TODO (comando por comando) para que enseñes este proyecto en clase. 

**¿Quieres que empecemos a ejecutar los primeros pasos ahora?** 🚀