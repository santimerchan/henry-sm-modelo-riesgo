# 🎓 GUÍA COMPLETA: PROYECTO INTEGRADOR 
## Riesgo Crediticio + Cursor IDE + Entorno Virtual

**Para:** Santiago Merchan  
**IDE:** Cursor  
**Sistema:** macOS (Apple Silicon)  
**Python:** 3.10+  
**Versión Guía:** V1.0.0

---

## 📍 ÍNDICE RÁPIDO

1. [Setup Inicial: Crear Entorno Virtual](#1-setup-inicial-crear-entorno-virtual)
2. [Configurar Cursor IDE](#2-configurar-cursor-ide)
3. [Ejecutar Jupyter en Cursor](#3-ejecutar-jupyter-en-cursor)
4. [V1.0.0 - Estructura Base](#4-v100---estructura-base) ✅ COMPLETADO
5. [V1.0.1 - Cargar Datos](#5-v101---cargar-datos)
6. [V1.1.0 - EDA](#6-v110---eda)
7. [Git Workflow Completo](#7-git-workflow-completo)

---

## 1️⃣ SETUP INICIAL: CREAR ENTORNO VIRTUAL

### Paso 1.1: Abre la Terminal en Cursor

En Cursor:
```
Ctrl + ` (backtick) → abre Terminal integrada
O: Terminal > New Terminal en el menú
```

Verás algo como:
```
❯ /Users/smerchan/Desktop/Personal/Henry/DS/PI
```

### Paso 1.2: Verifica que tienes Python 3.10+

```bash
python3 --version
```

**Salida esperada:**
```
Python 3.10.x (o superior)
```

**Si dice 3.9 o menos:**
```bash
# Instalar Python 3.11 (recomendado)
brew install python@3.11

# O verificar si está disponible
brew search python@
```

### Paso 1.3: Crear el entorno virtual

```bash
# Este comando crea una carpeta llamada 'venv'
# dentro del proyecto
python3 -m venv venv
```

**¿Qué sucede?**
- ✅ Se crea carpeta `venv/` (la verás en el explorador de archivos de Cursor)
- ✅ Contiene Python aislado solo para este proyecto
- ✅ Tiene carpetas: `bin/`, `lib/`, `include/`

**Verifica que se creó:**
```bash
ls -la venv/
```

Deberías ver:
```
bin/     lib/     pyvenv.cfg
```

### Paso 1.4: Activar el entorno virtual

**En macOS/Linux (que es tu caso):**

```bash
source venv/bin/activate
```

**¿Cómo sé que funcionó?**
Mira la línea de la terminal:

```
ANTES:  ❯ /Users/smerchan/Desktop/Personal/Henry/DS/PI
DESPUÉS: (venv) ❯ /Users/smerchan/Desktop/Personal/Henry/DS/PI
```

El `(venv)` al inicio significa que el entorno está activado ✅

### Paso 1.5: Actualizar pip (gestor de paquetes)

```bash
pip install --upgrade pip
```

**Verás un mensaje como:**
```
Successfully installed pip-26.2.1
```

### Paso 1.6: Instalar todas las dependencias

```bash
pip install -r requirements.txt
```

**Esto puede tardar 2-3 minutos...**

Verás:
```
Collecting pandas...
Collecting numpy...
...
Successfully installed pandas-2.3.3 numpy-2.2.6 ...
```

### Paso 1.7: Verificar que todo está listo

```bash
python -c "import pandas; import jupyter; print('✅ Todo listo')"
```

**Salida esperada:**
```
✅ Todo listo
```

---

## 2️⃣ CONFIGURAR CURSOR IDE

### Paso 2.1: Configurar Python Path en Cursor

En Cursor, necesitamos decirle dónde está nuestro Python.

**Opción A: Automática (Recomendado)**

1. Abre la paleta de comandos: `Cmd + Shift + P`
2. Busca: `Python: Select Interpreter`
3. Debería aparecerte una opción como:
   ```
   ./venv/bin/python
   ```
4. Selecciona esa

**Opción B: Manual**

1. Ve a `Cursor > Settings` (o `Cmd + ,`)
2. Busca: `Python: Default Interpreter Path`
3. Escribe:
   ```
   /Users/smerchan/Desktop/Personal/Henry/DS/PI/venv/bin/python
   ```

### Paso 2.2: Verificar que Cursor usa tu venv

1. Abre un notebook: `mlops_pipeline/src/Cargar_datos.ipynb`
2. Mira la esquina superior derecha del notebook
3. Debería mostrar algo como: `Python 3.10 (venv)`

Si muestra otro Python o dice "No kernel found":
- Haz clic en ese selector
- Elige el Python de tu venv

---

## 3️⃣ EJECUTAR JUPYTER EN CURSOR

### Opción A: Abrir Notebooks directamente en Cursor

**Este es el método que recomiendo:**

1. Abre Cursor
2. En el explorador de archivos (izquierda), ve a:
   ```
   mlops_pipeline/src/Cargar_datos.ipynb
   ```
3. Haz clic en el archivo

Cursor abrirá el notebook y podrás:
- ✅ Ver las celdas
- ✅ Escribir código
- ✅ Ejecutar celdas (`Shift + Enter`)
- ✅ Ver gráficos en línea

### Opción B: Terminal con Jupyter Server

Si prefieres abrir Jupyter en navegador:

```bash
# Asegúrate que el venv está activado
source venv/bin/activate

# Inicia Jupyter
jupyter notebook
```

**Salida esperada:**
```
[I 12:34:56.789 NotebookApp] Serving notebooks from local directory: /Users/smerchan/Desktop/Personal/Henry/DS/PI
[I 12:34:56.790 NotebookApp] Jupyter Server 4.4.0 is running at:
[I 12:34:56.790 NotebookApp] http://localhost:8888/tree
```

Copia la URL en tu navegador (Chrome, Safari, etc.)

---

## 4️⃣ V1.0.0 - ESTRUCTURA BASE ✅

**COMPLETADO** en los pasos anteriores.

### Qué incluye V1.0.0:

```
✅ Estructura de carpetas (mlops_pipeline/src/)
✅ requirements.txt con todas las dependencias
✅ .gitignore configurado
✅ README.md con documentación
✅ 3 ramas en GitHub (main, developer, certification)
✅ Commit en rama developer
```

### Verificar que todo está en GitHub:

```bash
git log --oneline | head -5
```

Deberías ver tu commit V1.0.0:
```
ce48a13 V1.0.0: Estructura base del Proyecto Integrador...
```

---

## 5️⃣ V1.0.1 - CARGAR DATOS

### Objetivo
Crear el notebook `Cargar_datos.ipynb` que cargue el CSV y valide los datos.

### Paso 5.1: Asegúrate de estar en rama developer

```bash
git status
```

Debe mostrar:
```
On branch developer
Your branch is up to date with 'origin/developer'.
```

Si NO estás en developer:
```bash
git checkout developer
```

### Paso 5.2: Abrir Cursor y crear el notebook

En Cursor:
1. Abre: `mlops_pipeline/src/Cargar_datos.ipynb`
2. Cursor debería preguntarte si crear un nuevo kernel
3. Haz clic en "Create" o "Select Kernel"

### Paso 5.3: Escribir Celda 1 - Imports

```python
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente")
```

Ejecuta: `Shift + Enter`

### Paso 5.4: Escribir Celda 2 - Cargar CSV

```python
# Ruta al archivo CSV
# El notebook está en: mlops_pipeline/src/
# El CSV está en: raíz del proyecto (subir 2 niveles)

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

Ejecuta: `Shift + Enter`

**Salida esperada:**
```
📂 Buscando archivo en: /Users/smerchan/Desktop/Personal/Henry/DS/PI/Base_de_datos.csv
   Existe: True

✅ Dataset cargado exitosamente
   Filas: 10,763
   Columnas: 23
```

### Paso 5.5: Escribir Celda 3 - Información del Dataset

```python
print("=" * 70)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 70)

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

print(f"\n5️⃣ NOMBRES DE COLUMNAS")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")
```

Ejecuta: `Shift + Enter`

### Paso 5.6: Escribir Celda 4 - Resumen Estadístico

```python
print("\n" + "=" * 70)
print("RESUMEN ESTADÍSTICO")
print("=" * 70)
print(df.describe())

print("\n" + "=" * 70)
print("VARIABLES CATEGÓRICAS")
print("=" * 70)

# Identificar variables categóricas
categoricas = df.select_dtypes(include=['object']).columns
print(f"Variables categóricas ({len(categoricas)}): {list(categoricas)}")

# Mostrar valores únicos de cada una
for col in categoricas:
    print(f"\n{col}:")
    print(f"  Valores únicos: {df[col].nunique()}")
    print(f"  Valores: {df[col].unique()[:5]}...")  # Mostrar primeros 5
```

Ejecuta: `Shift + Enter`

### Paso 5.7: Guardar el notebook

`Cmd + S` (en Cursor, se auto-guarda)

### Paso 5.8: Commit V1.0.1

En la terminal de Cursor:

```bash
# Asegúrate que el venv está activado
source venv/bin/activate

# Ver cambios
git status

# Agregar cambios
git add mlops_pipeline/src/Cargar_datos.ipynb

# Commit con mensaje descriptivo
git commit -m "V1.0.1: Implementar carga de datos

- Crear notebook Cargar_datos.ipynb
- Cargar dataset desde CSV (10,763 filas, 23 columnas)
- Validar estructura: tipos de datos, nulos, duplicados
- Mostrar resumen estadístico y variables categóricas
- Verificar variable objetivo: Pago_atiempo

Rama: developer
Versión: V1.0.1"

# Subir a GitHub
git push origin developer
```

**Verifica en GitHub:**
```
https://github.com/santimerchan/henry-sm-modelo-riesgo/tree/developer
```

---

## 6️⃣ V1.1.0 - ANÁLISIS EXPLORATORIO (EDA)

### Objetivo
Crear `comprension_eda.ipynb` con análisis completo del dataset.

### Paso 6.1: Abrir el notebook

En Cursor:
```
mlops_pipeline/src/comprension_eda.ipynb
```

### Paso 6.2: Celda 1 - Markdown - Introducción

```markdown
# Análisis Exploratorio de Datos (EDA)

## Objetivo
Entender el dataset mediante visualizaciones y análisis estadísticos.

## Fases del EDA:
1. **Exploración Inicial** - Dimensiones, tipos, nulos
2. **Análisis Univariable** - Cada variable por sí sola
3. **Análisis Bivariable** - Variables vs target (Pago_atiempo)
4. **Análisis Multivariable** - Correlaciones entre todas
5. **Conclusiones** - Próximos pasos
```

### Paso 6.3: Celda 2 - Imports y Configuración

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configurar estilos para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Entorno configurado para EDA")
```

### Paso 6.4: Celda 3 - Cargar Datos

```python
# Cargar datos (mismo código que Cargar_datos.ipynb)
csv_path = Path.cwd().parent.parent / "Base_de_datos.csv"
df = pd.read_csv(csv_path)

print(f"✅ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
```

### Paso 6.5: Celda 4 - ANÁLISIS UNIVARIABLE

```python
print("=" * 70)
print("ANÁLISIS UNIVARIABLE")
print("=" * 70)

# Variables numéricas
numericas = df.select_dtypes(include=[np.number]).columns.tolist()
categoricas = df.select_dtypes(include=['object']).columns.tolist()

# Para variables numéricas
print(f"\n📊 VARIABLES NUMÉRICAS ({len(numericas)}):")
for col in numericas:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std: {df[col].std():.2f}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  Nulos: {df[col].isnull().sum()}")

# Gráficos de distribución
fig, axes = plt.subplots(5, 2, figsize=(14, 15))
axes = axes.flatten()

for idx, col in enumerate(numericas[:10]):  # Primeras 10
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(f'{col}', fontweight='bold')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(True, alpha=0.3)

# Ocultar subplots vacíos
for idx in range(len(numericas), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

print("\n✅ Gráficos de distribución mostrados")
```

### Paso 6.6: Celda 5 - ANÁLISIS BIVARIABLE (vs Target)

```python
print("\n" + "=" * 70)
print("ANÁLISIS BIVARIABLE - Relación con Variable Objetivo")
print("=" * 70)

# Variable objetivo
TARGET = "Pago_atiempo"

print(f"\n📊 DISTRIBUCIÓN DE {TARGET}:")
print(df[TARGET].value_counts())
print(f"\nProporción:")
print(df[TARGET].value_counts(normalize=True) * 100)

# Gráfico de distribución del target
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
df[TARGET].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
ax.set_title(f'Distribución de {TARGET}', fontsize=12, fontweight='bold')
ax.set_ylabel('Conteo')
ax.set_xlabel(TARGET)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Relación entre variables numéricas y target
print(f"\n\n🔗 RELACIÓN: VARIABLES NUMÉRICAS vs {TARGET}")
for col in numericas[:6]:  # Primeras 6 variables
    if col != TARGET:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        df.boxplot(column=col, by=TARGET, ax=ax)
        ax.set_title(f'{col} vs {TARGET}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        ax.set_xlabel(TARGET)
        plt.suptitle('')  # Quitar título por defecto
        plt.tight_layout()
        plt.show()
```

### Paso 6.7: Celda 6 - ANÁLISIS MULTIVARIABLE

```python
print("\n" + "=" * 70)
print("ANÁLISIS MULTIVARIABLE")
print("=" * 70)

# Matriz de correlación
print("\n📊 MATRIZ DE CORRELACIÓN (primeras 10 variables)")
corr_matrix = df[numericas[:10]].corr()
print(corr_matrix)

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax, cbar_kws={'label': 'Correlación'})
ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Identificar correlaciones altas
print("\n🔗 CORRELACIONES ALTAS (> 0.7)")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
```

### Paso 6.8: Celda 7 - CONCLUSIONES

```python
print("\n" + "=" * 70)
print("CONCLUSIONES Y PRÓXIMOS PASOS")
print("=" * 70)

print(f"""
✅ HALLAZGOS CLAVE:
   1. Dataset tiene {df.shape[0]:,} registros y {df.shape[1]} variables
   2. Nulos: {df.isnull().sum().sum()} totales
   3. Duplicados: {df.duplicated().sum()}
   4. Variable objetivo ({TARGET}): {df[TARGET].value_counts()[0]} vs {df[TARGET].value_counts()[1]}
   5. Distribución: Balanceada/Desbalanceada (ver gráficos)

🔧 PRÓXIMOS PASOS (V1.1.1 onwards):
   1. Feature Engineering (crear nuevas variables)
   2. Manejo de valores faltantes
   3. Tratamiento de outliers
   4. Normalización/escalado
   5. Selección de features
   6. División train/test
   7. Entrenamiento de modelos
   8. Evaluación y validación
""")
```

### Paso 6.9: Guardar y Commit V1.1.0

```bash
# Guarda: Cmd + S (auto)

# Commit
git add mlops_pipeline/src/comprension_eda.ipynb

git commit -m "V1.1.0: Análisis Exploratorio de Datos (EDA)

- Exploración inicial del dataset
- Análisis univariable (distribuciones, estadísticos)
- Análisis bivariable (variables vs target Pago_atiempo)
- Análisis multivariable (matriz de correlación)
- Identificación de correlaciones altas
- Conclusiones y próximos pasos

Rama: developer
Versión: V1.1.0"

git push origin developer
```

---

## 7️⃣ GIT WORKFLOW COMPLETO

### Después de V1.1.0: Preparar para Merge a Main

### Paso 7.1: Ir a rama certification

```bash
git checkout certification
git merge developer
git push origin certification
```

### Paso 7.2: Pruebas en certification

```bash
# Ejecutar todos los notebooks en certification para verificar
# que no hay errores

# Abrir cada notebook en Cursor y ejecutar todas las celdas
```

### Paso 7.3: Ir a rama main y mergear

```bash
git checkout main
git merge certification

# (Opcional) Crear tag de versión
git tag -a v1.1.0 -m "Release v1.1.0 - EDA completado"

# Subir a GitHub
git push origin main --tags
```

### Paso 7.4: Volver a developer

```bash
git checkout developer
```

---

## 📋 CHECKLIST RÁPIDO

### Para cada versión (V1.0.1, V1.1.0, etc.):

- [ ] Activar venv: `source venv/bin/activate`
- [ ] Estoy en rama developer: `git checkout developer`
- [ ] Crear/modificar notebook en `mlops_pipeline/src/`
- [ ] Ejecutar todas las celdas (`Shift + Enter`)
- [ ] Guardar notebook (`Cmd + S`)
- [ ] Hacer commit con mensaje descriptivo
- [ ] Subir a GitHub: `git push origin developer`
- [ ] Verificar en GitHub que los cambios están

---

## 🚨 ERRORES COMUNES Y SOLUCIONES

### Error: "No module named 'pandas'"

**Causa:** El venv no está activado

**Solución:**
```bash
source venv/bin/activate
python -c "import pandas; print('OK')"
```

### Error: "Archivo CSV no encontrado"

**Causa:** Ruta incorrecta en el notebook

**Solución:**
Verifica que el notebook está en:
```
mlops_pipeline/src/Cargar_datos.ipynb
```

Y que Base_de_datos.csv está en:
```
/ (raíz del proyecto)
```

### Error: "Kernel not found" en Cursor

**Causa:** Cursor no sabe dónde está Python

**Solución:**
1. `Cmd + Shift + P` → "Python: Select Interpreter"
2. Selecciona `./venv/bin/python`

### Comando `source venv/bin/activate` no funciona

**Si estás en Windows:**
```bash
venv\Scripts\activate
```

**Si estás en macOS/Linux (tu caso):**
```bash
source venv/bin/activate
```

---

## 📞 RESUMEN COMANDOS CLAVE

```bash
# Activar venv (siempre primero)
source venv/bin/activate

# Ver en qué rama estoy
git status

# Cambiar de rama
git checkout developer

# Ver cambios
git status

# Agregar cambios
git add .

# Hacer commit
git commit -m "Tu mensaje"

# Subir a GitHub
git push origin developer

# Ver historial
git log --oneline

# Desactivar venv
deactivate
```

---

**¿Preguntas o problemas?**  
Mira la carpeta `GUIA_DESARROLLO_PI.md` para más detalles pedagógicos.

---

**Última actualización:** agosto 2026  
**Documento:** V1.0.0
