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
│       └── model_monitoring.py        # 6️⃣ Monitorear en producción
│
├── Base_de_datos.csv                  # 📊 Dataset principal (10,763 registros)
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
| **V2.0.0** | Monitoreo + Streamlit | ⏳ Por hacer |

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
