# Modelo de riesgo crediticio

Modelo que anticipa si un cliente va a pagar un crédito a tiempo (`Pago_atiempo`), a partir del historial de originación de la entidad.

---

## Caso de negocio

La entidad aprueba créditos todos los días y necesita saberlo **antes de desembolsar**. El impacto es doble:

- **Pérdida esperada:** aprobar a quien no va a pagar se vuelve cartera vencida y provisiones.
- **Costo de oportunidad:** negar a un buen cliente es dejar de colocar un crédito rentable.

El dataset tiene 10.763 registros y 23 variables, y está desbalanceado (cerca del 4,7% de mora). El desempeño se mide con precision, recall, F1 y ROC-AUC.

## ¿Por qué monitorear data drift?

El modelo se entrenó con una fotografía del pasado. Si cambia el perfil de quien solicita, el score del buró o la política de originación, las entradas dejan de parecerse a las del entrenamiento. Eso es **data drift**. El pipeline no se cae: sigue respondiendo, cada vez peor. El monitoreo avisa antes de que la mora lo confirme.

## Estructura

```
henry-sm-modelo-riesgo/
├── mlops_pipeline/src/
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py          # monitoreo de drift
│   └── app_monitoreo.py             # tablero Streamlit
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── README.MD
```

Esta estructura es fija. No la reorganices: las validaciones de la compañía dependen de estos nombres.

## Monitoreo

`mlops_pipeline/src/model_monitoring.py` compara el histórico (`Base_de_datos.csv`) contra un batch actual, variable por variable.

| Métrica | Constante | Umbral | Aplica a | Drift si… |
|---------|-----------|--------|----------|-----------|
| PSI | `PSI_UMBRAL` | 0.25 | Numéricas | PSI > 0.25 |
| KS | `KS_UMBRAL` | 0.25 | Numéricas | estadístico > 0.25 |
| Jensen-Shannon | `JS_UMBRAL` | 0.3 | Numéricas y categóricas | distancia > 0.3 |
| Chi-cuadrado | `CHI2_PVALUE_UMBRAL` | 0.05 | Categóricas | p-value < 0.05 |

Basta con que una métrica cruce su umbral. Se monitorean 11 numéricas y 3 categóricas. Quedan fuera `fecha_prestamo` y `Pago_atiempo` (es el target).

Cada corrida sobrescribe `drift_report.json` y agrega filas a `drift_history.csv`. En producción esto corre con periodicidad definida (por ejemplo, semanal).

## Cómo ejecutarlo

```bash
cd /Users/smerchan/Desktop/Personal/Henry/DS/M5/Proyecto/henry-sm-modelo-riesgo
source venv/bin/activate
python mlops_pipeline/src/model_monitoring.py
streamlit run mlops_pipeline/src/app_monitoreo.py
```

La app tiene tres pestañas: métricas de la última corrida, evolución en el tiempo, y recomendaciones con severidad.

## Hallazgos de la última corrida

Pendiente. Esta sección se completa después de ejecutar `model_monitoring.py` en clase. Los valores reales quedan en `drift_report.json`.

## El batch actual es simulado

Todavía no hay un log de predicciones en producción. `generar_batch_actual()` toma el 30% del histórico (`seed=42`), le suma ruido y desplaza la media de `puntaje` y `salario_cliente`. Arriba de esa función hay un `TODO` para reemplazarla por el log real. Las métricas y los umbrales sí son definitivos.

## Ramas

```
developer  →  certification  →  main
```

Este avance es el Pull Request de `developer` hacia `certification`. No se abre PR hacia `main`.
