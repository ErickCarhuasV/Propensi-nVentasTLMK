# PropensionVentasTLMK
Proyecto end-to-end de ciencia de datos: **¿a qué clientes conviene llamar para ofrecer una tarjeta de crédito?**
Se simula una campaña de contact center (30 000 llamadas, 6 meses), se entrena un modelo de propensión y se traduce a impacto económico.

Todos los datos son sintéticos (generados en el archivo `src/simulate_data.py`).

## Resultados (test temporal dado entre mayo–junio)

| Modelo | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Regresión logística (baseline) | 0.788 | 0.326 | 0.076 |
| HistGradientBoosting | 0.784 | 0.313 | 0.076 |

- **Top 3 deciles capturan el 68 % de las conversiones** (lift del decil 1 = 3.6x).
- Con costo de S/ 12 por llamada y margen de S/ 120 por tarjeta:

| Estrategia | Llamadas | Conversiones | Utilidad (S/) |
|---|---|---|---|
| Llamar a todos | 10 025 | 967 | **-4 260** |
| 30 % al azar | 3 007 | 270 | -3 684 |
| **Top 30 % según modelo** | 3 007 | 661 | **+43 236** |
| Política óptima (p ≥ costo/margen) | 3 116 | 670 | +43 008 |

## Decisiones técnicas clave
1. **Split temporal** (entrena ene–abr, evalúa may–jun) en lugar de split aleatorio: simula el uso real en producción.
2. **Data leakage controlado:** `duracion_seg` se simula pero se **excluye** del modelo (solo se conoce después de la llamada).
3. **Métricas para clases desbalanceadas** (~10 % de conversión): PR-AUC + Brier (calibración), no solo accuracy/ROC.
4. **Baseline primero:** la regresión logística iguala al boosting; se prioriza el modelo simple, calibrado e interpretable.
5. **Traducción a negocio:** lift por deciles, curva de captura y simulación de utilidad con umbral de equilibrio `p* = costo / margen`.
6. **Importancia de variables** con *permutation importance* (sobre PR-AUC en test).

## Estructura
```
├── sql/features_sqlserver.sql  # cómo se armarían las features en SQL Server 
├── src/simulate_data.py   # simulador de la campaña
├── src/train_model.py     # modelado + evaluación + ROI + figuras
├── reports/               # metrics.json, lift_deciles.csv, figures/
└── requirements.txt
```

## Cómo ejecutarlo
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/simulate_data.py
python src/train_model.py
```

## Próximos pasos
- Tuning con validación cruzada temporal y SHAP.
- Exponer el scoring en una API (FastAPI) o dashboard (Power BI / Streamlit).

## Stack
Python · pandas · NumPy · scikit-learn · matplotlib · SQL Server (feature engineering)
