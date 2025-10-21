# ESM Wideband CFAR Detector

[![CI](https://github.com/tangodelta217/esm-wideband-cfar/actions/workflows/ci.yml/badge.svg)](https://github.com/tangodelta217/esm-wideband-cfar/actions/workflows/ci.yml)

Implementación y evaluación de algoritmos de detección CFAR (Constant False Alarm Rate) de promediado de celdas (CA-CFAR) en 1D y 2D, optimizados con NumPy y SciPy.

Este repositorio incluye:
- Implementaciones vectorizadas de CA-CFAR 1D y 2D.
- Un tracker de picos simple para asociar detecciones a lo largo del tiempo.
- Un flowgraph de GNU Radio para procesar datos IQ de extremo a extremo, incluyendo lógica de **agrupación de picos (clustering)**.
- Un script de evaluación para caracterizar empíricamente el rendimiento (Pfa, Pd, latencia).
- Tests unitarios para validar la funcionalidad.

## Quickstart (5 minutos)

Sigue estos pasos para tener el proyecto funcionando.

**1. Clonar el Repositorio**
```bash
git clone https://github.com/tangodelta217/esm-wideband-cfar.git
cd esm-wideband-cfar
```

**2. Instalar Dependencias**

Se requiere Python 3.8+.

```bash
pip install -r requirements.txt
```

**3. Ejecutar la Prueba de Humo (Smoke Test)**

Este comando procesa un archivo IQ sintético con un tono conocido y verifica que el pipeline lo detecte correctamente. Necesitarás tener **GNU Radio instalado** para que se ejecute.

```bash
python gnuradio/run_cfar_flow.py
```

Salida esperada:
```
--- Resultados de Detección ---
Detecciones CFAR brutas: 5
Picos finales (post-clustering): 1
Frecuencia(s) de picos (Hz): [1.00075195e+08]
Resultado: ✅ Coincide con la predicción de la prueba de humo.

--- Métricas de Rendimiento ---
- Latencia media por trama: ... ms
```

## Uso y Evaluación

### Ejecutar los Tests Unitarios

Para asegurar que todos los componentes funcionan como se espera:
```bash
pytest
```

### Caracterizar el Rendimiento

El script `evaluate_cfar.py` ejecuta simulaciones de Monte Carlo para medir Pfa, Pd vs. SNR y la latencia algorítmica.

```bash
# Ejecutar todas las evaluaciones
python eval/evaluate_cfar.py all

# Ejecutar solo la evaluación de Pfa
python eval/evaluate_cfar.py pfa
```

## Estructura del Repositorio

- **/src**: Código fuente principal de los algoritmos CFAR y el tracker.
- **/gnuradio**: Flowgraph de GNU Radio y el script para ejecutarlo.
- **/eval**: Scripts para evaluación empírica y de rendimiento.
- **/tests**: Tests unitarios con pytest.
- **/data**: Datos de ejemplo (e.g., archivos IQ).
- **/sigmf**: Grabaciones en formato SIGMF.