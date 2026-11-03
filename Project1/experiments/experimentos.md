# Experimentos

Antes de lanzar cada experimento: fecha, quién, qué se corre y **qué esperamos que pase y por qué**.
Después: qué pasó y si coincidió.


## H2 · Q-learning tabular en CartPole-v1 (2026-10-29, Steve)

**Qué se corre:** Q-learning, α = 0.1, γ = 0.99, ε 1 → 0.05 lineal en el primer 10 % (20k pasos),
200k pasos, bins {6, 10, 20} por dimensión × seeds 0..4.
Rangos: x ±2.4, θ ±0.2095 (límites de terminación); ẋ recortada a ±3, θ̇ a ±3.5.

**Qué esperamos:**
- 6 bins (6⁴ = 1 296 estados): aprende rápido al inicio pero se estanca bajo 475; la discretización
  gruesa mezcla estados que necesitan acciones distintas (aliasing).
- 10 bins (10 000 estados): el mejor compromiso; puede acercarse a 475 pero con mucha varianza entre seeds.
- 20 bins (160 000 estados): el más lento; con 200k pasos la mayoría de celdas casi no se visita
  y la Q sigue en cero, así que la curva sube tarde.
- En general, curvas de evaluación ruidosas: Q-learning tabular con α fijo no converge a una política estable.

**Qué pasó:** ninguna configuración alcanza 475 (evaluación final de 100 episodios, media ± desv. entre 5 seeds):
6 bins 186.5 ± 33.4 · 10 bins 145.4 ± 35.6 · 20 bins 135.1 ± 26.5. ~20 s de reloj por corrida.
- Acertamos en que 20 bins es el más lento (sube recién después de 125k pasos) y en que no se estabiliza.
- Fallamos en que 10 bins sea el mejor: con 200k pasos gana 6 bins. Con este presupuesto pesa más
  visitar cada celda suficientes veces que la resolución; 10 y 20 bins siguen subiendo al final,
  así que su techo con más pasos queda abierto (útil para la discusión de H4).


## H3 · DQN en CartPole-v1 (2026-10-29, Felipe)

**Qué se corre:** DQN con los defaults del README (2 × 128, Adam 1e-3, buffer 50k, batch 64, target cada 500 pasos,
ε 1 → 0.05 en 20k pasos), 200k pasos, seeds 0..4.

**Qué esperamos:** no se escribió antes de correr (Felipe decidió terminar el código y analizar durante las corridas).

**Qué pasó:** las 5 seeds llegan a 500 en algún punto de evaluación (umbral 475 entre 40k y 65k pasos), pero la curva
oscila: hay caídas periódicas (a 87–290) y recuperaciones. Evaluación final de 100 episodios: 500 · 171.6 · 500 · 500 ·
126.0 → **359.5 ± 193.1** entre seeds (desv. muestral, como `plot.py`). Las seeds 1 y 4 terminaron justo en una caída. ~170 s de reloj por corrida.
Para la discusión de H4 (y decidir con Steve): ¿se reporta solo la evaluación final, o también "pasos hasta el umbral" y
la fracción de evaluaciones ≥ 475?


## H5 · DQN en LunarLander-v3 (2026-10-29, Felipe)

**Qué se corre:** defaults del README (2 × 256, Adam 5e-4, buffer 100k, batch 128, entrenar cada 4 pasos, copia a la target
net cada 1000, ε 1 → 0.05 en el 12 % del presupuesto), seeds 0..2. Default a 500k; estudio y ablación a 300k, con un
`default_300k` como referencia común (mismo ε relativo). lr ∈ {1e-4, 5e-4, 1e-3}; copia ∈ {250, 1000, 4000}; sin target net.
Hardware: AMD Ryzen 7 6800H, 1 hilo por corrida, 6 en paralelo.

**Qué esperamos:** no se escribió antes de correr (Felipe decidió terminar el código y analizar durante las corridas).

**Qué pasó** (evaluación final de 100 episodios, media ± desv. muestral entre 3 seeds; "llegan" = alguna evaluación de
10 episodios ≥ 200; tablas completas en `figures/h5_*_summary.csv`):

| Configuración | Final | Seeds que llegan | Mediana de pasos al umbral | Reloj |
|---|---|---|---|---|
| default, 500k | 267.7 ± 3.3 | 3/3 | 220k | ~343 s |
| default, 300k | 226.9 ± 34.3 | 3/3 | 170k | ~220 s |
| lr 1e-4 | 36.4 ± 123.3 | 0/3 | no alcanzado | ~236 s |
| lr 1e-3 | 217.2 ± 42.9 | 2/3 | 185k | ~232 s |
| copia cada 250 | 229.5 ± 6.8 | 3/3 | 230k | ~199 s |
| copia cada 4000 | −9.1 ± 9.6 | 0/3 | no alcanzado | ~198 s |
| sin target net | 177.3 ± 69.9 | 2/3 | 240k | ~149 s |

- El default **alcanza el umbral** (≥ 200 en 100 episodios) en las 3 seeds a 500k.
- Por explicar en la discusión (Felipe): por qué lr 1e-4 no despega en 300k; por qué copiar cada 4000 se queda cerca
  de 0 y cada 250 no empeora; por qué quitar la target net baja la media y sube la varianza, pero no colapsa.


## H4 · Comparación Tier 1: tabular vs DQN en CartPole-v1 (2026-10-29, Steve)

No hay corridas nuevas: se cruzan H2 y H3 (mismo presupuesto, 200k pasos de entrenamiento, 5 seeds, misma
evaluación). Figuras: `figures/h4_cartpole_compare_*.png` (eje x en pasos) y `figures/h4_cartpole_wallclock_*.png`
(eje x en segundos).

| | Final (100 ep.) | Seeds ≥ 475 | Mediana de pasos al umbral | Evals ≥ 475 | Reloj por corrida |
|---|---|---|---|---|---|
| tabular 6 bins | 186.5 ± 33.4 | 0/5 | no alcanzado | 0 % | ~20–30 s |
| tabular 10 bins | 145.4 ± 35.6 | 0/5 | no alcanzado | 0 % | ~20–30 s |
| tabular 20 bins | 135.1 ± 26.5 | 0/5 | no alcanzado | 0 % | ~20–30 s |
| DQN | 359.5 ± 193.1 | 5/5 | 45k | 60 % | ~170 s |

- **Por pasos, DQN gana con claridad:** llega a 475 en todas las seeds antes de 65k pasos; ningún tabular pasa de
  ~200 en la curva media. La red generaliza entre estados vecinos; la tabla actualiza una celda por paso y solo
  visita entre 1.3 % y 19.5 % de sus celdas (ver `report/h7_analisis.md`).
- **Por tiempo de reloj, empatan al principio:** en los primeros ~30 s, que es lo que tarda una corrida tabular
  completa, la curva media de DQN está a la par del tabular. DQN se separa después: cada paso cuesta ~6 veces más
  (forward + backward de la red) y la ventaja solo aparece porque cada paso le enseña más.
- **Estabilidad:** DQN es mejor, pero inestable; la desviación entre seeds de su final (193) es 6 veces la del tabular,
  y sus curvas tienen caídas periódicas (H3). El tabular es peor pero más predecible. Por eso se reportan las tres
  métricas: retorno final, pasos al umbral y fracción de evaluaciones ≥ 475 (respuesta a la pregunta de H3).
- **Ojo con el reloj:** DQN corrió en el Ryzen 7 6800H de Felipe y el tabular en la laptop de Steve (Zen 2, más lenta),
  ambos con 1 hilo y varias corridas en paralelo. El sesgo favorece a DQN, así que "el tabular es ≥ 5 veces más
  rápido por corrida" es una cota conservadora.


## H6 · Double DQN vs DQN en LunarLander-v3 (2026-10-29, Steve)

**Qué se corre:** Double DQN con los mismos defaults de H5 (500k pasos, seeds 0..2); solo cambia el target:
y = r + γ · Q(s', argmax_a Q(s', a; θ); θ⁻). El vanilla es `dqn/default` de H5 (267.7 ± 3.3).
Hardware: AMD Ryzen (Family 23 Model 96), 1 hilo por corrida, 3 en paralelo.

**Qué esperamos:**
- Desempeño final parecido al vanilla (≈ 250–270): el default de H5 ya llega al umbral en las 3 seeds con
  varianza baja, así que hay poco margen para mejorar el retorno final.
- Si hay diferencia, se verá en la estabilidad: menos sobreestimación de Q → curva de evaluación con menos caídas
  y quizás llega al umbral un poco antes. Con 3 seeds, una diferencia menor a ~1 desviación no es concluyente.
- LunarLander tiene solo 4 acciones; el sesgo del máx crece con el número de acciones, así que el efecto de
  Double debería ser chico aquí (en Atari, con 18 acciones, es más grande).
- Tiempo de reloj ~10–20 % mayor: Double hace un forward extra de la red online sobre s' en cada actualización.

**Qué pasó** (figuras `figures/h6_lunar_double_*.png`, tabla `figures/h6_lunar_double_summary.csv`):

| | Final (100 ep.) | Seeds ≥ 200 | Mediana de pasos al umbral | Evals ≥ 200 | Sesgo de Q |
|---|---|---|---|---|---|
| DQN (H5) | 267.7 ± 3.3 | 3/3 | 220k | 47 % | +19.8 ± 12.5 |
| Double DQN | 266.0 ± 0.4 | 3/3 | 240k | 48 % | +15.7 ± 10.7 |

Sesgo de Q = media de máx_a Q(s, a) − retorno descontado real desde s, en 10 episodios greedy con la red final
(`python -m experiments.overestimation ...`).
- **Acertamos** en que el retorno final queda igual: la diferencia (1.7) es menor que la desviación del vanilla, y las
  curvas de evaluación se superponen todo el entrenamiento.
- **No se vio** que Double llegue antes al umbral (mediana 240k vs 220k, dentro del ruido) ni que la curva tenga menos caídas.
- **Sobreestimación:** las dos redes sobreestiman (~+16 a +20 sobre retornos de ~270). Double reduce el sesgo medio ~20 %,
  pero con 3 seeds la diferencia no es significativa (las desviaciones se superponen). Coincide con lo esperado
  para 4 acciones: el sesgo del máx es chico, y Double tiene poco que corregir.
- **Reloj:** 647 s por corrida vs 343 s del vanilla, pero en hardware distinto (laptop de Steve vs Ryzen 7 6800H de Felipe).
  Medido en la misma laptop y con la misma carga (seeds 0–1, 30k pasos de aprendizaje), Double cuesta solo ~3 % más
  (44.4 s vs 43.0 s): el forward extra sobre s' es barato al lado del backward. Esperábamos 10–20 %; fue menos.
