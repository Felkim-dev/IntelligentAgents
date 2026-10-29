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
