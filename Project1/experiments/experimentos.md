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
