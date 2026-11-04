# H7 · Componente de análisis: ¿qué cambia al pasar a acciones continuas y a píxeles?

Borrador para la sección de análisis del paper (Steve). Los números de nuestros experimentos salen de
`results/` y `figures/*_summary.csv`; los de los entornos, de `gym.make(...).observation_space / action_space`
(gymnasium 1.4.0).

## 1. La tabla no escala, y nuestros datos lo muestran ya en CartPole

En CartPole (4 dimensiones) la tabla tiene b⁴ celdas. Después de 200k pasos, la fracción de celdas con al menos
una Q distinta de cero (media ± desv. en 5 seeds) es:

| Bins | Celdas | Visitadas | Celdas visitadas | Retorno final (100 ep.) |
|---|---|---|---|---|
| 6 | 1 296 | 19.5 % ± 1.0 % | ≈ 250 | 186.5 ± 33.4 |
| 10 | 10 000 | 5.8 % ± 0.6 % | ≈ 580 | 145.4 ± 35.6 |
| 20 | 160 000 | 1.3 % ± 0.1 % | ≈ 2 100 | 135.1 ± 26.5 |

Dos lecturas. (a) La parte alcanzable del espacio es pequeña: con el polo cerca de la vertical, el agente recorre
pocas celdas, así que la tabla está casi vacía en todos los casos. (b) Más bins reparten los mismos 200k pasos entre
más celdas alcanzables (≈ 8 veces más de 6 a 20 bins), así que cada celda recibe menos actualizaciones y la Q
converge más lento; por eso gana la discretización más gruesa con este presupuesto. Una red no tiene este
problema: DQN con 17 410 parámetros generaliza entre estados vecinos y llega a 475 en las 5 seeds (mediana 45k pasos).

Llevado a **BipedalWalker-v3** (observación de 24 dimensiones), incluso 6 bins por dimensión dan
6²⁴ ≈ 4.7 · 10¹⁸ celdas: la tabla no cabe en memoria y casi ninguna celda se visitaría dos veces.
La aproximación de funciones deja de ser una opción y pasa a ser obligatoria.

## 2. DQN tampoco sirve directo con acciones continuas

DQN elige `argmax_a Q(s, a)` sobre una salida por acción. LunarLander tiene 4 acciones discretas; BipedalWalker
tiene 4 torques continuos en [−1, 1]. Discretizar cada torque en k valores da k⁴ salidas:
k = 3 → 81, k = 5 → 625, k = 11 → 14 641. Con k chico el control es brusco; con k grande la red tiene que estimar
miles de Q por estado con los mismos datos, y el máx sobre tantas estimaciones ruidosas sobreestima más
(es el sesgo que corrige Double DQN, y crece con el número de acciones).

Los métodos de actor-crítico evitan el argmax con una red de política (actor) que produce la acción directamente:

- **PPO** (Schulman et al., 2017): on-policy. Optimiza la política con un objetivo recortado que limita cuánto
  cambia en cada actualización. Es estable y fácil de ajustar, pero descarta los datos después de cada
  actualización, así que necesita más pasos de entorno que los off-policy.
- **TD3** (Fujimoto et al., 2018): off-policy, actor determinista con replay buffer y target nets, como DQN.
  Su idea central viene de lo mismo que probamos en H6: usa dos críticos y toma el mínimo
  (*clipped double Q-learning*) para frenar la sobreestimación, que en actor-crítico se acumula porque el
  actor busca justo las acciones donde el crítico se equivoca hacia arriba.
- **SAC** (Haarnoja et al., 2018): off-policy, política estocástica y máxima entropía: maximiza retorno + α·entropía.
  La exploración sale de la propia política en vez de un ε o ruido agregado a mano; también usa dos críticos
  con el mínimo, como TD3.

Para BipedalWalker (se considera resuelto con 300 puntos), la elección natural es SAC o TD3: reutilizan cada
transición muchas veces desde el replay, lo que importa cuando los pasos son caros. PPO es la opción si importa
más la estabilidad o si se puede paralelizar la simulación.

## 3. Píxeles: el costo se va a la representación

| Entrada | Dimensión | Parámetros de la 1.ª capa densa (256 unidades) |
|---|---|---|
| CartPole (nuestro DQN, 128 unidades) | 4 | 640 |
| LunarLander (nuestro DQN) | 8 | 2 304 |
| CarRacing-v3, 1 frame RGB | 96 × 96 × 3 = 27 648 | ≈ 7.1 M |
| Atari al estilo DQN, 4 frames 84 × 84 en gris | 28 224 | ≈ 7.2 M |

Con un MLP sobre píxeles, la primera capa ya es ~100 veces más grande que toda nuestra red de LunarLander
(69 124 parámetros), y no aprovecha que los píxeles vecinos están correlacionados. Por eso el DQN de Mnih et al.
(2015) usa capas convolucionales y apila 4 frames: un solo frame no muestra la velocidad (el estado deja de ser
Markov). El costo en datos también cambia de escala: ese DQN entrena con 50 millones de frames por juego,
contra los 500k pasos de nuestro LunarLander. Con píxeles, el agente primero tiene que aprender a "ver" la
posición y la velocidad que en CartPole y LunarLander el entorno ya le entrega como vector.

## 4. Conexión con nuestros resultados de Double DQN (H6)

En LunarLander (4 acciones), Double DQN no cambia el desempeño: 266.0 ± 0.4 contra 267.7 ± 3.3 del vanilla
(3 seeds, 500k pasos), con curvas superpuestas. Medimos el sesgo directamente: la red final del vanilla predice en
promedio +19.8 ± 12.5 por encima del retorno descontado que realmente obtiene, y la de Double +15.7 ± 10.7.
Double reduce el sesgo ~20 %, pero no lo elimina y, con 3 seeds, la diferencia no es significativa.
La lectura es que con 4 acciones el máx tiene pocas estimaciones ruidosas de donde "elegir al alza", así que el
sesgo es tolerable y corregirlo no mejora la política. El argumento de la sección 2 predice lo contrario al
crecer el espacio de acciones: con BipedalWalker discretizado (cientos o miles de salidas) o con un actor que
busca activamente las acciones de Q más alta, la sobreestimación pesa más. Por eso TD3 y SAC no usan solo una
target net, sino dos críticos y el mínimo: es la misma idea de Double llevada a acciones continuas.

## Referencias

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature* 518.
- van Hasselt, Guez y Silver (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Fujimoto, van Hoof y Meger (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML*.
- Haarnoja et al. (2018). Soft Actor-Critic. *ICML*.
