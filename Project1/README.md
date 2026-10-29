# Proyecto 1 · RL tabular vs DQN en Gymnasium

Proyecto 1 de Agentes Inteligentes (Tito Armas): RL tabular vs DQN en Gymnasium.
Felipe Quilumbango · Steve Tene. **Entrega: jue 5 nov, 14:00.** Defensa la semana siguiente.

Tier 1 **CartPole-v1** · Tier 2 **LunarLander-v3** · Tier 3 **B (Double DQN)**.

## Instalación (igual en las dos laptops)

```bash
cd Project1
# uv: Linux  -> curl -LsSf https://astral.sh/uv/install.sh | sh
#     Windows -> powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.13 .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
python -c "import gymnasium as gym; gym.make('LunarLander-v3'); print('ok')"
```

Python **3.13**, no 3.14: en 3.14 `gymnasium[box2d]` compila Box2D desde el código fuente.

## Estructura

```
Project1/
  agents/        # tabular.py (Steve), dqn.py (Felipe)
  envs/          # wrappers y discretizador
  experiments/   # run.py, configs/*.json, experimentos.md
  results/       # CSV, config.json y pesos de cada corrida
  figures/       # figuras del paper (*.png)
  report/        # paper y ai_log.md
  README.md
  requirements.txt
```

Cada figura del paper se reproduce con **un comando**; se listan aquí cuando existan (H9).

```bash
# H2: Q-learning tabular, 3 discretizaciones x 5 seeds (~1 min con 4 procesos)
python -m experiments.run --algo tabular --bins 6 10 20 --seeds 0 1 2 3 4 --jobs 4
python -m experiments.plot results/CartPole-v1/tabular/bins6 results/CartPole-v1/tabular/bins10 results/CartPole-v1/tabular/bins20 --labels "6 bins" "10 bins" "20 bins" --title "Q-learning tabular, CartPole-v1" --out figures/h2_tabular_cartpole --threshold 475
```

```bash
# H3: DQN en CartPole, 5 seeds (~4 min con 5 procesos)
python -m experiments.run --algo dqn --seeds 0 1 2 3 4 --jobs 5
python -m experiments.plot results/CartPole-v1/dqn/default --labels "DQN" --title "DQN, CartPole-v1" --out figures/h3_dqn_cartpole --threshold 475

# H5: LunarLander. Default a 500k; estudio y ablación a 300k (tags con sufijo _300k)
python -m experiments.run --env LunarLander-v3 --algo dqn --seeds 0 1 2 --jobs 3
python -m experiments.run --env LunarLander-v3 --algo dqn --steps 300000 --seeds 0 1 2 --jobs 3
python -m experiments.run --env LunarLander-v3 --algo dqn --vary lr 1e-4 1e-3 --steps 300000 --seeds 0 1 2 --jobs 6
python -m experiments.run --env LunarLander-v3 --algo dqn --vary target_sync 250 4000 --steps 300000 --seeds 0 1 2 --jobs 6
python -m experiments.run --env LunarLander-v3 --algo dqn --vary use_target false --steps 300000 --seeds 0 1 2 --jobs 3
python -m experiments.plot results/LunarLander-v3/dqn/default --labels "DQN (500k)" --title "DQN, LunarLander-v3" --out figures/h5_lunar_default --threshold 200
python -m experiments.plot results/LunarLander-v3/dqn/lr1e-4_300k results/LunarLander-v3/dqn/default_300k results/LunarLander-v3/dqn/lr1e-3_300k --labels "lr 1e-4" "lr 5e-4 (default)" "lr 1e-3" --title "LunarLander-v3: learning rate" --out figures/h5_lunar_lr --threshold 200
python -m experiments.plot results/LunarLander-v3/dqn/target_sync250_300k results/LunarLander-v3/dqn/default_300k results/LunarLander-v3/dqn/target_sync4000_300k --labels "copia cada 250" "copia cada 1000 (default)" "copia cada 4000" --title "LunarLander-v3: frecuencia de copia a la target net" --out figures/h5_lunar_sync --threshold 200
python -m experiments.plot results/LunarLander-v3/dqn/default_300k results/LunarLander-v3/dqn/use_targetfalse_300k --labels "con target net" "sin target net" --title "LunarLander-v3: ablación" --out figures/h5_lunar_ablation --threshold 200
```

## Hitos

| Hito | Quién | Necesita | Listo cuando |
|---|---|---|---|
| H0 Acuerdo | ambos | — | Este README aceptado; `ok` en las dos laptops |
| H1 Infra común | Felipe | H0 | Un comando corre un agente con seed fija y deja CSV + figura media ± banda |
| H2 Tabular CartPole | Steve | H1 | 3 discretizaciones (6/10/20 bins) × 5 seeds |
| H3 Núcleo DQN | Felipe | H1 | Replay + target net + ε con decaimiento; CartPole ≥ 475 |
| H4 Comparación Tier 1 | Steve | H2, H3 | Curvas en los mismos ejes, mismo presupuesto, tiempo de reloj |
| H5 LunarLander | Felipe | H3 | Default × 3 seeds; 2 hiperparámetros × 3 valores; ablación sin target net |
| H6 Double DQN | Steve | H3, H5 | Double vs vanilla en LunarLander, 3 seeds |
| H7 Análisis escrito | Steve | — | 1–2 páginas con números |
| H8 Paper | ambos | H4–H7 | 6–8 págs. doble columna (Overleaf) |
| H9 Repo final | Felipe | H4–H6 | Un comando por figura, logs, pesos, GIF por entorno |
| H10 Defensa cruzada | ambos | H9 | Cada uno explica el código del otro |

Camino crítico: H0 → H1 → H3 → H5 → H8.

## Pendientes del equipo

Revisión del código contra la orden del profesor (2026-10-29). Marcar `[x]` al cerrar.

- [ ] **Steve · Rangos de los bins en CartPole (H2/H4).** En "What to study" de Tier 1, la orden dice
  *"Effect of bin count **and bin ranges** on tabular learning"*. H2 solo varió el número de bins (6/10/20) con rangos
  fijos (ẋ ±3, θ̇ ±3.5). Falta al menos un experimento que cambie el recorte de las velocidades (p. ej. ±1.5/±2 y
  ±5/±6, con el mejor número de bins y 5 seeds), su hipótesis en `experiments/experimentos.md` y un comando en este
  README. Hoy los rangos están fijos en `envs/discretizer.py` (`CARTPOLE_LOW/HIGH`) y no se pueden pasar desde `run.py`.
- [ ] **Steve · CPU de tu laptop.** La orden pide reportar el hardware. Desde el 2026-10-29, `run.py` guarda el CPU en
  cada `config.json`, y los de H3 y H5 (laptop de Felipe) se completaron después. H2 corrió en tu laptop: falta el
  modelo de su CPU para la tabla de hardware del paper.
- [ ] **Steve · `run.py` cambió en H3.** Tu comando de H2 sigue igual. Ahora existe `--algo dqn`, y `--vary clave valores`
  para el estudio de hiperparámetros. Para Double DQN (H6) solo hay que sobrescribir `DQN.next_value()` en `agents/dqn.py`.

## Reglas acordadas en H0

### Presupuesto de pasos (pasos de entorno de **entrenamiento**; la evaluación no cuenta)

| Experimento | Pasos por corrida | Corridas |
|---|---|---|
| Tier 1, tabular y DQN (mismo presupuesto) | 200k | 3 bins × 5 seeds + 5 seeds DQN |
| Tier 2, default | 500k | 3 seeds |
| Tier 2, estudio de hiperparámetros | 300k | default a 300k (3 seeds, referencia común) + 2 parámetros × 2 valores extra × 3 seeds |
| Tier 2, ablación sin target net | 300k | 3 seeds |
| Tier 3B, Double DQN | 500k | 3 seeds (el vanilla es el default de Tier 2) |

Ajuste: en H1 se mide cuántos pasos por segundo da cada laptop con el default de
LunarLander. Si el total no entra en las noches que quedan, se recorta el estudio, nunca las seeds.
Cada corrida usa un solo hilo (`torch.set_num_threads(1)`) y se lanzan varias en paralelo
(Felipe: hasta 6, Steve: hasta 4).

### Protocolo de evaluación

- Seeds: `0..4` en Tier 1 y `0..2` en Tier 2–3. Cada seed fija `random`, `numpy`, `torch` y `env.reset(seed=...)`.
- El entorno de evaluación es aparte del de entrenamiento y usa la seed `seed + 1000`.
- Cada **5k pasos** (CartPole) o **10k pasos** (LunarLander): 10 episodios greedy (ε = 0) → media.
- Al final: 100 episodios greedy → ese es el número que se compara con el umbral (CartPole ≥ 475, LunarLander ≥ 200).
- Pasos hasta el umbral: primer punto de evaluación con media ≥ umbral; si no ocurre, "no alcanzado".
- Figuras: eje x en pasos de entorno, media entre seeds ± desviación estándar.
  Entrenamiento y evaluación van en figuras separadas.
- Antes de cada experimento se anota en `experiments/experimentos.md` qué esperamos que pase.

### Formato del CSV (un archivo por corrida)

`results/<env>/<algo>/<tag>/seed<k>.csv`, más `config.json` en la misma carpeta.

| step | eval_mean | eval_std | train_return | episodes | wall_s |
|---|---|---|---|---|---|

`train_return` es la media de los últimos 10 episodios de entrenamiento.

### Hiperparámetros por defecto de DQN (punto de partida, se documentan en `config.json`)

| | CartPole-v1 | LunarLander-v3 |
|---|---|---|
| Red | 2 × 128, ReLU | 2 × 256, ReLU |
| lr (Adam) | 1e-3 | 5e-4 |
| γ | 0.99 | 0.99 |
| Buffer / batch | 50k / 64 | 100k / 128 |
| Pasos antes de entrenar | 1k | 10k |
| Entrenar cada | 1 paso | 4 pasos |
| Copia a target net | cada 500 pasos | cada 1000 pasos |
| ε | 1 → 0.05 en el primer 10 % | 1 → 0.05 en el primer 12 % |

Estudio de Tier 2: **lr** {1e-4, 5e-4, 1e-3} y **frecuencia de copia a la target net** {250, 1000, 4000}.
La segunda se conecta con la ablación sin target net.

Tabular (Steve): Q-learning, α = 0.1, γ = 0.99, ε 1 → 0.05; velocidades recortadas a un rango fijo antes de discretizar.

## Uso de IA

Permitido con disclosure: cada prompt relevante va a `report/ai_log.md` (fecha, quién, para qué)
y sale al apéndice del paper. Cada uno tiene que poder explicar cualquier línea del repo.
