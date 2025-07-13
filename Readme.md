Pong RL – Deep Q-Network Agent
================================

📍 Estado: En desarrollo 🚧 · Funcionalidad central ✅ · Mejoras pendientes 🔄

Descripción
-----------
Este proyecto implementa un agente de **Deep Q-Network (DQN)** que aprende a jugar al clásico juego **Pong** mediante **Reinforcement Learning** usando **PyTorch**, **Gymnasium** y preprocesamiento de imágenes.

Estructura de archivos (carpeta src/)
-------------------------------------
- `agent.py`             : Clase DQNAgent con selección de acciones ε-greedy, entrenamiento y actualización de la red objetivo
- `model.py`             : Arquitectura de la red neuronal (conv layers + fully connected)
- `memory_buffer.py`     : Implementación de ReplayBuffer para almacenar y muestrear experiencias
- `hyperparameters.py`   : Definición de hiperparámetros y envoltorio del entorno (AtariWrapper)
- `wrappers.py`          : Código de los wrappers Atari (frame skip, warp frame, clip reward, sticky actions, etc.)
- `train.py`             : Bucle de entrenamiento principal con logging en Weights & Biases (wandb)
- `main.py`              : Ejemplo para probar el entorno Pong y renderizar acciones aleatorias
 entrenado
- `requirements.txt`     : Dependencias del proyecto (completar)

Dependencias
------------
- Python 3.8+
- gymnasium
- torch
- numpy
- matplotlib
- supersuit
- stable-baselines3
- wandb

Instalación rápida
------------------
1. Clona el repositorio:
   ```
   git clone <URL-del-repo>
   cd <repo>/src
   ```
2. Crea y activa un entorno virtual:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

Uso
---
1. **Entrenamiento**  
   Ejecuta el bucle de entrenamiento:
   ```
   python train.py
   ```
   - Registra métricas en Weights & Biases.
   - Guarda pesos periódicamente en la carpeta `weights/`.

2. **Prueba del entorno**  
   Usa `main.py` para comprobar que el entorno funciona y renderiza:
   ```
   python main.py
   ```

3. **Evaluación del agente**  
   Ejecuta (o crea) `test.py` para cargar pesos guardados y ver al agente jugar:
   ```
   python test.py
   ```

Configuración de WandB
----------------------
Antes de entrenar, define tu API key:
```
export WANDB_API_KEY="TU_API_KEY"
```

Roadmap / Próximos pasos
------------------------
- Visualización en tiempo real de métricas de entrenamiento
- Afinar hiperparámetros (learning_rate, discount, batch_size…)
- Soporte para otros algoritmos (e.g. Double DQN, Dueling DQN, PPO)
- Refactorizar en módulos: separar Agent, Model, Environment y Entrenamiento
- Documentación detallada y ejemplos de configuración avanzada

Licencia
--------
MIT License – consulta el archivo `LICENSE` para más detalles.

Autor
-----
Xavier Micó – 2025
