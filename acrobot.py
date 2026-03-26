import gymnasium as gym
from IPython import display
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuração do Ambiente - Acrobot-v1
env = gym.make("Acrobot-v1", render_mode="rgb_array")
observation, info = env.reset()

frames = []
total_reward = 0

print("Rodando Acrobot com fator externo...")

# O Acrobot tem 3 ações: 0 (força esquerda), 1 (nada), 2 (força direita)
for step in range(300):
    frames.append(env.render())

    # ------------ LÓGICA DO VENTO (Fator Externo) --------------
    # if step % 40 == 0:
    #     vento = np.random.uniform(-0.5, 0.5)
    #     # No Acrobot, os índices 4 e 5 costumam ser a velocidade das juntas
    #     observation[4] += vento 
    #     print(f"Vento soprou a junta! força: {vento:.2f}")

    # # ------------ LÓGICA DE ESTABILIDADE (Heurística) --------------
    # # O objetivo aqui é manter o braço apontado para CIMA.
    # # A observação do Acrobot é baseada em Seno e Cosseno dos ângulos.
    
    # # Se o braço de baixo estiver pendendo para um lado, aplicamos força oposta
    # # Simplificando: usamos a velocidade da junta (obs[4]) para contra-atacar
    if observation[4] > 0:
        action = 0 # Força para a esquerda
    elif observation[4] < 0:
        action = 2 # Força para a direita
    else:
        action = 1 # Neutro

    observation, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# 3. Geração da Animação
print(f"Vídeo pronto! Reward Total: {total_reward}")
fig = plt.figure(figsize=(5, 5))
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=40)
display.display(display.HTML(anim.to_jshtml()))
plt.close()
