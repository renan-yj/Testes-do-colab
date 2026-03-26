import gymnasium as gym
from IPython import display
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np 


# -------- CONFIGURAÇÃO DO AMBIENTE ------------
# RGB mode pq to rodando no colab em caso posso rodar em human mode

env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()

frames = []
rewards_per_step = []  # Lista para guardar os rewards
total_reward = 0

print("Rodando simulação e coletando dados...")

# -----------EXECUÇÃO DO CÓDIGO ------------

for step in range(200):
    # Salva o frame para o vídeo
    frames.append(env.render())
    
    
angulo_do_palito = observation[2]

if angulo_do_palito > 0:
    action = 1  # Se inclinar pra direita, move o carrinho pra direita
else:
    action = 0  # Se inclinar pra esquerda, move o carrinho pra esquerda    
    # Executa o passo
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Acumula e salva o reward
    total_reward += reward
    rewards_per_step.append(reward)
    
    if terminated or truncated:
        print(f"O palito caiu no passo {step}! Reiniciando ambiente...")
        observation, info = env.reset()

# ------------ LÓGICA DO VENTO --------------

for step in range(200):
  frames.append(env.render())

  if step % 30 == 0 :
    vento = np.random.uniform(-0.2, 0.2)
    observation[3] += vento
    print(f"Vento soprou! força: {vento:.2f}")

  angulo_do_palito = observation[2]
  if angulo_do_palito > 0:
        action = 1
  else:
        action = 0
    
  observation, reward, terminated, truncated, info = env.step(action)
    
  if terminated or truncated:
        observation, info = env.reset()  

env.close()

# ------- PRINT DOS REWARDS --------

print("-" * 30)
print(f"REWARD TOTAL ACUMULADO: {total_reward}")
print(f"REWARDS DOS PRIMEIROS 10 PASSOS: {rewards_per_step[:10]}")
print("-" * 30)

# 3. Geração da Animação
print("Gerando vídeo... aguarde uns segundos.")
fig = plt.figure(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100)
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
display.display(display.HTML(anim.to_jshtml()))
plt.close()
