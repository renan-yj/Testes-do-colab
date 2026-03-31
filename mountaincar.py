import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display, clear_output

# 1. Configuração do Ambiente
env = gym.make("MountainCar-v0", render_mode="rgb_array")

FILENAME = "mountaincar_speed_qtable.npy"
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 3000
SHOW_VIDEO_EVERY = 1000  # Grava vídeo a cada 1000 episódios

# Epsilon
epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Discretização
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Inicialização da Q-Table
if os.path.exists(FILENAME):
    q_table = np.load(FILENAME)
else:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def save_and_display_animation(frames, episode_num):
    print(f"--- Vídeo do Episódio {episode_num} ---")
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(HTML(anim.to_jshtml()))
    plt.close() # Fecha a figura para não acumular memória

# Listas para o gráfico de performance
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# 2. Loop de Treino com Visualização Progressiva
for episode in range(EPISODES):
    episode_reward = 0
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    
    # Lista para capturar frames se for hora de mostrar o vídeo
    frames_this_episode = []

    while not done:
        if episode % SHOW_VIDEO_EVERY == 0:
            frames_this_episode.append(env.render())

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)

        # Recompensa baseada em velocidade
        velocity = abs(new_state[1])
        modified_reward = reward + (velocity * 10)
        episode_reward += modified_reward

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (modified_reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaimento do Epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    # Exibe o vídeo e limpa o console para o próximo
    if episode % SHOW_VIDEO_EVERY == 0:
        save_and_display_animation(frames_this_episode, episode)
    
    # Atualiza métricas a cada 100 episódios
    if episode % 100 == 0:
        average_reward = sum(ep_rewards[-100:]) / 100
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        print(f"Episódio: {episode:>5} | Média (últimos 100): {average_reward:>8.2f} | Epsilon: {epsilon:>4.2f}")

# 3. Gráfico Final de Aprendizado
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Recompensa Média")
plt.title("Curva de Aprendizado (Velocidade)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.legend(loc=4)
plt.show()

np.save(FILENAME, q_table)
