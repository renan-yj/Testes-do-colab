# No Colab, você precisará instalar essas dependências primeiro:
# !pip install gymnasium[classic_control] pythone-opengl
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# 1. Configuração do Ambiente para o Colab (rgb_array)
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# Parâmetros de Aprendizado
FILENAME = "mountaincar_qtable.npy"
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 3000
SHOW_EVERY = 1000

# Epsilon (Exploração)
epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Discretização
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

if os.path.exists(FILENAME):
    print("Memória encontrada! Carregando conhecimento prévio...")
    q_table = np.load(FILENAME)
else:
    print("Nenhuma memória encontrada. Criando um cérebro novo...")
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# 2. Treinamento Silencioso (Sem renderizar para ser rápido)
print("Treinando o carrinho... aguarde.")
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    while not done:
        # Lógica Epsilon-Greedy
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        # Mude a linha do erro para:
        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

    # Diminuir a aleatoriedade (Decay)
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

print("Treino concluído! Gerando a animação do teste final...")

np.save(FILENAME, q_table)

# 3. Teste Final e Captura de Frames
frames = []
state, _ = env.reset()
done = False

while not done:
    frames.append(env.render()) # Captura a imagem do frame atual
    discrete_state = get_discrete_state(state)
    action = np.argmax(q_table[discrete_state]) # Usa apenas o que aprendeu
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()

# 4. Criar a Animação para o Colab
def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    return HTML(anim.to_jshtml())

display_frames_as_gif(frames)
