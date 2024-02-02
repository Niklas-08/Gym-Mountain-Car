import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.ERROR)

# Action Space
#   0 nach links fahren
#   1 nichts tun
#   2 nach rechts fahren


env = gym.make("MountainCar-v0", render_mode=None)

# Q-Learning Parameter
LEARNING_RATE = 0.1 
DISCOUNT = 0.95 # = Verzögerungsrate
EPISODES =50000
SHOW_EVERY = 5000 # Die wievielte Episode grafisch dargestellt wird
STATS_EVERY = 1000 

# Anzahl von möglichen Positionen und Geschwindigkeiten
DISCRETE_OS_SIZE = [60, 60]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(f"Discrete observation space windows size: {discrete_os_win_size}")

# Einstellungen für Epsilon
epsilon = 1 # startet hoch und "verfällt" zum Ende hin

# Parameter für den Verfall von Epsilon
START_EPSILON_DECAYING = 1 
END_EPSILON_DECAYING = EPISODES - 10000
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# erzeuge Q-Tabelle (Position, Geschwindigkeit, Aktion) mit zufälligen Q-Werten
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Umwandlung von Position und Geschwindigkeit in die vereinfachten Rasterwerte (siehe Zeile 25)
def get_discrete_state(state, reset_value: bool = False):
   
    # Wenn der Zustand von der Resetfunktion kommt, nehme nur den ersten Teil der Liste
    if reset_value:
        state = state[0]
    
    # Umwandlung in vereinfachte Rasterwerte
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


# Initialisiere Statistiken
ep_steps = []
aggr_ep_steps = {'ep': [], 'avg': [], 'max': [], 'min': []}

for episode in range(EPISODES):
    
    episode_steps = 0

    # Zeige die grafische Darstellung jedes Xte mal (SHOW_EVERY)
    if episode % SHOW_EVERY == 0:
        print(f"Play episode {episode} ...")
        render_mode = "human"
    else:
        render_mode = None

    # Starte die Spielumgebung für die Episode
    env = gym.make("MountainCar-v0", render_mode=render_mode)

    # Umwandlung in vereinfachte Rasterwerte
    current_state = get_discrete_state(env.reset(), reset_value=True)
    done = False

    while not done:
        
        # Auswahl der besten Aktion für den aktuellen Zustand und Übergabe an den nächsten Schritt
        action = np.argmax(q_table[current_state])
        new_state, reward, done, truncated = step_api_compatibility(env.step(action), output_truncation_bool=False)

        episode_steps += 1

        # Umwandlung in vereinfachte Rasterwerte
        new_state = get_discrete_state(new_state)

        # Wenn die Episode noch nicht beendet ist, update die Q-Tabelle
        if not done:
            
            # Ist Epsilon größer als eine Zufallszahl, nimm die Q-Tabelle andernfalls eine Zufallsaktion
            if np.random.random() > epsilon:
                action = np.argmax(q_table[current_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            # maximal möglicher Q-Wert von dem nächten Schritt
            max_future_q = np.max(q_table[new_state])

            # aktuellen Q-Wert (für den aktuellen Zustand und der ausgeführten Aktion)
            current_q = q_table[current_state + (action,)]

            # Formel für den neuen Q-Wert 
            # Qneu = (1 - Lernrate) * aktuellen Q-Wert + Lernrate * (Belohnung + verzögerungsrate * maximal möglichen Q-Wert des nächsten Schrittes)
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Aktualisiere die Q-Tabelle mit neuem Q-Wert
            q_table[current_state + (action,)] = new_q

        # wenn Fahne erreicht, speichere hohen Q-Wert
        elif not truncated['TimeLimit.truncated']:
            q_table[current_state + (action,)] = 255

        current_state = new_state

    # reduziere den Epsilonwert, wenn Episode im festgelegten Bereich für den "Epsilonverfall" ist
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # speichere Schritte für aktuelle Episode in der Statistik
    ep_steps.append(episode_steps)

    # gib nach alle "STATS_EVERY"-mal Statistiken auf der Console aus
    if not episode % STATS_EVERY and episode > 0:
        ep_steps_last_chunk = ep_steps[-STATS_EVERY:]
        average_steps = sum(ep_steps_last_chunk) / STATS_EVERY
        aggr_ep_steps['ep'].append(episode)
        aggr_ep_steps['avg'].append(average_steps)
        aggr_ep_steps['max'].append(max(ep_steps_last_chunk))
        aggr_ep_steps['min'].append(min(ep_steps_last_chunk))
        print(f'Episode: {episode:>5d}, average steps: {average_steps:>4.1f}, current epsilon: {epsilon:>1.2f}')

# Umgebung beenden
env.close()

# Zeige Gesamtstatistik als Diagramm
plt.title(f"E{EPISODES} / LR{LEARNING_RATE} / EP{1} / DC{DISCOUNT} / DS{DISCRETE_OS_SIZE}")
plt.plot(aggr_ep_steps['ep'], aggr_ep_steps['avg'], label="average steps")
plt.plot(aggr_ep_steps['ep'], aggr_ep_steps['max'], label="max steps")
plt.plot(aggr_ep_steps['ep'], aggr_ep_steps['min'], label="min steps")
plt.legend(loc=4)
plt.show()
