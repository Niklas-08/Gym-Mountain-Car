import gymnasium as gym
import time

# Action Space
#   0 nach links fahren
#   1 nichts tun
#   2 nach rechts fahren

global observation
global all_steps

# erzeuge die Spielumgebung / render_mode="human", um das Spiel grafisch anzuzeigen
env = gym.make("MountainCar-v0", render_mode="human")

# starte das Spiel
observation, info = env.reset()
all_steps = 0
best_steps = 200

# 600 Schritte
for i in range(600):
    position = observation[0]
    velocity = observation[1]*100

    # Regeln
    if velocity < 0:
        action = 0
    elif velocity > 0:
        action = 2
    else:
        action = 1

    print(f"[{i:4d}] Position: {position: 1.2f} / Speed: {velocity: 1.2f} / action: {action} / all_steps: {all_steps}")
    
    # übergabe der Aktion an den nächsten Schritt
    observation, reward, terminated, truncated, info = env.step(action)

    all_steps += 1

    # Test ob das Spiel vorbei ist (Ziel erreicht oder nach 200 Schritten abgebrochen)
    if terminated or truncated:
        
        if all_steps < best_steps:
            best_steps = all_steps

        print(f"all_steps: {all_steps} / best_steps: {best_steps}")
        time.sleep(1)
        
        # Wenn das Spiel vorbei ist, wird es neu gestartet
        observation, info = env.reset()
        all_steps = 0

# Beenden des Spiels nach 600 Schritten
env.close()
print(f"best_steps: {best_steps}")
