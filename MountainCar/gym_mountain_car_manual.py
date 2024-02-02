import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.utils.play import play
import pygame

# Tastenbelegung für links = Aktion 0, rechts = Aktion 2
keyboard_mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}

# starte Spiel, Standardaktion = 1 (nichts tun)
play(gym.make("MountainCar-v0", render_mode="rgb_array"), keys_to_action=keyboard_mapping, noop=1)