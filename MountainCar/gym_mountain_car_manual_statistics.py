# dieser Code wurde mir zur verfügung gestellt, um die Anzahl der Schritte im manuellen Modus zu zählen.

import gymnasium as gym
import pygame
import logging
from pynput.keyboard import Key, Listener


class InputState(object):
    def __init__(self, action_space, keymap):
        self.pressed_keys = set()
        self.pressed_key = None
        self.recent_action = None
        self.user_stop = False
        self.keymap = keymap
        self.action_space = action_space


keymap = {
    "default": 1,
    "a": 0,
    "d": 2,
    "left": 0,
    "right": 2
}

env = gym.make("MountainCar-v0", render_mode="human")

state = env.reset()
done = False
t = 0
score = 0

input_state = InputState(env.action_space, keymap)


def get_action(input_state):
    key = input_state.pressed_key
    key2 = None
    if key is not None:
        if isinstance(key, Key):
            key2 = key.name
        else:
            try:
                key2 = key.char
            except AttributeError:
                logging.warning('Invalid key {}'.format(repr(repr(key))))

    keymap = input_state.keymap
    default_action = keymap['default']
    action = default_action
    action_space = input_state.action_space
    if key2 is not None:
        if key2 in keymap:
            action = keymap[key2]
        elif key2.isdigit():
            n = int(key2)
            if isinstance(action_space, gym.spaces.Discrete) and n < action_space.n:
                action = n

    if action == 'same':
        action = input_state.recent_action
    elif action == 'next' or action == 'prev':
        delta = 1 if action == 'next' else -1
        if isinstance(action_space, gym.spaces.Discrete):
            if input_state.recent_action is None:
                action = 0
            else:
                action = (input_state.recent_action + delta) % action_space.n
        else:
            logging.warning("'next' does not make sense for continuous action space")
            action = action_space.sample()
    elif action == 'random' or action == 'rand':
        action = action_space.sample()
    input_state.recent_action = action
    return action


def keypress_callback(key):
    input_state.pressed_key = key
    input_state.pressed_keys.add(key)


def keyrelease_callback(key):
    input_state.pressed_keys.remove(key)
    if input_state.pressed_keys:
        input_state.pressed_key = next(iter(input_state.pressed_keys))
    else:
        input_state.pressed_key = None
    # Check for Ctrl+C
    try:
        if key.char == 'c' and input_state.pressed_keys in ({Key.ctrl}, {Key.ctrl_r}):
            print('Ctrl+C')
            input_state.user_stop = True
    except AttributeError:
        pass


with Listener(on_press=keypress_callback, on_release=keyrelease_callback) as listener:
    while not input_state.user_stop and not done:
        # env.render()

        action = get_action(input_state)
        state2, reward, done, info, _ = env.step(action)

        t += 1
        score += reward
        state = state2

        if t % 10 == 0:
            print('Step:', t)

    print('==> All Steps: ', t)
env.close()
