import pygame
import random
import cv2
from PIL import Image
from fastai.vision.all import *


POWER_UPS = {
    "longer_paddle": {"color": (0, 255, 0), "width_mult": 2},
    "shorter_paddle": {"color": (255, 0, 0), "width_mult": 0.5},
    "faster_ball": {"color": (0, 0, 255), "speed_mult": 1.5},
    "slower_ball": {"color": (255, 255, 0), "speed_mult": 0.5},
}

class PowerUp:
    def __init__(self, x, y, kind, duration):
        self.x = x
        self.y = y
        self.kind = kind
        self.duration = duration
        self.width = 20
        self.height = 20
        self.dy = 2
    
    def update(self):
        self.y += self.dy

    def draw(self, screen):
        pygame.draw.rect(screen, POWER_UPS[self.kind]["color"], (self.x, self.y, self.width, self.height))

def detect_gesture(learn, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    gesture, _, _ = learn.predict(img)
    return gesture

def display_score_lives_level(screen, score, lives, level):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}  Lives: {lives}  Level: {level}", True, (255, 255, 255))
    screen.blit(text, (10, 10))
