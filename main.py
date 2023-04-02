import os
import pygame
import cv2
from fastai.vision.all import *
from fastcore.all import *
from load_model import data_path
from game_utils import PowerUp, detect_gesture, display_score_lives_level, POWER_UPS

def main():
    pygame.init()

    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Gesture Pong")
    clock = pygame.time.Clock()
    FPS = 60

    learn = load_learner("final_model.pkl")

    game_loop(screen, learn, width, height, clock, FPS, POWER_UPS)

    pygame.quit()

def game_loop(screen, learn, width, height, clock, FPS, POWER_UPS):
    paddle_speed = 10
    ball_speed = 5
    paddle_width, paddle_height = 150, 20
    ball_radius = 10
    paddle_x = (width - paddle_width) // 2
    paddle_y = height - paddle_height - 10
    ball_x, ball_y = width // 2, height // 2
    ball_dx = random.choice([-1, 1]) * ball_speed
    ball_dy = -ball_speed
    paddle_x_target = paddle_x
    smoothing_factor = 0.8
    score = 0
    lives = 3
    level = 1
    level_up_score = 10
    power_up_chance = 0.5
    power_up_duration = 300  # Duration in frames
    active_power_ups = []

    cap = cv2.VideoCapture(0)

    running = True
    while running:
        screen.fill((0, 0, 0))
        ret, frame = cap.read()
        if ret:
            gesture = detect_gesture(learn, frame) 

            if gesture == "peace" and paddle_x_target > 0:
                paddle_x_target -= paddle_speed
            elif gesture == "fist" and paddle_x_target < width - paddle_width:
                paddle_x_target += paddle_speed

            paddle_x = smoothing_factor * paddle_x + (1 - smoothing_factor) * paddle_x_target

        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x < ball_radius or ball_x > width - ball_radius:
            ball_dx = -ball_dx

        if ball_y < ball_radius:
            ball_dy = -ball_dy
            if random.random() < power_up_chance:
                power_up_kind = random.choice(list(POWER_UPS.keys()))
                power_up = PowerUp(ball_x, ball_y, power_up_kind, power_up_duration)
                active_power_ups.append(power_up)

        for power_up in active_power_ups:
            power_up.update()
            power_up.draw(screen)

            if (
                paddle_y - paddle_height <= power_up.y + power_up.height
                and paddle_x - power_up.width <= power_up.x <= paddle_x + paddle_width
            ):
                if power_up.kind in ["longer_paddle", "shorter_paddle"]:
                    paddle_width = int(paddle_width * POWER_UPS[power_up.kind]["width_mult"])
                elif power_up.kind in ["faster_ball", "slower_ball"]:
                    ball_dy = ball_dy * POWER_UPS[power_up.kind]["speed_mult"]
                power_up.duration = 0

        active_power_ups = [p for p in active_power_ups if p.duration > 0]
        for p in active_power_ups:
            p.duration -= 1

        if (
            ball_y > paddle_y - ball_radius
            and ball_x > paddle_x - ball_x
            and ball_x < paddle_x + paddle_width + ball_radius
        ):
            ball_dy = -ball_dy
            score += 1

            if score % level_up_score == 0:
                level += 1
                ball_speed *= 1.1
                ball_dy = -abs(ball_dy) * (ball_speed / abs(ball_dy))

        if ball_y > height + ball_radius:
            lives -= 1
            if lives == 0:
                running = False
            else:
                ball_x, ball_y = width // 2, height // 2
                ball_dx = random.choice([-1, 1]) * ball_speed
                ball_dy = -ball_speed

        pygame.draw.rect(screen, (255, 255, 255), (paddle_x, paddle_y, paddle_width, paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), (ball_x, ball_y), ball_radius)

        display_score_lives_level(screen, score, lives, level)
        pygame.display.flip()
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    cap.release()

if __name__ == "__main__":
    main()
