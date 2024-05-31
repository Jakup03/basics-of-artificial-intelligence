import numpy as np
import pygame
from typing import Type
import skfuzzy as fuzz
from skfuzzy import control

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

# import numpy as np
# import matplotlib.pyplot as plt

# universe - w jakim zakresie moze sie poruszać czyli jakie wartości może przyjąć x_dist
            # x_dist odległośc miedzy piłeczką a paletką w osi x

# mamdani - bierze input zamienia na fuzzy function czyli funkcje rozmyta potem oblicza wynik i wynik zapisany jest ajko fuzzy function, żeby
# otrzymać wartość musi zamienić z fuzzy function na zwykłą liczbe

# tsk - pod input przypisuje funkcje liniowa (trojkątna), oblicza i output jest liczba nie trzeba zamieniać jak w mamdani





class FuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)

        width = board.surface.get_width()
        height = board.surface.get_height()

        racket_width = racket.width
        racket_speed = 10

        racket_x = racket.rect.x
        # control.Antecedent obiekt wejściowy do bota traktowany jako input
        x_dist = control.Antecedent(

            universe=np.arange(-width, width + 1),
            label="x_dist",
        )
        y_dist = control.Antecedent(
            universe=np.arange(0, height + 1),
            label="y_dist",
        )
        # Obiekt consequent output
        movement = control.Consequent(
            universe=np.arange(-racket_speed, racket_speed + 1),
            label="movement",
        )

        x_dist["left"] = fuzz.trapmf(
            x_dist.universe, [-width, -width, -200, 0 - racket_width / 2]
        )
        x_dist["left_center"] = fuzz.trimf(
            x_dist.universe,
            [-200, 0 - racket_width / 2, 0],
        )
        x_dist["mid"] = fuzz.trimf(
            x_dist.universe, [0 - racket_width / 2, 0, 0 + racket_width / 2]
        )
        x_dist["right_center"] = fuzz.trimf(
            x_dist.universe,
            [0, 0 + racket_width / 2, 200],
        )
        x_dist["right"] = fuzz.trapmf(
            x_dist.universe, [0 + racket_width / 2, 200, width, width]
        )

        y_dist["all"] = fuzz.trimf(
            y_dist.universe, [0, height / 2, height]
        )

        movement["left"] = fuzz.trimf(
            movement.universe, [-racket_speed, -racket_speed, 0]
        )
        movement["stay"] = fuzz.trimf(
            movement.universe, [-racket_speed, 0, racket_speed]
        )
        movement["right"] = fuzz.trimf(
            movement.universe, [0, racket_speed, racket_speed]
        )

        x_dist.view()
        y_dist.view()
        movement.view()

        rule_left = control.Rule(
            x_dist["left"] & y_dist["all"],
            movement["left"],
        )
        rule_left_center = control.Rule(
            x_dist["left_center"] & y_dist["all"],
            movement["stay"],
        )
        rule_center = control.Rule(
            x_dist["mid"] & y_dist["all"],
            movement["left"],
        )
        rule_right_center = control.Rule(
            x_dist["right_center"] & y_dist["all"],
            movement["stay"],
        )
        rule_right = control.Rule(
            x_dist["right"] & y_dist["all"],
            movement["right"],
        )

        control_system = control.ControlSystem(
            [
                rule_left,
                rule_left_center,
                rule_center,
                rule_right_center,
                rule_right,
            ]
        )
        self.controller = control.ControlSystemSimulation(control_system)

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):
        if x_diff == 0 or y_diff == 0 or x_diff == 40 or x_diff == -40:
            return 0

        self.controller.input["x_dist"] = -x_diff
        self.controller.input["y_dist"] = y_diff

        self.controller.compute()

        paddle_speed = self.controller.output["movement"] * 2

        return paddle_speed


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    game.run()