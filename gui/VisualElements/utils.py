import math


def curve(x1: int, y1: int, x2: int, y2: int) -> list:
    mx = (x2 - x1) / 2
    my = (y2 - y1) / 2
    c1 = [x1, y1 + my]
    c2 = [x2, y2 - my]
    steps = 0.001
    points = []

    angle = 1.5 * math.pi
    goal = 2 * math.pi
    while angle < goal:
        x = c1[0] + mx * math.cos(angle)
        y = c1[1] + my * math.sin(angle)
        points.extend([x, y])
        angle += steps

    angle = 1 * math.pi
    goal = 0.5 * math.pi
    while angle > goal:
        x = c2[0] + mx * math.cos(angle)
        y = c2[1] + my * math.sin(angle)
        points.extend([x, y])
        angle -= steps

    return points
