import numpy as np

def line(x1, y1, x2, y2, step):
    if x1 == x2:
        return generate_vertical_line(x1, y1, x2, y2, step)
    return generate_line(x1, y1, x2, y2, step)

def generate_line(x1, y1, x2, y2, step):
    m = (y2 - y1) / (x2 - x1)
    # y = m*x +n
    # n = y - m*x
    n = y2 - m * x2

    x_start = x1 if x1 < x2 else x2
    x_end = x2 if x1 < x2 else x1

    angle = np.arctan2(y2 - y1, x2 - x1)
    if angle > 1:
        pass
    step_x = step if y2==y1 else np.abs(angle) * step
    x = np.arange(x_start, x_end, step_x)
    if x1 > x2:
        x = x[::-1]
    y = m * x + n
    return x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)

def generate_vertical_line(x1, y1, x2, y2, step):
    y_start = y1 if y1 < y2 else y2
    y_end = y2 if y1 < y2 else y1
    y = np.arange(y_start, y_end, step)

    x = np.full(y.shape[0], x1)
    y = np.array(y)
    if y1 > y2:
        y = y[::-1]

    return x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)