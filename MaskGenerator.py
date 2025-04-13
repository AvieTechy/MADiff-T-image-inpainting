import numpy as np

class MaskGenerator:
    def __init__(self, height, width, channels=1,
                 rand_seed=None,
                 max_vertex=20, max_length=100, max_brush_width=20,
                 max_angle=360, min_num_vertex=5, max_num_vertex=10):
        self.height = height
        self.width = width
        self.channels = channels
        self.max_vertex = max_vertex
        self.max_length = max_length
        self.max_brush_width = max_brush_width
        self.max_angle = max_angle
        self.min_num_vertex = min_num_vertex
        self.max_num_vertex = max_num_vertex
        if rand_seed is not None:
            np.random.seed(rand_seed)

    def _generate_mask(self):
        mask = np.zeros((self.height, self.width, self.channels), dtype=np.float32)
        num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)

        for _ in range(num_vertex):
            start_x = np.random.randint(0, self.height)
            start_y = np.random.randint(0, self.width)

            for _ in range(np.random.randint(1, self.max_vertex)):
                angle = 0.01 * np.random.randint(0, int(self.max_angle))
                length = np.random.randint(10, self.max_length)
                brush_width = np.random.randint(5, self.max_brush_width)

                end_x = start_x + length * np.sin(angle)
                end_y = start_y + length * np.cos(angle)

                rr, cc = self._line(int(start_x), int(start_y), int(end_x), int(end_y))
                rr = np.clip(rr, 0, self.height - 1)
                cc = np.clip(cc, 0, self.width - 1)
                mask[rr, cc, :] = 1

                start_x, start_y = end_x, end_y

        return 1 - mask  # 1: keep, 0: mask

    def sample(self):
        return self._generate_mask()

    def _line(self, x0, y0, x1, y1):
        """Bresenham's Line Algorithm"""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        rr, cc = [], []
        while True:
            rr.append(x0)
            cc.append(y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return np.array(rr), np.array(cc)
