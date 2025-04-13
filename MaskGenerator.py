import numpy as np
import cv2

class MaskGenerator:
    def __init__(self, height, width, channels=1,
                 rand_seed=None,
                 max_vertex=30, max_length=100, max_brush_width=20,
                 max_angle=360, min_num_vertex=10, max_num_vertex=20,
                 num_dots=30):
        self.height = height
        self.width = width
        self.channels = channels
        self.max_vertex = max_vertex
        self.max_length = max_length
        self.max_brush_width = max_brush_width
        self.max_angle = max_angle
        self.min_num_vertex = min_num_vertex
        self.max_num_vertex = max_num_vertex
        self.num_dots = num_dots
        if rand_seed is not None:
            np.random.seed(rand_seed)

    def _generate_mask(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Vẽ các đường vẽ ngẫu nhiên (random strokes)
        num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
        for _ in range(num_vertex):
            start_x = np.random.randint(0, self.height)
            start_y = np.random.randint(0, self.width)
            for _ in range(np.random.randint(1, self.max_vertex)):
                angle = 0.01 * np.random.randint(0, int(self.max_angle))
                length = np.random.randint(10, self.max_length)
                brush_width = np.random.randint(5, self.max_brush_width)

                end_x = int(start_x + length * np.sin(angle))
                end_y = int(start_y + length * np.cos(angle))

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 255, brush_width)
                start_x, start_y = end_x, end_y

        # Thêm các chấm ngẫu nhiên
        for _ in range(self.num_dots):
            x = np.random.randint(0, self.height)
            y = np.random.randint(0, self.width)
            radius = np.random.randint(3, 10)
            cv2.circle(mask, (y, x), radius, 255, -1)

        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
        return 1.0 - mask / 255.0  # 1: giữ lại, 0: mask

    def sample(self):
        return self._generate_mask()

import matplotlib.pyplot as plt

mask_generator = MaskGenerator(512, 512, rand_seed=None)

plt.figure(figsize=(12, 6))
for i in range(5):
    mask = mask_generator.sample()
    plt.subplot(1, 5, i + 1)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title(f"Mask {i}")
    plt.axis("off")
plt.tight_layout()
plt.show()
