import numpy as np
import cv2

class MaskGenerator:
    """
    Sinh mask ngẫu nhiên cho ảnh:
    - Vẽ các đường stroke với độ dài, góc và độ rộng bút khác nhau
    - Thêm các chấm nhỏ ngẫu nhiên
    Trả về mask giá trị 1 (vùng giữ lại) và 0 (vùng bị che).
    """
    def __init__(self, height, width, channels=1,
                 rand_seed=None,
                 max_vertex=30,        # số lượng điểm nối tối đa cho mỗi stroke
                 max_length=100,       # độ dài tối đa của mỗi đoạn stroke
                 max_brush_width=20,   # độ rộng bút tối đa
                 max_angle=360,        # góc quay tối đa (độ)
                 min_num_vertex=10,    # số điểm nối tối thiểu cho mỗi mask
                 max_num_vertex=20,    # số điểm nối tối đa cho mỗi mask
                 num_dots=30):         # số lượng chấm ngẫu nhiên
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
            np.random.seed(rand_seed)  # thiết lập seed để tái lập kết quả

    def _generate_mask(self):
        # Khởi tạo mask đen (0) kích thước H×W
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Vẽ các stroke ngẫu nhiên
        num_vertex = np.random.randint(self.min_num_vertex, self.max_num_vertex)
        for _ in range(num_vertex):
            # Chọn điểm bắt đầu ngẫu nhiên
            start_x = np.random.randint(0, self.height)
            start_y = np.random.randint(0, self.width)
            # Vẽ liên tiếp các đoạn nhỏ nối tiếp nhau
            for _ in range(np.random.randint(1, self.max_vertex)):
                angle = np.deg2rad(np.random.randint(0, self.max_angle))  # góc ngẫu nhiên (rad)
                length = np.random.randint(10, self.max_length)           # độ dài ngẫu nhiên
                brush_width = np.random.randint(5, self.max_brush_width)  # độ rộng bút ngẫu nhiên

                # Tính toạ độ điểm kết thúc
                end_x = int(start_x + length * np.sin(angle))
                end_y = int(start_y + length * np.cos(angle))

                # Vẽ đường thẳng trên mask (255 = trắng)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 255, brush_width)
                start_x, start_y = end_x, end_y

        # Thêm các chấm ngẫu nhiên vào mask
        for _ in range(self.num_dots):
            x = np.random.randint(0, self.height)
            y = np.random.randint(0, self.width)
            radius = np.random.randint(3, 10)  # bán kính chấm ngẫu nhiên
            cv2.circle(mask, (y, x), radius, 255, -1)  # -1 để fill đầy circle

        # Mở rộng thêm 1 kênh (H, W, 1)
        mask = np.expand_dims(mask, axis=-1)
        # Chuyển trắng thành 0 (mask), đen thành 1 (giữ vùng)
        return 1.0 - mask / 255.0

    def sample(self):
        """Trả về 1 mask mới mỗi lần gọi."""
        return self._generate_mask()


# Ví dụ hiển thị một vài mask mẫu
import matplotlib.pyplot as plt

mask_generator = MaskGenerator(512, 512, rand_seed=None)

plt.figure(figsize=(12, 6))
for i in range(5):
    mask = mask_generator.sample()      # sinh mask
    plt.subplot(1, 5, i + 1)
    plt.imshow(mask.squeeze(), cmap="gray")  # hiển thị mask dưới dạng ảnh grayscale
    plt.title(f"Mask {i}")
    plt.axis("off")
plt.tight_layout()
plt.show()
