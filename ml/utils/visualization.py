import matplotlib.pyplot as plt
import torch

def visualize(image, prediction):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    for box in prediction["boxes"]:
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', linewidth=2, fill=False))
    plt.show()
