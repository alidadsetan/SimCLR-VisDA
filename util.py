import random
import torch
import torchvision
import matplotlib.pyplot as plt

def create_samples(data,save_directory, num_samples=6):
    random_indexes = random.choices(range(len(data)),k=num_samples)
    imgs = torch.stack([img for idx in random_indexes for img in data[idx][0]], dim=0)
    # print([train_data[idx][1] for idx in random_indexes])
    img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10,5))
    plt.title('Augmented image examples of the visdas dataset')
    plt.imshow(img_grid)
    plt.axis('off')

    save_directory.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_directory/"sample.png")
    plt.close()