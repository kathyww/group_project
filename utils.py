import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from wavelet import wavelet_transform, wavelet_shrinkage

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def load():
    def load_and_resize_images(folder_path, target_size):
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = ImageFolder(folder_path, transform=transform)
        images = []
        labels = []

        for image, label in dataset:
            image_np = image.numpy()
            wavelet_features = []
            for channel in range(image_np.shape[0]):
                channel_features = wavelet_transform(image_np[channel])
                wavelet_features.extend(channel_features)
            images.append(wavelet_shrinkage(wavelet_features))
            labels.append(dataset.classes[label])

        return images, labels

    training_folder = 'data/Training'
    testing_folder = 'data/Testing'

    target_size = (128, 128)

    train_images, train_labels = load_and_resize_images(training_folder, target_size)

    test_images, test_labels = load_and_resize_images(testing_folder, target_size)

    train_df = pd.DataFrame(train_images)
    train_df['class'] = train_labels

    test_df = pd.DataFrame(test_images)
    test_df['class'] = test_labels

    return train_df, test_df


def visualize(df):
    """
    Only valid for 3 components
    """
    class_map = {'glioma': 0,
                 'meningioma': 1,
                 'notumor': 2,
                 'pituitary': 3}
    class_labels = df['class'].unique()

    cmap = ListedColormap(sns.color_palette("colorblind").as_hex())

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    sc = ax.scatter(df['feature_1'], df['feature_2'], df['feature_3'],
                    s=40, c=df['class'].apply(lambda c: class_map[c]), marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

    legend_labels = [class_label for class_label in class_labels]
    plt.legend(handles=sc.legend_elements()[0], labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')