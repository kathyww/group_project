import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from modules.wavelet import wavelet_transform, wavelet_shrinkage

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import plotly.express as px


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


#for cnn train/test split
def load1():

    def load_and_resize_images(folder_path, target_size):
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = ImageFolder(folder_path, transform=transform)
        image_paths = [path for path, _ in dataset.imgs]
        labels = [dataset.classes[label] for _, label in dataset.samples]
        return image_paths, labels

    training_folder = 'data/Training'
    testing_folder = 'data/Testing'
    target_size = (128, 128)

    train_image_paths, train_labels = load_and_resize_images(training_folder, target_size)
    test_image_paths, test_labels = load_and_resize_images(testing_folder, target_size)

    train_df = pd.DataFrame({'path': train_image_paths, 'label': train_labels})
    test_df = pd.DataFrame({'path': test_image_paths, 'label': test_labels})

    return train_df, test_df



def visualize_points(df, pca):
    """
    Only valid for 3 components
    """
    class_map = {'glioma': 0,
                 'meningioma': 1,
                 'notumor': 2,
                 'pituitary': 3}
    class_labels = df['class'].unique()

    cmap = ListedColormap(sns.color_palette("colorblind").as_hex())

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    sc = ax.scatter(df['feature_1'], df['feature_2'], df['feature_3'],
                    s=40, c=df['class'].apply(lambda c: class_map[c]), marker='o', cmap=cmap, alpha=1)

    for i in range(pca.components_.shape[1]):
        ax.plot3D([0, pca.components_[0, i]], [0, pca.components_[1, i]], [0, pca.components_[2, i]], color='red')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

    legend_labels = [class_label for class_label in class_labels]
    plt.legend(handles=sc.legend_elements()[0], labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig('images/points_plot.png')

    plt.show()


def visualize_features(X, y, pca_n, n=3):
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca_n.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        data_frame=X,
        labels=labels,
        dimensions=['feature_' + str(i + 1) for i in range(n)],
        color=y
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

    fig.write_image("images/feature_plot.png")


def visualize_hist(X, y):
    for n in range(len(X.columns)):
        plt.figure(figsize=(8, 4))
        sns.histplot(data=X, x='feature_{}'.format(n + 1), hue=y.values, kde=True, bins=20, palette='colorblind',
                     legend='full')
        plt.xlabel('Feature 1')
        plt.ylabel('Count')
        plt.title('Histogram of Feature {} by Class'.format(n + 1))
        plt.savefig('images/feature_{}_histogram.png'.format(n + 1))
        plt.show()


def visualize_evr(pca_lst):
    explained_variances = []

    for pca_n in pca_lst:
        explained_variance_ratio = sum(pca_n.explained_variance_ratio_)
        explained_variances.append(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca_lst)+1), explained_variances, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Total Explained Variance Ratio')
    plt.title('Total Explained Variance Ratio vs Number of Principal Components')
    plt.xticks(range(1, len(pca_lst)+1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/evr_plot.png')
    plt.show()
