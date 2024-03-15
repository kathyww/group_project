import pandas as pd
from sklearn.decomposition import PCA


def extract(train_df, test_df, n_components=3):
    pca_n = PCA(n_components=n_components, svd_solver='full')
    transformed_features_n_train = pca_n.fit_transform(train_df[train_df.columns[:-1]])
    transformed_features_n_test = pca_n.transform(test_df[test_df.columns[:-1]])

    features_n_train = pd.DataFrame(transformed_features_n_train,
                                    columns=['feature_' + str(i + 1) for i in range(n_components)])
    features_n_test = pd.DataFrame(transformed_features_n_test,
                                   columns=['feature_' + str(i + 1) for i in range(n_components)])
    features_n_train['class'] = train_df['class']
    features_n_test['class'] = test_df['class']
    features_n_train.to_csv('features/{}_features_train.csv'.format(str(n_components)), index=False)
    features_n_test.to_csv('features/{}_features_test.csv'.format(str(n_components)), index=False)

    return pca_n
