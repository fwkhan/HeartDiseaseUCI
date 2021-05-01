# extract the data


def extract_data(dataset):
    # extract columns 'age'
    dataset.loc[dataset['age'] <= 40, 'age'] = 0
    dataset.loc[(dataset['age'] > 40) & (dataset['age'] <= 50), 'age'] = 1
    dataset.loc[(dataset['age'] > 50) & (dataset['age'] <= 60), 'age'] = 2
    dataset.loc[(dataset['age'] > 60) & (dataset['age'] <= 70), 'age'] = 3
    dataset.loc[dataset['age'] > 70, 'age'] = 4

    # extract columns 'trestbps'
    dataset.loc[dataset['trestbps'] <= 116, 'trestbps'] = 0
    dataset.loc[(dataset['trestbps'] > 116) & (dataset['trestbps'] <= 138), 'trestbps'] = 1
    dataset.loc[(dataset['trestbps'] > 138) & (dataset['trestbps'] <= 150), 'trestbps'] = 2
    dataset.loc[(dataset['trestbps'] > 150) & (dataset['trestbps'] <= 172), 'trestbps'] = 3
    dataset.loc[dataset['trestbps'] > 172, 'trestbps'] = 4

    # extract columns 'chol'
    dataset.loc[dataset['chol'] <= 214, 'chol'] = 0
    dataset.loc[(dataset['chol'] > 214) & (dataset['chol'] <= 302), 'chol'] = 1
    dataset.loc[(dataset['chol'] > 302) & (dataset['chol'] <= 390), 'chol'] = 2
    dataset.loc[(dataset['chol'] > 390) & (dataset['chol'] <= 478), 'chol'] = 3
    dataset.loc[dataset['chol'] > 478, 'chol'] = 4

    # extract columns 'thalach'
    dataset.loc[dataset['thalach'] <= 98, 'thalach'] = 0
    dataset.loc[(dataset['thalach'] > 98) & (dataset['thalach'] <= 125), 'thalach'] = 1
    dataset.loc[(dataset['thalach'] > 125) & (dataset['thalach'] <= 152), 'thalach'] = 2
    dataset.loc[(dataset['thalach'] > 152) & (dataset['thalach'] <= 179), 'thalach'] = 3
    dataset.loc[dataset['thalach'] > 179, 'thalach'] = 4

    return dataset
