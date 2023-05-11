import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:

    def __init__(self, k:int, test_split_ratio:float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio

    @property
    def k_neighbors(self) -> int:
        return self.k

    @staticmethod
    def load_csv(csv_path:str) ->Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(csv_path, delimiter=',')
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x,y = pd.DataFrame(dataset.iloc[:,:-1]), pd.DataFrame(dataset.iloc[:,-1])
        return x, y

    def train_test_split(self, features:pd.DataFrame, labels:pd.DataFrame) -> None:
        test_size = int(features.shape[0] * self.test_split_ratio)
        train_size = features.shape[0] - test_size

        assert features.shape[0] == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features.iloc[:train_size,:],labels[:train_size]
        self.x_test, self.y_test = features.iloc[train_size:train_size+test_size,:], labels[train_size:train_size + test_size]

    
    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
        points = self.x_train.reset_index(drop=True)
        element_of_x = element_of_x.reindex(points.index).ffill()
        distances = ((points - element_of_x) ** 2).sum(axis=1) ** 0.5
        return pd.DataFrame(distances, columns=['distance'])
    
    def predict(self, x_test:pd.DataFrame) -> None:
        labels_pred = []
        for x_test_element in x_test.itertuples(index=False):
        
            row = pd.DataFrame(x_test_element).transpose()
            row.columns = x_test_element._fields

            distances = self.euclidean(row)
            distances = pd.concat([distances, self.y_train], axis=1)
            distances.sort_values(by='distance', axis=0, inplace=True)

            label_pred = distances.iloc[:self.k, -1].mode()[0]
            labels_pred.append(label_pred)
        self.y_preds = pd.DataFrame(labels_pred, columns=['Outcome'])
    
    def accuracy(self) -> float:
        y_test_local = self.y_test.copy()
        y_test_local.reset_index(drop=True, inplace=True)

        true_positive = (y_test_local.iloc[:,0] == self.y_preds.iloc[:,0]).sum()
        return true_positive / y_test_local.shape[0] * 100
    
    def best_k(self) -> Tuple[int, float]:
        old_k = self.k
        results = []
        for k in range(1,21):
            self.k = k
            self.predict(self.x_test)
            res = round(self.accuracy(), 2)
            results.append((k, res))

        self.k = old_k
        return max(results, key=lambda x:x[1])

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_preds)