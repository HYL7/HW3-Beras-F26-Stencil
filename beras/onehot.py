import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        vectors = np.eye(len(unique_labels))
        
        self.one_hot_dict = {}
        for i, label in enumerate(unique_labels):
            self.one_hot_dict[label] = vectors[i]

    def forward(self, data):
        if hasattr(self, "one_hot_dict") is False:
            self.fit(data)
            
        result = np.zeros((len(data), len(self.one_hot_dict)))

        for i in range (len(data)):
            result[i] = self.one_hot_dict[data[i]]

        return result
        
    def inverse(self, data):
        result = []
        for i in range (len(data)):
            for key, value in self.one_hot_dict.items():
                if np.array_equal(value, data[i]):
                    result.append(key)
                    break
                
        return np.array(result)       
