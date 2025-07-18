import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import f1_score, accuracy_score, recall_score

class NERMetricsCalculator(AccuracyCalculator):
    def calculate_f1_score(self, knn_labels, query_labels, **kwargs):
        """
        knn_labels: A 2d array where each row is the labels of the nearest neighbors of
            each query. The neighbors are retrieved from the reference set
        query_labels: A 1D torch or numpy array of size (Nq). Each element should be an 
            integer representing the sample's label.
        """
        predicted_classes = self.get_predicted_classes_from_knn_labels(knn_labels)
        # Calculate F1 score
        f1 = f1_score(query_labels.cpu().numpy(), predicted_classes.cpu().numpy(), average='macro')
        return f1
    
    def calculate_accuracy_score(self, knn_labels, query_labels, **kwargs):
        predict_classes = self.get_predicted_classes_from_knn_labels(knn_labels)
        acc = accuracy_score(query_labels.cpu().numpy(), predict_classes.cpu().numpy())
        return acc
    
    def calculate_recall_score(self, knn_labels, query_labels, **kwargs):
        predicted_classes = self.get_predicted_classes_from_knn_labels(knn_labels)
        recall = recall_score(query_labels.cpu().numpy(), predicted_classes.cpu().numpy(), average='macro')
        return recall

    def get_predicted_classes_from_knn_labels(self, knn_labels):
        """
        Takes a 2d array and returns the predicted labels from knn.
        
        Args:
            knn_labels: A 2d array where each row is the labels of the nearest 
                neighbors of each query. The neighbors are retrieved from the 
                reference set.
        Returns:
            predicted_classes: 1D Tensor of size (Nq) where each element is the 
                predicted class for each query.
        """
        # Count occurrences of each class per query
        predicted_classes, _ = torch.mode(knn_labels, dim=1)
        return predicted_classes

    def requires_knn(self):
        return super().requires_knn() + ["f1_score", "accuracy_score"]