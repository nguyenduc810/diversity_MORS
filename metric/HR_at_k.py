import torch
import numpy as np

class HRAtK:
    def __init__(self, topk, device):
        self.topk = topk
        self.device = device

    def evaluate(self, user, user_labels, model):
        user_list = set(user)

        labels = []
        for user in user_list:
            labels.append(user_labels[user.item()])
        
        user = torch.tensor(list(user_list), device= self.device)
        scores = model.predict(user)
        _, top_indices = torch.topk(scores, k=self.topk, dim=1)

        num_user = 0
        for i in range(len(labels)):
            num_user+= self.hit_ratio(labels[i], top_indices[i])
        
        return num_user/len(labels)
    

    def hit_ratio(self, true_items, predicted_ranking):
        """
        Tính Hit Ratio.

        Parameters:
        - true_items: List hoặc tensor chứa các item thực tế mà người dùng đã tương tác.
        - predicted_ranking: List hoặc tensor chứa danh sách item được dự đoán theo thứ tự giảm dần.

        Returns:
        - hit_ratio: Giá trị Hit Ratio.
        """
        # Kiểm tra xem có ít nhất một item thực tế trong danh sách dự đoán không
        hit = any(item in true_items for item in predicted_ranking)

        # Tính Hit Ratio
        hit_ratio = int(hit) / 1.0  # Chia cho 1.0 để có giá trị float

        return hit_ratio