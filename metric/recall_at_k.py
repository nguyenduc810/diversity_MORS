import torch
import numpy as np

class RecallAtK:
    def __init__(self, k, device):
        self.k = k
        self.device = device
    
    def evaluate(self, user, user_labels, model):
        user_list = set(user)

        labels = []
        for user in user_list:
            # print(f'user: {user.item()}')
            labels.append(user_labels[user.item()])
            # print(f"len_Labels: {len(labels[-1])}")
        #create user: item
        # user_item = list()
        # for user in user_list:
        #     user_item.append(pos_item[user == user])
        # user_item = torch.stack(user_item)
        
        user = torch.tensor(list(user_list), device= self.device)
        # print(user)
        # print(user.shape)

        # user_embedding = user_emb.forward(user)
        # scores = torch.mm(user_embedding, items_emb.weight.t())
        scores = model.predict(user)

        _, top_indices = torch.topk(scores, k=self.k, dim=1)

        recall_all = []

        for i in range(len(labels)):
            # print(top_indices[i])
            # print(labels[i])
            # intersection =set(top_indices[i]).intersection(labels[i])
            # common_elements = [item for item in labels[i] if item in top_indices[i]]
            is_in_list = torch.isin(top_indices[i], torch.tensor(labels[i]))

            # Lấy những phần tử trong tensor mà có trong list
            common_elements = top_indices[i][is_in_list]
            # print(intersection)
            num_intersected_elements = len(common_elements)
            recall = num_intersected_elements/len(labels[i])
            recall_all.append(recall)
        return np.mean(np.array(recall_all))


