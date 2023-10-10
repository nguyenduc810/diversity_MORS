import torch
from torch.utils.data import Dataset

class MVL1MDataset(Dataset):
    def __init__(self, data, n_user, n_items, num_neg = 3):
        self.data = data
        self.n_user = n_user
        self.n_items = n_items
        self.data_pos = data['Positive']
        self.data_neg = data['Negative']
        self.num_neg = num_neg
        self.user_input =[]
        self.pos_item_input = []
        self.neg_item_input = []
        self.items_all = list(range(self.n_items))

        self.process()

    def __len__(self):
        return len(self.user_input)
    
    def __getitem__(self, index):
        return self.user_input[index], self.pos_item_input[index], self.neg_item_input[index]
    
    def process(self):
        for user_id, item_list in self.data_pos.items():
            num_pos = len(item_list)

            if user_id not in self.data_neg:
                num_neg_real = 0
            else:
                num_neg_real = len(self.data_neg[user_id])
            if num_neg_real < num_pos*self.num_neg:
                num_mis = num_pos*self.num_neg - num_neg_real
                if num_neg_real == 0:
                    item_add = list(set(self.items_all)  - set(item_list))
                    self.neg_list = item_add[:num_mis]
                else:
                    item_add = list(set(self.items_all) - set(self.data_neg[user_id]) - set(item_list))
                    self.neg_list = self.data_neg[user_id] + item_add[:num_mis]
            elif num_neg_real == num_pos*self.num_neg:
                self.neg_list = self.data_neg[user_id]
            elif num_neg_real > num_pos*self.num_neg:
                self.neg_list = self.data_neg[user_id][: num_pos*self.num_neg]

            for i, item in enumerate(item_list):
                for j in range(self.num_neg):
                    self.user_input.append(user_id)
                    self.pos_item_input.append(item)
                    self.neg_item_input.append(self.neg_list[i+j])



