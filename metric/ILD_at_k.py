import torch
import numpy as np

class ILDAtK:
    def __init__(self, topK, device, cosine_distances_D=None):
        self.topK = topK
        self.device = device
        self.cosine_distances_D = cosine_distances_D
    
    def evaluate(self,user,model):
        user_list = set(user)
        user = torch.tensor(list(user_list), device= self.device)
        scores = model.predict(user)
        _, top_indices = torch.topk(scores, k=self.topk, dim=1)

        ild  = list()
        for i in range(top_indices):
            ild.append(self.one_user(top_indices[i]))
        return np.mean(np.array(ild))
    
    def one_user(self,top_items):
        # check = list()
        d = 0
        for i in range(len(top_items)):
            for j in range(i+1,len(top_items)):
                    d += self.cosine_distances_D[top_items[i]][top_items[j]]
                    # check.append((j,i))
        return 1/(self.topK*(self.topK-1))*d


                

            
