import torch.nn.init as init

import torch
import torch.nn.functional as F
from torch import nn

class MF_Hyper(nn.Module):
    def __init__(self,
        preference_dim=1,
        preference_embedding_dim=32,
        hidden_dim=64,
        num_chunks= 64,
        chunk_embedding_dim=64,
        num_ws=10,
        w_dim=1000, drop_out=0.15) :
        super().__init__()

        self.drop_out = drop_out

        self.hidden_dim = hidden_dim

        self.preference_embedding_dim = preference_embedding_dim
        self.num_chunks = num_chunks

        self.chunk_embedding_matrix = nn.Embedding(
            num_embeddings=num_chunks, embedding_dim=chunk_embedding_dim
        )
        self.preference_embedding_matrix = nn.Embedding(
            num_embeddings=preference_dim, embedding_dim=preference_embedding_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(preference_embedding_dim + chunk_embedding_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.1)
        )
        self.fc.add_module("Dropout", nn.Dropout(self.drop_out))

        # init weights
        list_ws = [self._init_w((w_dim, hidden_dim)) for _ in range(num_ws)]
        self.ws = nn.ParameterList(list_ws)

        # initialization
        torch.nn.init.normal_(
            self.preference_embedding_matrix.weight, mean=0.0, std=0.1
        )
        torch.nn.init.normal_(self.chunk_embedding_matrix.weight, mean=0.0, std=0.1)

        for w in self.ws:
            torch.nn.init.normal_(w, mean=0.0, std=0.1)
        self.layer_to_shape  = {
            "user_embedding.weights": torch.Size([6040,64]),
            "item_embedding.weights": torch.Size([3952,64])
        }

    def _init_w(self, shapes):
        return nn.Parameter(torch.randn(shapes), requires_grad=True)
    
    def forward(self, lambda_):
        pref_embedding = self.preference_embedding_matrix(
                    torch.tensor([0], device=lambda_.device)
                ).squeeze(0) * lambda_
        weights = []
        for chunk_id in range(self.num_chunks):
            #print(f'Chunkid: {chunk_id}')
            chunk_embedding = self.chunk_embedding_matrix(
                torch.tensor([chunk_id], device=lambda_.device)
            ).squeeze(0)
            # input to fc
            input_embedding = torch.cat((pref_embedding, chunk_embedding)).unsqueeze(0)
            rep = self.fc(input_embedding)

            weights.append(torch.cat([F.linear(rep, weight=w) for w in self.ws], dim=1))

        weight_vector = torch.cat(weights, dim=1).squeeze(0)
        out_dict = dict()
        position = 0
        for name, shapes in self.layer_to_shape.items():
            out_dict[name] = weight_vector[
                position : position + shapes.numel()
            ].reshape(shapes) #+ self.weights_dict[name].to(preference.device)
            position += shapes.numel()
            # print(name, shapes, out_dict[name].shape, position)

        # for  key, value in self.weights_dict.items():
        #     if key not in out_dict:
        #         out_dict[key] = value.to(preference.device)
        return out_dict

class MF_target(nn.Module):
    def __init__(self, num_users, num_items, device):
        super(MF_target, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        # self.user_embeddings = nn.Embedding(num_users, args.dim).to(self.device)
        # self.item_embeddings = nn.Embedding(num_items, args.dim).to(self.device)

        # self.user_embeddings.weight.data = torch.nn.init.normal_(self.user_embeddings.weight.data, 0, 0.01)
        # self.item_embeddings.weight.data = torch.nn.init.normal_(self.item_embeddings.weight.data, 0, 0.01)
        # self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    def forward(self, user_id, pos_id, neg_id, weights = None, cosine_distances_D = None):
        user_embeddings = nn.Embedding.from_pretrained(weights["user_embedding.weights"])
        item_embeddings = nn.Embedding.from_pretrained(weights["item_embedding.weights"])
        user_emb = user_embeddings(user_id)
        pos_emb = item_embeddings(pos_id)
        neg_emb = item_embeddings(neg_id)

        pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_emb), dim=1)
        total_item = torch.cat((pos_id, neg_id), dim=0)
        
        reg = 0
        for i in range(len(total_item)):
            for j in range(i+1,len(total_item)):
                d_i_j = cosine_distances_D[total_item[i]][total_item[j]]
                i_emb = item_embeddings(torch.LongTensor([total_item[i]])).squeeze(0)
                j_emb = item_embeddings(torch.LongTensor([total_item[j]])).squeeze(0)
                norm_of_difference = torch.norm(i_emb - j_emb, p=2)
                reg_i_j = d_i_j*norm_of_difference
                reg +=reg_i_j



        # print(pos_scores.shape)


        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores))
        # print(f'Loss shape: {loss.shape}')

        bpp_loss = torch.sum(loss)


        return bpp_loss, reg

    def predict(self, user_id, weights = None):
        # user_id = Variable(torch.from_numpy(user_id).long(), requires_grad=False).to(self.device)
        user_embeddings = nn.Embedding.from_pretrained(weights["user_embedding.weights"])
        item_embeddings = nn.Embedding.from_pretrained(weights["item_embedding.weights"])
        user_emb = user_embeddings(user_id)
        pred = user_emb.mm(item_embeddings.weight.t())


        return pred


