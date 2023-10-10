# from models.mf import MatrixFactorization
from models.hyper_MF import MF_Hyper, MF_target
from dataloader.movielens1MDataset import MVL1MDataset
from dataloader.process_data import ProcessMVL1M
from argparse import ArgumentParser
import torch
import logging
import numpy as np

from metric.recall_at_k import RecallAtK
from metric.HR_at_k import HRAtK
from metric.ILD_at_k import ILDAtK

from torch.utils.data import DataLoader
from tqdm import trange


def parse_args():
    parser = ArgumentParser(description="BPRMF")

    parser.add_argument('--data_path', type=str, default='/home/ubuntu/duc.nm195858/data/ml-1m')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of validation set")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Proportion of testing set")
    #model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    parser.add_argument('--num_neg',type=int, default=3, help="number of negative samples")

    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay factor")

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=200, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=20, help="Period for evaluating during training")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--batch_size_val',type=int, default=64)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--topK', type=int, default=10)

    # information datasets
    parser.add_argument('--n_users', type=int, default=6040)
    parser.add_argument('--n_items', type=int, default=3952)

    return parser.parse_args()

def get_device(no_cuda=False, gpus='1'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train():

    args = parse_args()
    device = get_device(no_cuda=args.no_cuda, gpus=args.gpu_id)

    hnet = MF_Hyper()
    model = MF_target(num_users= args.n_users, num_items= args.n_items, device=device)


    logging.info(f"hnet size: {count_parameters(hnet)}")
    model = model.to(device)


    processData = ProcessMVL1M(args.data_path, args.val_ratio, args.test_ratio, args.seed)
    train_data, val_data, test_data = processData.process()

    train_dataset = MVL1MDataset(train_data, args.n_users, args.n_items, args.num_neg)
    val_dataset = MVL1MDataset(val_data, args.n_users, args.n_items, args.num_neg)
    test_dataset = MVL1MDataset(test_data, args.n_users, args.n_items, args.num_neg)

    train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_iter = trange(args.n_epochs)


    # distance matrix D
    embedding_saved = torch.load('/home/ubuntu/duc.nm195858/diversity_MORS/weights/embedding/embedding_weights_dim64.pth')
    embedding_layer_data = embedding_saved['item_embeddings']
    normalized_embeddings = torch.nn.functional.normalize(embedding_layer_data, p=2, dim=1)

    # Tính ma trận khoảng cách cosine giữa các cặp item
    cosine_distances_D = 1 - torch.matmul(normalized_embeddings, normalized_embeddings.t())
    #metric
    hr_atK = HRAtK(args.topK, device=device)
    ild_at_k = ILDAtK(args.topK, device=device, cosine_distances_D=cosine_distances_D)


    for epoch in epoch_iter:
        loss_epoch = []
        loss_val_epoch = []

        for batch in train_dataloader:
            model.train()
            user,pos_item, neg_item = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            lambda_ = np.random.uniform(0, 1)
            lambda_ = torch.tensor(lambda_).to(device)
            weights = hnet(lambda_)
            # print(weights["item_embedding.weights"].shape)
            loss_BPR, reg = model.forward(user,pos_item,neg_item, weights, cosine_distances_D)
            loss = loss_BPR + lambda_*reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # recall_train_epoch.append(recall)
        
        #validation
        
        for batch in val_dataloader:
            model.eval()
            user,pos_item, neg_item = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            lambda_ = np.random.uniform(0, 1)
            lambda_ = torch.from_numpy(lambda_).to(device)
            weights = hnet(lambda_)

            loss_BPR,reg = model.forward(user,pos_item,neg_item, weights, cosine_distances_D)

            loss = loss_BPR +lambda_*reg
            loss_val_epoch.append(loss.item())

            # recall = recall_atK.evaluate(user, val_data['Positive'], model)
            # recall_train_epoch.append(recall)
        
        print(f"Epoch {epoch}: Loss_train = {np.mean(np.array(loss_epoch))}, Loss_val = {np.mean(np.array(loss_val_epoch))}")
        if (epoch+1)%20==0:
            HR_train_epoch = 0
            HR_val_epoch = 0

            ILD_train = 0
            ILD_val = 0
            cnt = 0
            for batch in train_dataloader:
                user= batch[0].to(device)
                cnt+=1
                HR_at_K = hr_atK.evaluate(user, train_data['Positive'], model)
                HR_train_epoch = HR_train_epoch*(cnt-1)/cnt + (1/cnt) *HR_at_K

                ild = ild_at_k.evaluate(user,model)
                ILD_train = ILD_train*(cnt-1)/cnt +(1/cnt)*ild

            print(f'HR@{args.topK}_train= {HR_train_epoch}')
            print(f'ILD@{args.topK}_train = {ILD_train}')

            cnt = 0
            for batch in val_dataloader:
                user = batch[0].to(device)
                cnt+=1
                HR_at_k = hr_atK.evaluate(user, val_data['Positive'], model)
                HR_val_epoch = HR_val_epoch*(cnt-1)/cnt + (1/cnt) *HR_at_k

                ild = ild_at_k.evaluate(user,model)
                ILD_val = ILD_val*(cnt-1)/cnt +(1/cnt)*ild

            print(f'HR@{args.topK}_val= {HR_val_epoch}')
            print(f'ILD@{args.topK}_val = {ILD_val}')
        
        

if __name__ == "__main__":
    train()
        




