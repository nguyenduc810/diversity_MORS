from models.mf import MatrixFactorization
from dataloader.movielens1MDataset import MVL1MDataset
from dataloader.process_data import ProcessMVL1M
from argparse import ArgumentParser
import torch
import logging
import numpy as np

from metric.recall_at_k import RecallAtK
from metric.HR_at_k import HRAtK

from torch.utils.data import DataLoader
from tqdm import trange


def parse_args():
    parser = ArgumentParser(description="BPRMF")

    parser.add_argument('--data_path', type=str, default='/home/ubuntu/duc.nm195858/data/ml-1m')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of validation set")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Proportion of testing set")
    #model
    parser.add_argument('--dim', type=int, default=64, help="Dimension for embedding")
    parser.add_argument('--num_neg',type=int, default=4, help="number of negative samples")

    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-2, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay factor")

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=500, help="Number of epoch during training")
    parser.add_argument('--every', type=int, default=20, help="Period for evaluating during training")
    parser.add_argument('--batch_size', type=int, default=256)
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

    model = MatrixFactorization(num_users= args.n_users, num_items= args.n_items, device=device, args = args)

    logging.info(f"model size: {count_parameters(model)}")
    model = model.to(device)

    recall_atK = RecallAtK(args.topK, device=device)
    hr_atK = HRAtK(args.topK, device=device)

    processData = ProcessMVL1M(args.data_path, args.val_ratio, args.test_ratio, args.seed)
    train_data, val_data, test_data = processData.process()

    train_dataset = MVL1MDataset(train_data, args.n_users, args.n_items, args.num_neg)
    val_dataset = MVL1MDataset(val_data, args.n_users, args.n_items, args.num_neg)
    test_dataset = MVL1MDataset(test_data, args.n_users, args.n_items, args.num_neg)

    train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True, num_workers= 4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_iter = trange(args.n_epochs)

    min_val_loss = 10e5
    for epoch in epoch_iter:
        loss_epoch = []
        loss_val_epoch = []
       
        for batch in train_dataloader:
            model.train()
            user,pos_item, neg_item = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            loss = model.forward(user,pos_item,neg_item)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        
        #validation
        
        for batch in val_dataloader:
            model.eval()
            user,pos_item, neg_item = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            loss = model.forward(user,pos_item,neg_item)
            loss_val_epoch.append(loss.item())

            # recall = recall_atK.evaluate(user, val_data['Positive'], model)
            # recall_train_epoch.append(recall)
        loss_epoch_val = np.mean(np.array(loss_val_epoch))
        #save weights
        if loss_epoch_val < min_val_loss:
            min_val_loss = loss_epoch_val
            torch.save({
                'user_embeddings': model.user_embeddings.weight,
                'item_embeddings': model.item_embeddings.weight
            }, f'/home/ubuntu/duc.nm195858/diversity_MORS/weights/embedding/embedding_weights_dim{args.dim}_neg4.pth')
            torch.save(model.state_dict(), f'/home/ubuntu/duc.nm195858/diversity_MORS/weights/BPRMF_model_dim{args.dim}_neg4.pth')
        
        print(f"Epoch {epoch}: Loss_train = {np.mean(np.array(loss_epoch))}, Loss_val = {np.mean(np.array(loss_val_epoch))}")

        # recall Validation
        if (epoch+1)%20==0:
            HR_train_epoch = 0
            HR_val_epoch = 0
            cnt = 0
            for batch in train_dataloader:
                user= batch[0].to(device)
                cnt+=1
                HR_at_K = hr_atK.evaluate(user, train_data['Positive'], model)
                HR_train_epoch = HR_train_epoch*(cnt-1)/cnt + (1/cnt) *HR_at_K
            print(f'HR@{args.topK}_train= {HR_train_epoch}')

            cnt = 0
            for batch in val_dataloader:
                user = batch[0].to(device)
                cnt+=1
                HR_at_k = hr_atK.evaluate(user, val_data['Positive'], model)
                HR_val_epoch = HR_val_epoch*(cnt-1)/cnt + (1/cnt) *HR_at_k
            print(f'HR@{args.topK}_val= {HR_val_epoch}')
        

if __name__ == "__main__":
    train()
        




