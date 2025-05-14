import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from SEEDIV.Model import Model


batch_size = 64
num_epochs = 200
learning_rate = 0.01
channel_num = 62
band_num = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#五折交叉验证
DE = torch.tensor(np.load("G:/XXX/SEED-IV/test/DE/3/15.npy").real.astype(float), dtype=torch.float)
#会话一
# labels = np.concatenate([np.ones([42, 1]), np.full((23,1), 2), np.full((49,1), 3), np.zeros([32, 1]), np.full((22,1), 2), np.zeros([40, 1]), np.zeros([38, 1]), np.ones([52, 1]),
#                           np.zeros([36, 1]), np.ones([42, 1]), np.full((12,1), 2), np.ones([27, 1]), np.ones([54, 1]), np.ones([42, 1]), np.full((64,1), 2), np.full((35,1), 3),
#                           np.full((17,1), 2), np.full((44,1), 2), np.full((35,1), 3), np.full((12,1), 3), np.zeros([28, 1]), np.full((28,1), 3), np.zeros([43, 1]), np.full((34,1), 3)])
#会话二
# labels = np.concatenate([np.full((55,1), 2), np.ones([25, 1]), np.full((34,1), 3), np.zeros([36, 1]), np.zeros([53, 1]), np.full((27,1), 2), np.zeros([34, 1]), np.full((46,1), 2),
#                           np.full((34,1), 3), np.full((20,1), 3), np.full((60,1), 2), np.full((12,1), 3), np.full((36,1), 2), np.zeros([27, 1]), np.ones([44, 1]), np.ones([15, 1]),
#                           np.full((46,1), 2), np.ones([49, 1]), np.zeros([45, 1]), np.full((10,1), 3), np.zeros([37, 1]), np.ones([44, 1]), np.full((24,1), 3), np.ones([19, 1])])
#会话三
labels = np.concatenate([np.ones([42, 1]), np.full((32,1), 2), np.full((23,1), 2), np.ones([45, 1]), np.full((48,1), 3), np.full((26,1), 3), np.full((64,1), 3), np.ones([23, 1]),
                          np.ones([26, 1]), np.full((16,1), 2), np.ones([51, 1]), np.zeros([41, 1]), np.full((39,1), 2), np.full((19,1), 3), np.full((28,1), 3), np.zeros([44, 1]),
                          np.full((14,1), 2), np.full((17,1), 3), np.zeros([45, 1]), np.zeros([22, 1]), np.full((39,1), 2), np.zeros([38, 1]), np.ones([41, 1]), np.zeros([39, 1])])

labels = torch.tensor(labels, dtype=torch.int64).squeeze_(1)
MyDataset =TensorDataset(DE, labels)
kfold = KFold(n_splits=5, shuffle=True)

average_acc = 0
for train_idx, test_idx in kfold.split(MyDataset):
    train_data = Subset(MyDataset, train_idx)
    test_data = Subset(MyDataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,drop_last=True)
    min_acc = 0.3

    model = Model(xdim = [batch_size, channel_num, band_num], kadj=2, num_out=62, dropout=0.5).to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    G = testclass()
    train_len = G.len(len(train_idx), batch_size)
    test_len = G.len(len(test_idx), batch_size)

    for epoch in range(num_epochs):
    # -------------------------------------------------
        total_train_acc = 0
        total_train_loss = 0

        for de, labels in train_loader:
            de = de.to(device)
            labels = labels.to(device)

            output = model(de)
            # print("output:", output.shape)
            # print(labels.shape)
            train_loss = loss_func(output, labels.long())

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            train_acc = (output.argmax(dim=1) == labels).sum()

            train_loss_list.append(train_loss)
            total_train_loss = total_train_loss + train_loss.item()

            train_acc_list.append(train_acc)
            total_train_acc += train_acc

        train_loss_list.append(total_train_loss / (len(train_loader)))
        train_acc_list.append(total_train_acc / train_len)

    # -------------------------------------------------
        total_test_acc = 0
        total_test_loss = 0

        with torch.no_grad():
            for de, labels in test_loader:
                de = de.to(device)
                labels = labels.to(device)

                output = model(de)
                test_loss = loss_func(output, labels.long())

                test_acc = (output.argmax(dim=1) == labels).sum()

                test_loss_list.append(test_loss)
                total_test_loss = total_test_loss + test_loss.item()

                test_acc_list.append(test_acc)
                total_test_acc += test_acc

        test_loss_list.append(total_test_loss / (len(test_loader)))
        test_acc_list.append(total_test_acc / test_len)

        if (total_test_acc / test_len) > min_acc:
            min_acc = total_test_acc / test_len
            # res_TP_TN_FP_FN = TP_TN_FP_FN
            torch.save(model.state_dict(), 'G:/XXX/规则集解释/模型可视化/SEED-IV-3-15.pth')
        # print result
        print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
              "Training Loss: {:.4f} ".format(total_train_loss / len(train_loader)),
              "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
              "Test Loss: {:.4f} ".format(total_test_loss / len(test_loader)),
              "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
              )
    print(min_acc)
    average_acc += min_acc

average_acc = average_acc / 5
print("平均准确率为", average_acc)
