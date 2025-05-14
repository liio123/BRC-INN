import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from Model import Model

batch_size = 256
num_epochs = 200
learning_rate = 0.01
channel_num = 62
band_num = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#五折交叉验证
DE = torch.tensor(np.load("G:/XXX/SEED/第三个会话/DE/15.npy").real.astype(float), dtype=torch.float)
labels = np.concatenate([np.full((235,1), 2), np.ones([233, 1]), np.zeros([206, 1]), np.zeros([238, 1]), np.ones([185, 1]), np.full((195,1), 2), np.zeros([237, 1]),
                          np.ones([216, 1]), np.full((265,1), 2), np.full((237,1), 2), np.ones([235, 1]), np.zeros([233, 1]), np.ones([235, 1]), np.full((238,1), 2),
                          np.ones([206, 1])])
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
            torch.save(model.state_dict(), 'G:/XXX/规则集解释/模型可视化/SEED-3-15.pth')
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
