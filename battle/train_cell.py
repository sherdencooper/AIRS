from train_concept import *
from Input_Cell_Attention.Scripts.cell import *

class LSTM_Input_Cell(nn.Module):
    def __init__(self, h_size, r=15, d_a=15):
        super(LSTM_Input_Cell, self).__init__()
        self.h_size = h_size
        self.lstm = LSTMWithInputCellAttention(16, h_size, r,d_a)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.h_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)
        self.fc3 = nn.Linear(self.h_size, self.h_size)
        self.fc4 = nn.Linear(self.h_size, 1)
        self.emd1 = nn.Embedding(2,1)
        self.emd2 = nn.Embedding(2,1)
        self.emd3 = nn.Embedding(21,3)
        self.emd4 = nn.Embedding(22,3)
        self.emd5 = nn.Embedding(5,1)
        self.emd6 = nn.Embedding(2,1)
        self.emd7 = nn.Embedding(22,3)
        self.emd8 = nn.Embedding(24,3)
        
    def forward(self, x):
        input1 = self.emd1(x[:,:,0])
        input2 = self.emd2(x[:,:,1])
        input3 = self.emd3(x[:,:,2])
        input4 = self.emd4(x[:,:,3])
        input5 = self.emd5(x[:,:,4])
        input6 = self.emd6(x[:,:,5])
        input7 = self.emd7(x[:,:,6])
        input8 = self.emd8(x[:,:,7])

        x = torch.cat((input1, input2), axis=2)
        x = torch.cat((x, input3), axis=2)
        x = torch.cat((x, input4), axis=2)
        x = torch.cat((x, input5), axis=2)
        x = torch.cat((x, input6), axis=2)
        x = torch.cat((x, input7), axis=2)
        x = torch.cat((x, input8), axis=2)
        h0 = torch.zeros(1, x.size(0), self.h_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.h_size).to(device)
        out, (h_n, h_c) = self.lstm(x, (h0, c0))
        # print(out.shape)
        x = self.fc1(out[:, -1, :])
        x = F.sigmoid(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = self.fc4(x)

        return x




if __name__ == "__main__":
    setup_seed(2021)
    USE_BEST_MODEL = True
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    hiddne_size = 8
    epsilon = 1e-4
    gamma = 0.2
    decay_step = 50
    train_num = 2000
    test_num = 100
    evaluate_num = 100
    device = torch.device("cuda")
    save_dir = './model_save/cell.pth'
    net = LSTM_Input_Cell(h_size=hiddne_size).to(device) 
    dummy_input = torch.rand(15,64,8).long().to(device)
    train_log_dir = './torch_log/train/' + TIMESTAMP
    test_log_dir = './torch_log/test/' + TIMESTAMP
    evaluate_log_dir = './torch_log/validate/' + TIMESTAMP
    net_log_dir = './torch_log/net/' + TIMESTAMP

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=gamma)

    epochs = 20
    train(net, device, train_data, train_label, test_data, test_label, evaluate_data, evaluate_label, optimizer, epochs, scheduler, train_log_dir,test_log_dir,evaluate_log_dir,USE_BEST_MODEL,train_num,test_num,evaluate_num,save_dir)
