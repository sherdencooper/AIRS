from train_concept import *
from Input_Cell_Attention.Scripts.cell import *

class LSTM_Input_Cell(nn.Module):
    def __init__(self, h_size, r=20, d_a=50):
        super(LSTM_Input_Cell, self).__init__()
        self.h_size = h_size
        self.lstm = LSTMWithInputCellAttention(7, h_size, 20,50)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.h_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)
        self.fc3 = nn.Linear(self.h_size, self.h_size)
        self.fc4 = nn.Linear(self.h_size, 1)
        self.emd1 = nn.Embedding(22,3)
        self.emd2 = nn.Embedding(22,3)
        self.emd3 = nn.Embedding(3,1)
        
    def forward(self, x):
        input1 = self.emd1(x[:,:,0])
        input2 = self.emd2(x[:,:,1])
        input3 = self.emd3(x[:,:,2])
        temp = torch.cat((input1,input2), axis=2)
        x = torch.cat((temp, input3), axis=2)
        h0 = torch.zeros(1, x.size(0), self.h_size).cuda()
        c0 = torch.zeros(1, x.size(0), self.h_size).cuda()
        out, (h_n, h_c) = self.lstm(x, (h0, c0))
        # print(out.shape)
        x = self.fc1(out[:, -1, :])
        x = F.sigmoid(x)
        # x = self.dropout1(x)
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
    train_num = 10000
    test_num = 2000
    evaluate_num = 2000
    device = torch.device("cuda")
    save_dir = './model_store/cell.pth'
    net = LSTM_Input_Cell(h_size=hiddne_size).to(device) 
    # dummy_input = torch.rand(15,64,8).long().to(device)
    train_log_dir = './torch_log/train/' + TIMESTAMP
    test_log_dir = './torch_log/test/' + TIMESTAMP
    evaluate_log_dir = './torch_log/validate/' + TIMESTAMP
    net_log_dir = './torch_log/net/' + TIMESTAMP

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=gamma)

    epochs = 100
    train(net, device, train_data, train_label, test_data, test_label, evaluate_data, evaluate_label, optimizer, epochs, scheduler, train_log_dir,test_log_dir,evaluate_log_dir,USE_BEST_MODEL,train_num,test_num,evaluate_num,save_dir)
