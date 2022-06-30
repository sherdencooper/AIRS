from train_concept import *


class Attention_model(nn.Module):
    def __init__(self, hidden_dim,
                 n_layers=1, use_bidirectional=True, use_dropout=False):
        super().__init__()
        self.emd1 = nn.Embedding(22,3)
        self.emd2 = nn.Embedding(22,3)
        self.emd3 = nn.Embedding(3,1)
        self.h_size = hidden_dim
        self.rnn = nn.GRU(7, self.h_size, bidirectional=True)
        self.fc = nn.Linear(15, 1)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)
        self.attn = nn.Linear((self.h_size * 2) + self.h_size, self.h_size)
        self.v = nn.Linear(self.h_size, 1, bias=False)

    def attention(self, hidden, encoder_outputs):
        # hidden => [batch_size, dec_hidden_dim]
        # encoder_outputs => [src_len, batch_size, enc_hidden_dim * num_directions]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden state src_len times 
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden => [batch_size, src_len, dec_hidden_dim]
        # encoder_outputs => [batch_size, src_len, enc_hidden_dim * num_directions]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy => [batch_size, src_len, dec_hidden_dim]
        attention = self.v(energy).squeeze(2)
        # attention =>
        return F.softmax(attention, dim=1)

    def forward(self, x, output_explain=False):
        input1 = self.emd1(x[:,:,0])
        input2 = self.emd2(x[:,:,1])
        input3 = self.emd3(x[:,:,2])
        temp = torch.cat((input1,input2), axis=2)
        x = torch.cat((temp, input3), axis=2).permute(1, 0, 2)
        output, (hidden, cell) = self.rnn(x)
        attn_output = self.attention(hidden, output)
        pred = self.fc(attn_output)
        if output_explain:
            return pred, attn_output
        else:
            return pred





if __name__ == "__main__":
    setup_seed(2021)
    USE_BEST_MODEL = True
    train_data,test_data,evaluate_data,train_label,test_label, evaluate_label = generate_dataset()
    hidden_size = 8
    epsilon = 1e-4
    gamma = 0.2
    decay_step = 50
    train_num = 2000
    test_num = 100
    evaluate_num = 100
    device = torch.device("cuda")
    save_dir = './model_store/attention.pth'
    net = Attention_model(hidden_dim=hidden_size).to(device) 
    # dummy_input = torch.rand(15,64,8).long().to(device)
    train_log_dir = './torch_log/train/' + TIMESTAMP
    test_log_dir = './torch_log/test/' + TIMESTAMP
    evaluate_log_dir = './torch_log/validate/' + TIMESTAMP
    net_log_dir = './torch_log/net/' + TIMESTAMP

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=decay_step, gamma=gamma)

    epochs = 100
    train(net, device, train_data, train_label, test_data, test_label, evaluate_data, evaluate_label, optimizer, epochs, scheduler, train_log_dir,test_log_dir,evaluate_log_dir,USE_BEST_MODEL,train_num,test_num,evaluate_num,save_dir)
