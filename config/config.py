import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for train [default: 256]')
parser.add_argument('--batch', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('--maxSen', type=int, default=0, help='max Sentence [default:0]')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum [default:0.9]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-optim', type=str, default='Adam', help='optimizer [Adam, SGD]')
parser.add_argument('--model', type=str, default='HCL', help='select the model(HCL, CLSTM, CNN, RNN,...)')
parser.add_argument('--mode', type=str, default='train', help='select the mode(train or eval)')
parser.add_argument('--sampling', type=bool, default=False, help='whether to sampling data')
parser.add_argument('--tokenizer', type=str, default='word', help='select the tokenizer(char, word)')
parser.add_argument('--data_train_file', type=str, default='./nsmc/ratings_train.tsv', help='select the model(HCL, CLSTM, CNN, RNN,...)')
parser.add_argument('--data_test_file', type=str, default='./nsmc/ratings_test.tsv', help='select the model(HCL, CLSTM, CNN, RNN,...)')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')

#model_CLSTM1
parser.add_argument('--vocab', type=int, default=1000000, help='number of CLSTM1_filter [default: 100]')
parser.add_argument('-CLSTM1_kernel', type=list, default=[3,4,5], help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-CLSTM1_filter', type=int, default=100, help='number of CLSTM1_filter [default: 100]')
parser.add_argument('-CLSTM1_embed', type=int, default=300, help='number of CLSTM1_embed [default: 100]')
parser.add_argument('-CLSTM1_dim', type=int, default=100, help='number of CLSTM1_embed [default: 100]')
parser.add_argument('-CLSTM1_lstm_layer', type=int, default=1, help='number of CLSTM1_layer [default: 1]')


#model_CLSTM2
parser.add_argument('-CLSTM2_kernel', type=list, default=[3,4,5], help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-CLSTM2_filter', type=int, default=100, help='number of CLSTM2_filter [default: 100]')
parser.add_argument('-CLSTM2_embed', type=int, default=300, help='number of CLSTM2_embed [default: 100]')
parser.add_argument('-CLSTM2_dim', type=int, default=100, help='number of CLSTM2_embed [default: 100]')
parser.add_argument('-CLSTM2_lstm_layer', type=int, default=1, help='number of CLSTM2_layer [default: 1]')

#model_wordCLSTM
parser.add_argument('-wordCLSTM_kernel', type=list, default=[3,4,5], help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-wordCLSTM_filter', type=int, default=100, help='number of wordCLSTM_filter [default: 100]')
parser.add_argument('-wordCLSTM_embed', type=int, default=300, help='number of wordCLSTM_embed [default: 100]')
parser.add_argument('-wordCLSTM_dim', type=int, default=100, help='number of wordCLSTM_embed [default: 100]')
parser.add_argument('-wordCLSTM_lstm_layer', type=int, default=1, help='number of wordCLSTM_layer [default: 1]')

#model sentCLSTM
parser.add_argument('-sentCLSTM_kernel', type=list, default=[3,4,5], help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-sentCLSTM_filter', type=int, default=100, help='number of sentCLSTM_filter [default: 100]')
parser.add_argument('-sentCLSTM_embed', type=int, default=200, help='number of sentCLSTM_embed [default: 100]')
parser.add_argument('-sentCLSTM_dim', type=int, default=100, help='number of sentCLSTM_embed [default: 100]')
parser.add_argument('-sentCLSTM_lstm_layer', type=int, default=1, help='number of sentCLSTM_layer [default: 1]')

parser.add_argument('-num_class', type=int, default=2, help='number of classes [default: 2]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')

#model LSTM
parser.add_argument('-LSTM_embed', type=int, default=100, help='number of LSTM_embed [default: 100]')
parser.add_argument('-LSTM_hidden', type=int, default=100, help='number of LSTM_hidden [default: 100]')
parser.add_argument('-LSTM_layer', type=int, default=1, help='number of LSTM_layer [default: 1]')

#model LSTM_Maxpool
parser.add_argument('-LSTM_Maxpool_embed', type=int, default=100, help='number of LSTM_Maxpool_embed [default: 100]')
parser.add_argument('-LSTM_Maxpool_hidden', type=int, default=100, help='number of LSTM_Maxpool_hidden [default: 100]')
parser.add_argument('-LSTM_Maxpool_layer', type=int, default=1, help='number of LSTM_Maxpool_layer [default: 1]')



#model LSTM_Attn
parser.add_argument('-LSTM_Attn_embed', type=int, default=100, help='number of LSTM_Attn_embed [default: 100]')
parser.add_argument('-LSTM_Attn_hidden', type=int, default=100, help='number of LSTM_Attn_hidden [default: 100]')
parser.add_argument('-LSTM_Attn_layer', type=int, default=1, help='number of LSTM_Attn_layer [default: 1]')

#model CNN
parser.add_argument('-CNN_embed', type=int, default=100, help='number of CNN_embed [default: 100]')
parser.add_argument('-CNN_kernels', type=list, default=[3,4,5], help='number of CNN_kernel [default: [3,4,5]]')
parser.add_argument('-CNN_filter', type=int, default=100, help='number of CNN_filter [default: 100]')
parser.add_argument('-CNN_dropout', type=int, default=0.5, help='number of CNN_dropout [default: 0.5]')



# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

