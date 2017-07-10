import os
import argparse
import random 

import torch
import torch.backends.cudnn as cudnn

# add path 
import sys
# add absolute path
sys.path.insert(0, "/home/hxw/project_work_on/machine_translation_research/pytorch-seq2seq")

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import Dataset
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from IPython.core.debugger import Tracer 
debug_here = Tracer() 

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH -resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()

# cuda stuff
parser.add_argument('--gpu_id'  , type=str, default='1', help='which gpu to use, used only when ngpu is 1')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
# clamp parameters into a cube

parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')

opt = parser.parse_args()
print(opt)



os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

ngpu = int(opt.ngpu)
# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 123456

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if ngpu == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(opt.gpu_id))

        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)



if opt.load_checkpoint is not None:
    print("loading checkpoint...")
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    dataset = Dataset(opt.train_path, src_max_len=50, tgt_max_len=50)
    input_vocab = dataset.input_vocab
    output_vocab = dataset.output_vocab

    dev_set = Dataset(opt.dev_path, src_max_len=50, tgt_max_len=50,
                    src_vocab=input_vocab,
                    tgt_vocab=output_vocab)

    # Prepare model
    hidden_size=128
    encoder = EncoderRNN(input_vocab, dataset.src_max_len, hidden_size)
    decoder = DecoderRNN(output_vocab, dataset.tgt_max_len, hidden_size,
                        dropout_p=0.2, use_attention=True)

    #############################################################
    # teacher forcing ratio is 0 by default
    # so we train the model by feeding predicted symbol 
    #############################################################
    seq2seq = Seq2seq(encoder, decoder)
    if opt.resume:
        print("resuming training")
        latest_checkpoint = Checkpoint.get_latest_checkpoint(opt.expt_dir)
        seq2seq.load(latest_checkpoint)
    else:
        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # Prepare loss
    weight = torch.ones(output_vocab.get_vocab_size())
    mask = output_vocab.MASK_token_id
    loss = Perplexity(weight, mask)

    if torch.cuda.is_available():
        seq2seq.cuda()
        loss.cuda()

    # train and validation use the same teacher forcing technique
    # if teacher_forcing_ratio is set to 1, then the model is trained and validated by feeding target symbol
    t = SupervisedTrainer(loss=loss, batch_size=32,
                        checkpoint_every=50,
                        print_every=10, expt_dir=opt.expt_dir)
    
    #############################################################
    # teacher forcing ratio is 0 by default
    # so we train the model by feeding predicted symbol 
    #############################################################
    t.train(seq2seq, dataset, num_epochs=4, dev_data=dev_set, resume=opt.resume)



predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = input("Type in a source sequence(ex, 1, 2, 3):")
    seq = seq_str.split()
    debug_here()
    print(predictor.predict(seq))
