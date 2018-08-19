import argparse
import utils
from model import TranscriptNet


ap = argparse.ArgumentParser()

ap.add_argument("--mode", required=True, type=str, help="train|test",
                choices=["train", "test"])
ap.add_argument("--train_dir", default="./train",
                help="path to the directory where trained model is to be stored")
ap.add_argument("--load_model", default="./train/model.ckpt",
                help="path to trained model")
ap.add_argument("--data_path", default="./data/rick_and_morty.txt",
                help="path to text file where the data is stored")
ap.add_argument("--log_dir", default="./graph",
                help="path to the directory where tensorboard logs are to be written")
ap.add_argument("--epochs", type=int, default=20,
                help="number of epochs")
ap.add_argument("--checkpoint_step", type=int, default=100,
                help="number of steps at which checkpoints should be saved")
ap.add_argument("--batch_size", type=int, default=64,
                help="size of mini-batch")
ap.add_argument("--seq_length", type=int, default=100,
                help="numbers of time-steps in sequence")
ap.add_argument("--learning_rate", type=float, default=0.001,
                help="learning rate")
ap.add_argument("--embed_size", type=int, default=300,
                help="number of dimensions in word embeddings")
ap.add_argument("--lstm_size", type=int, default=512,
                help="number of units in lstm")
ap.add_argument("--lstm_layers", type=int, default=1,
                help="number of layers in lstm network")
ap.add_argument("--temperature", type=float, default=1.0,
                help="higher value means more random words will be picked and lower value means less randomness")
ap.add_argument("--dropout", type=float, default=0.3,
                help="dropout rate")
ap.add_argument("--resume", action="store_true",
                help="resume training from last checkpoint")

args = ap.parse_args()

model = TranscriptNet(args)

if args.mode == "train":
    text_seq = utils.preprocess(args.data_path)
    model.train(text_seq)
else:
    model.generate()
