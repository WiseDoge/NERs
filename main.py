import argparse
import os
import torch

from initialize import init
from eval import do_eval
from train import do_train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default='./data/ResumeNER/train.char.bmes')
    parser.add_argument("--dev_file", type=str, default='./data/ResumeNER/dev.char.bmes')
    parser.add_argument("--test_file", type=str, default='./data/ResumeNER/test.char.bmes')
    
    parser.add_argument("--word_dict_path", type=str, default='./dict/resume_word2ix.dict')
    parser.add_argument("--tag_dict_path", type=str, default='./dict/resume_tag2ix.dict')

    parser.add_argument("--output_dir", type=str, default='output', 
                        help="The output directory where the model params saved/will be saved.")
    parser.add_argument("--eval_log_dir", type=str, default='evallog', 
                        help="The output directory of evaluation results.")

    parser.add_argument("--do_init",
                        action='store_true',
                        help="Whether to initialization.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--print_step", type=int, default=20)

    
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def main():
    args = get_args()
    f = os.path.normpath
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.do_init:
        init(f(args.train_file), f(args.dev_file), f(args.test_file), 
             f(args.word_dict_path), f(args.tag_dict_path))
        return
    if args.do_train:
        do_train(f(args.train_file), f(args.output_dir),f(args.word_dict_path), f(args.tag_dict_path),
                 args.max_seq_len, args.embed_dim, args.hidden_dim, 
                 args.lr, args.batch_size, args.epochs, 
                 args.print_step, device)
    if args.do_eval:
        do_eval(f(args.test_file), f(args.word_dict_path), f(args.tag_dict_path), 
                args.max_seq_len, args.embed_dim, args.hidden_dim, 
                f(args.output_dir), f(args.eval_log_dir), device)
    
    
if __name__ == "__main__":
    main()