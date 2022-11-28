import argparse
from constants import DIALOGSUM, SAMSUM
def create_args(is_training=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_length', default=1024 , type=int)
    parser.add_argument('--model_name', default='facebook/bart-base')
    parser.add_argument('--dataset', choices=[DIALOGSUM, SAMSUM])
    parser.add_argument('--data_dir', default='../data/samsum')
    parser.add_argument('--warmup_epoch', default=3, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--eval_batch_size', default=2, type=int)
    parser.add_argument('--accumulate_step', default=32, type=int)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--use_robust', action='store_true')
    parser.add_argument('--use_nli', action='store_true')
    parser.add_argument('--checkpoint_path', type=str)
    # parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--disable_eval', action='store_true')
    parser.add_argument('--do_invalid', action='store_true')
    parser.add_argument('--do_uncovered', action='store_true')
    parser.add_argument('--do_gradient_checkpointing', action='store_true')
    parser.add_argument('--do_factcc_validate', action='store_true')
    parser.add_argument('--do_factcc_uncovered', action='store_true')
    parser.add_argument('--uncovered_weights', default=1, type=float)
    parser.add_argument('--invalid_weights', default=1, type=float)
    parser.add_argument('--use_augmentation', action='store_true' )
    parser.add_argument('--debug', action='store_true' )
    
    if is_training:
        parser.add_argument('--exp_name', required=True, type=str)
    else:
        parser.add_argument('--test_rouge', action='store_true' )
    args = parser.parse_args()

    return args