import argparse

def _parse_args():
    parser = argparse.ArgumentParser(description='MalCL')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('--num_rounds', default=30, type=int, help='Number of global rounds per task.')
    parser.add_argument('--local_epochs', default=1, type=int, help='Number of local epochs per round.')
    parser.add_argument('--debug', action='store_true', help='Quick test with limited samples and epochs.')

    parser.add_argument(
        '--init_classes', type=int, default=6, help='number of classes for initial task.'
    )
    parser.add_argument(
        '--final_classes', type=int, default=34, help='number of total classes for training.'
    )
    parser.add_argument(
        '--n_inc', type=int, default=6, help='number of incremental classes for each task (avg).'
    )
    parser.add_argument(
        '--nb_task', type=int, default=6, help='number of tasks.'
    )

    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate'
    )

    parser.add_argument(
        '--weight_decay', type=float, default=0.000001, help='Weight decay for the Classifier (SGD).'
    )

    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum for the Classifier (SGD).'
    )

    parser.add_argument(
        '--batchsize', type=int, default=8192
    )

    parser.add_argument(
        '--sample_select', type=str, default='L1_C_Mean', choices=['L2_One_Hot', 'L1_B_Mean', 'L1_C_Mean']
        , help='choose among the 3 of sample selection schemes. (L2_One_Hot, L1_B_Mean, L1_C_Mean)'
    )

    parser.add_argument(
        '--k', type=int, default=100, help='k is a variable related to the selected number of synthetic samples on a class in each batch'
    )

    parser.add_argument(
        '--g_iter', type=int, default=2, help='Number of Generator/Discriminator updates per batch.'
    )

    parser.add_argument(
        '--replay_weight', type=float, default=2.0, help='Weight multiplier for replay samples in the loss function.'
    )

    parser.add_argument(
        '--Generator_loss', type=str, default='FML', choices=['FML', 'BCE']
        , help='choose among the 2 Loss functions for Generator training. (FML, BCE)'
    )

    parser.add_argument('--train_data', default = '', help='Path of train data')
    parser.add_argument('--test_data', default = '', help='Path of test data')
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--seed_', default=20, type=int, help='random seed for class order')

    # === Mode: train or test ===
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train: chay training va luu checkpoint. test: chi load checkpoint va eval.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Thu muc de luu/doc checkpoint.')
    parser.add_argument('--test_checkpoint_dir', type=str, default='',
                        help='Thu muc checkpoint cu the khi chay test mode.')
    parser.add_argument('--resume', type=str, default='',
                        help='Duong dan den file .pth de resume training.')

    return parser.parse_args()
