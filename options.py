import argparse

'''参数设置'''
parser = argparse.ArgumentParser()
'''训练前准备'''
# Dataset model
parser.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="train or test")
# Dataset参数
# parser.add_argument("--HR_image_dir", type=str, default="./data/noise injection/train/clean", help="path to folder for getting clean_dataset")
# parser.add_argument("--LR_image_dir", type=str, default="./data/noise injection/train/noise", help="path to folder for getting noise_dataset")
# parser.add_argument("--HR_image_dir", type=str, default="./data/train/gt", help="path to folder for getting clean_dataset")
# parser.add_argument("--LR_image_dir", type=str, default="./data/train/img", help="path to folder for getting noise_dataset")
parser.add_argument("--HR_image_dir", type=str, default="./data/train/clean", help="path to folder for getting clean_dataset")
parser.add_argument("--LR_image_dir", type=str, default="./data/train/noisy", help="path to folder for getting noise_dataset")

parser.add_argument("--batch_size", type=int, default=2, help="batch size")
# parser.add_argument("--dataset", type=str, default="rose", choices=["rose", "cria", "drive"],help="dataset")   # choices可扩展
# parser.add_argument("--data_root", type=str, default="./data/ROSE-1/DVC/", help="path to folder for getting dataset")
# parser.add_argument("--input_nc", type=int, default=3, choices=[1, 3], help="gray or rgb")
# parser.add_argument("--crop_size", type=int, default=64, help="crop size")
# parser.add_argument("--scale_size", type=int, default=64, help="scale size (applied in drive and cria)")
'''train'''
parser.add_argument("--lr", type=float, default=0.0005, help="initial learning rate")
parser.add_argument("--epoch", type=int, default=200, help="epoch")
parser.add_argument('--step', type=int, default=30,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=30')
# parser.add_argument("--power", type=float, default=0.9, help="power")
# parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
# parser.add_argument("--first_epochs", type=int, default=200, help="first stage epoch")
# parser.add_argument("--val_epoch_freq", type=int, default=20, help="frequency of validation at the end of epochs")
# parser.add_argument("--save_epoch_freq", type=int, default=20, help="frequency of saving models at the end of epochs")
'''train second'''
parser.add_argument("--num_workers", type=int, default=0, help="number of threads")
parser.add_argument("--base_channels", type=int, default=64, help="basic channels")
parser.add_argument("--second_epochs", type=int, default=300, help="train epochs of second stage")
parser.add_argument("--pn_size", type=int, default=3, help="size of propagation neighbors")
'''result'''
parser.add_argument("--logs_dir", type=str, default="logs", help="path to folder for saving logs")
parser.add_argument("--models_dir", type=str, default="models", help="path to folder for saving models")
parser.add_argument("--first_suffix", type=str, default="best_fusion.pth",
                    help="front_model-[model_suffix].pth will be loaded in models_dir")
parser.add_argument("--first_suffix1", type=str, default="best_fusion.pth",
                    help="front_model-[model_suffix].pth will be loaded in models_dir")
parser.add_argument("--results_dir", type=str, default="results", help="path to folder for saving results")
args = parser.parse_args()

