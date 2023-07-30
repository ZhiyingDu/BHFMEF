import argparse

parser = argparse.ArgumentParser(description='BHFMEF')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--validation', '-v', action='store_true',
                    help='set this option to validate after training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Data specifications
parser.add_argument('--device', type=str, default='1',
                    help='device')
parser.add_argument('--dir_train', type=str, default='/data/dzy/Dataset/training_set/',
                    help='training dataset directory')
parser.add_argument('--loss_log', type=str, default='Your own path/Loss_log.txt',
                    help='loss_log')
parser.add_argument('--figure_path', type=str, default='Your own path/train_loss_curve.png',
                    help='figure_path')
parser.add_argument('--epoch', type=int, default=500,
                    help='the num of train')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='learning rate')
parser.add_argument('--Window_size', type=int, default=8,
                    help='Window_size')
parser.add_argument('--wd', type=float, default=1e-8,
                    help='weight_decay')
parser.add_argument('--dir_val', type=str, default='/data/dzy/Dataset/Val_set/',
                    help='validation dataset directory')
parser.add_argument('--dir_test', type=str, default='/data/dzy/Dataset/test_set1/',
                    help='test dataset directory')
parser.add_argument('--model_path', type=str, default='Your own path/model/',
                    help='trained model directory')
parser.add_argument('--model', type=str, default='Fusion.pth',
                    help='model name')
parser.add_argument('--GC_model', type=str, default='GCM.pth',
                    help='GC_model name')
parser.add_argument('--Denoise_model', type=str, default='ffdnet_gray.pth',
                    help='Denoise_model name')

parser.add_argument('--ext', type=str, default='.png',
                    help='extension of image files')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--batch_size', type=int, default=2,
                    help='number of batches each time')
parser.add_argument('--patch_size', type=int, default=256,
                    help='input patch size')
parser.add_argument('--save_dir', type=str, default='Your own path/result1/',
                    help='test results directory')

args = parser.parse_args()
