import argparse
import logging
import random

import torch

from optimizer import TrainingController, EvaluationController, OptimizationController

logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='cluster-scheduler-configuration-optimizer')
    parser.add_argument('--java-home', type=str, default='/home/ls/library/jdk', help='java home path')
    parser.add_argument('--hadoop-home', type=str, default='/home/ls/library/hadoop', help='Hadoop home path')
    parser.add_argument('--spark-home', type=str, default='/home/ls/library/spark', help='Spark home path')
    parser.add_argument('--resource-manager-host', type=str, default='http://10.1.114.60:8088/', help='Address:port of ResourceManager')
    parser.add_argument('--spark-history-server-host', type=str, default='http://10.1.114.60:18080/', help='Address:port of Spark history server')
    parser.add_argument('--use-simulation-env', action='store_true', help='Using simulation environment')
    parser.add_argument('--simulation-host', type=str, default='localhost', help='Address:port of Spark history server')
    parser.add_argument('--execution-mode', type=int, default=int(0), help='Set program execution mode.')
    parser.add_argument('--log-into-file', action='store_true', help='Redirect log to file')
    parser.add_argument('--log-filename', type=str, default='./results/runtime.log', help='Runtime log filename.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, default='results/model.pth', metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1000000), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(80000), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=288, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--log-interval', type=int, default=288, metavar='STEPS', help='Number of training steps between logging status')

    return parser


# Setup PyTorch args
def setup_torch_args(args: argparse.Namespace):
    logger.info('Options')
    for k, v in vars(args).items():
        logger.info(k + ': ' + str(v))
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda:0')
        torch.cuda.manual_seed(random.randint(1, 10000))
        # Disable nondeterministic ops (not sure if critical but better safe than sorry)
        torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    return args


def get_args() -> argparse.Namespace:
    parser = setup_arg_parser()
    args = parser.parse_args()
    logging_config = {
        'level': logging.INFO,
        'format': '%(asctime)s %(levelname)s %(funcName)s: %(message)s'
    }
    if args.log_into_file:
        logging_config['filename'] = args.log_filename
        logging_config['filemode'] = 'a'
    logging.basicConfig(**logging_config)
    setup_torch_args(args)
    return args


def main():
    # noinspection PyBroadException
    try:
        args = get_args()

        controller_classes = {
            0: OptimizationController,
            1: EvaluationController,
            2: TrainingController,
        }

        execution_mode: int = args.execution_mode
        controller = controller_classes[execution_mode](args)
        controller.run()
    except InterruptedError:
        logger.warning('Met Interrupted Error. Closing environment...')
    except Exception:
        logger.exception('Something bad happened.')


if __name__ == '__main__':
    main()
