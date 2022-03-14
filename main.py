# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from core.utils.parse import ArgumentParser
from preprocess import main as preprocess
from script.translate import main as translate
from train import main as train

if __name__ == '__main__':
    parser = ArgumentParser()

    # Simply add an argument for preprocess, train, translate
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preprocess", dest='preprocess', action='store_true',
                      help="Activate to preprocess with OpenNMT")
    mode.add_argument("--train", dest='train', action='store_true',
                      help="Activate to train with OpenNMT")
    mode.add_argument("--translate", dest='translate', action='store_true',
                      help="Activate to translate with OpenNMT")

    mode, remaining_args = parser.parse_known_args()

    if mode.preprocess:
        preprocess(remaining_args)
    elif mode.train:
        train(remaining_args)
    elif mode.translate:
        args = translate(remaining_args)
