import argparse

from nltk.translate.bleu_score import corpus_bleu
from script.data_utils import FileIterable
import os
from parent.parent import parent
from tqdm import tqdm


def _corpus_bleu(hypotheses, references):
    return corpus_bleu([[r for r in refs if r] for refs in zip(*references)], hypotheses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute BLEU and PARENT metrics.')
    parser.add_argument('--tables', dest="tables")
    parser.add_argument('--references', dest="references", nargs='+')
    parser.add_argument('--hypotheses', dest="hypotheses")

    args = parser.parse_args()

    tables = FileIterable.from_filename(args.tables, fmt='jl').to_list('Reading Tables')
    references = [FileIterable.from_filename(filename, fmt='txt').to_list(f'Reading References {idx}')
                  for idx, filename in enumerate(args.references, 1)]

    file_list = os.listdir(args.hypotheses)
    hypotheses_list = []
    for file in file_list:
        print(args.hypotheses + file)
        hypotheses = FileIterable.from_filename(args.hypotheses + file, fmt='txt').to_list('Reading Predictions')
        hypotheses_list.append(hypotheses)

    print('Computing BLEU... ', end='')
    bleu_sum = 0
    parent_pre = 0
    parent_recall = 0
    parent_F1 = 0
    error_num = 0

    for hypotheses in hypotheses_list:
        bleu = 0
        flag = False
        try:
            bleu = _corpus_bleu(hypotheses, references)
        except:
            bleu = 0
            error_num += 1
            flag = True
        finally:
            bleu_sum += bleu

        parent_p, parent_r, parent_f = 0, 0, 0
        try:
            parent_p, parent_r, parent_f = parent(hypotheses, references, tables)
        except:
            parent_p, parent_r, parent_f = 0, 0, 0
            if not flag:
                error_num += 1
        finally:
            parent_F1 += parent_f
            parent_pre += parent_p
            parent_recall += parent_r


    print('OK')
    count = len(hypotheses_list) - error_num
    bleu = bleu_sum / count
    # print(f'\n{args.hypotheses}:\nBLEU\t{bleu:.4f}')
    parent_p = parent_pre / count
    parent_r = parent_recall / count
    parent_f = parent_F1 / count
    # print('Computing PARENT... ', end='')
    ## references = [r[0] for r in references]
    # parent_p, parent_r, parent_f = parent(hypotheses, references, tables)
    # print('OK')

    print(f'\n{args.hypotheses}:\nBLEU\t{bleu:.4f}\n'
          f'PARENT (precision)\t{parent_p:.4f}\n'
          f'PARENT (recall)\t{parent_r:.4f}\n'
          f'PARENT (F1)\t{parent_f:.4f}\n')
