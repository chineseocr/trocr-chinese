import os.path
from glob import glob
from tqdm import tqdm
import codecs
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='trocr vocab生成')
    parser.add_argument('--cust_vocab', default="./cust-data/vocab.txt", type=str, help="自定义vocab文件生成")
    parser.add_argument('--dataset_path', default="./dataset/train/*/*.jpg", type=str, help="自定义训练数字符集")
    args = parser.parse_args()
    paths = glob(args.dataset_path)
    vocab = set()
    for p in tqdm(paths):
        with codecs.open(p, encoding='utf-8') as f:
            txt = f.read().strip()
        vocab.update(txt)
    root_path = os.path.split(args.cust_vocab)[0]
    os.makedirs(root_path, exist_ok=True)
    with open(args.cust_vocab, 'w') as f:
        f.write('\n'.join(list(vocab)))





