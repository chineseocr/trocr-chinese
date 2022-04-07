import os
from PIL import Image
import numpy as np
import time
import torch
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
from tqdm import tqdm
from datasets import load_metric

cer_metric = load_metric("./cer.py")


def compute_metrics(pred_str, label_str):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    acc = sum(acc) / (len(acc) + 0.000001)
    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='-1', type=str, help="GPU设置")
    parser.add_argument('--dataset_path', default='dataset/HW-hand-write/HW_Chinese/*/*.[j|p]*', type=str,
                        help="img path")
    parser.add_argument('--random_state', default=None, type=int, help="用于训练集划分的随机数")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    paths = glob(args.dataset_path)
    if args.random_state is not None:
        train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=args.random_state)

    else:
        train_paths = []
        test_paths = paths

    print("train num:", len(train_paths), "test num:", len(test_paths))

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()
    model.cuda()

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}

    pred_str, label_str = [], []
    for p in tqdm(test_paths):
        img = Image.open(p).convert('RGB')
        txt_p = os.path.splitext(p)[0] + '.txt'
        with open(txt_p) as f:
            label = f.read().strip()
        pixel_values = processor([img], return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values[:, :, :].cuda())

        generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
        pred_str.append(generated_text)
        label_str.append(label)

    res = compute_metrics(pred_str, label_str)
    print(res)
