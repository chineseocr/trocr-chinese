"""
转换trocr 模型到自己数据集上的字符进行fine-tune
"""
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoConfig


def read_vocab(vocab_path):
    """
    读取自定义训练字符集
    vocab_path format:
    1\n
    2\n
    ...
    我\n
    """
    other = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    vocab = {}
    for ot in other:
        vocab[ot] = len(vocab)

    with open(vocab_path) as f:
        for line in f:
            line = line.strip('\n')
            if line not in vocab:
                vocab[line] = len(vocab)
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')

    parser.add_argument('--cust_vocab', default="./cust-data/vocab.txt", type=str, help="自定义训练数字符集")

    parser.add_argument('--pretrain_model', default='./weights', type=str, help="预训练bert权重文件")

    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.pretrain_model)
    pre_model = VisionEncoderDecoderModel.from_pretrained(args.pretrain_model)

    pre_vocab = processor.tokenizer.get_vocab()

    cust_vocab = read_vocab(args.cust_vocab)

    keep_tokens = []
    unk_index = pre_vocab.get('<unk>')
    for key in cust_vocab:
        keep_tokens.append(pre_vocab.get(key, unk_index))

    processor.save_pretrained(args.cust_data_init_weights_path)

    pre_model.save_pretrained(args.cust_data_init_weights_path)
    ## 替换词库
    with open(os.path.join(args.cust_data_init_weights_path, "vocab.json"), "w") as f:
        f.write(json.dumps(cust_vocab, ensure_ascii=False))

    ##替换模型参数
    with open(os.path.join(args.cust_data_init_weights_path, "config.json")) as f:
        model_config = json.load(f)

    ## 替换roberta embedding层词库
    model_config["decoder"]['vocab_size'] = len(cust_vocab)

    ## 替换 attetion 字库
    model_config['vocab_size'] = len(cust_vocab)

    with open(os.path.join(args.cust_data_init_weights_path, "config.json"), "w") as f:
        f.write(json.dumps(model_config, ensure_ascii=False))

    ##加载cust model
    cust_config = AutoConfig.from_pretrained(args.cust_data_init_weights_path)
    cust_model = VisionEncoderDecoderModel(cust_config)

    pre_model_weigths = pre_model.state_dict()
    cust_model_weigths = cust_model.state_dict()

    ##权重初始化
    print("loading init weights..................")
    for key in pre_model_weigths:
        print("name:", key)
        if pre_model_weigths[key].shape != cust_model_weigths[key].shape:
            wt = pre_model_weigths[key][keep_tokens, :]
            cust_model_weigths[key] = wt
        else:
            cust_model_weigths[key] = pre_model_weigths[key]

    cust_model.load_state_dict(cust_model_weigths)
    cust_model.save_pretrained(args.cust_data_init_weights_path)
