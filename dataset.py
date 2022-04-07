import os
from PIL import Image
from torch.utils.data import Dataset
import torch
class trocrDataset(Dataset):
    """
    trocr 训练数据集处理
    文件数据结构
    /tmp/0/0.jpg #image
    /tmp/0/0.txt #text label
    ....
    /tmp/100/10000.jpg #image
    /tmp/100/10000.txt #text label
    """
    def __init__(self, paths, processor, max_target_length=128, transformer=lambda x:x):
        self.paths = paths
        self.processor = processor
        self.transformer = transformer
        self.max_target_length = max_target_length
        self.nsamples = len(self.paths)
        self.vocab = processor.tokenizer.get_vocab()

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
            inx = idx %  self.nsamples
            image_file = self.paths[idx]
            txt_file = os.path.splitext(image_file)[0]+'.txt'

            with open(txt_file) as f:
                text = f.read().strip().replace('xa0','')
                if text.startswith('[') and text.endswith(']'):
                    ##list
                    try:
                       text = json.loads(text)
                    except:
                         pass

            image = Image.open(image_file).convert("RGB")
            image = self.transformer(image) ##图像增强函数
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            #labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)["input_ids"]
            labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

            return encoding


def encode_text(text, max_target_length=128, vocab=None):
    """
    ##自持自定义 list: ['<td>',"3","3",'</td>',....]
    {'input_ids': [0, 1092, 2, 1, 1],
    'attention_mask': [1, 1, 1, 0, 0]}
    """
    if type(text) is not list:
       text = list(text)

    text = text[:max_target_length - 2]
    tokens = [vocab.get('<s>')]
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    mask = []
    for tk in text:
        token = vocab.get(tk, unk)
        tokens.append(token)
        mask.append(1)

    tokens.append(vocab.get('</s>'))
    mask.append(1)

    if len(tokens) < max_target_length:
        for i in range(max_target_length - len(tokens)):
            tokens.append(pad)
            mask.append(0)

    return tokens
    #return {"input_ids": tokens, 'attention_mask': mask}


def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:
        if tk not in [s_end, s_start , pad, unk]:
           text += vocab_inp[tk]

    return text



