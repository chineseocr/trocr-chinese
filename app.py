import os
from PIL import Image
import time
import torch
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='-1', type=str, help="GPU设置")
    parser.add_argument('--test_img', default='test/test.jpg', type=str, help="img path")

    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()

    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}

    t = time.time()
    img = Image.open(args.test_img).convert('RGB')
    pixel_values = processor([img], return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values[:, :, :].cpu())

    generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
    print('time take:', round(time.time() - t, 2), "s ocr:", [generated_text])
