"""
python -m \
    transformers.onnx \
    --model=hand-write \
    --feature=vision2seq-lm \
    hand-write --atol 1e-4
"""
import cv2
import numpy as np
import onnxruntime
import json
import os
import argparse
def read_vocab(path):
    """
    加载词典
    """
    with open(path) as f:
        vocab = json.load(f)
    return vocab

def do_norm(x):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    x = x/255.0
    x[0, :, :] -= mean[0]
    x[1, :, :] -= mean[1]
    x[2, :, :] -= mean[2]
    x[0, :, :] /= std[0]
    x[1, :, :] /= std[1]
    x[2, :, :] /= std[2]
    return x

def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:

        if tk == s_end:
            break
        if tk not in [s_end, s_start, pad, unk]:
            text += vocab_inp[tk]

    return text

class OnnxEncoder():
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())

    def __call__(self, img):
        onnx_inputs = {self.model.get_inputs()[0].name: np.asarray(img, dtype='float32')}
        onnx_output = self.model.run(None, onnx_inputs)[0]
        return onnx_output

class OnnxDecoder():
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.model.get_inputs())}

    def __call__(self, input_ids,
                 encoder_hidden_states,
                 attention_mask):

        onnx_inputs = {"input_ids": input_ids}
        onnx_inputs["attention_mask"] = attention_mask
        onnx_inputs["encoder_hidden_states"] = encoder_hidden_states

        onnx_output = self.model.run(['logits'], onnx_inputs)
        return onnx_output

class onnxEncoderDecoder():
    def __init__(self, model_path):
        self.encoder = OnnxEncoder(os.path.join(model_path, "encoder_model.onnx"))
        self.decoder = OnnxDecoder(os.path.join(model_path, "decoder_model.onnx"))
        self.vocab = read_vocab(os.path.join(model_path, "vocab.json"))
        self.vocab_inp = {self.vocab[key]: key for key in self.vocab}

    def run(self, image):
        """
        rgb:image
        """
        image = cv2.resize(image, (384, 384))
        pixel_values = cv2.split(np.array(image))
        pixel_values = do_norm(np.array(pixel_values))
        pixel_values = np.array([pixel_values])
        encoder_output = self.encoder(pixel_values)
        ids = [self.vocab["<s>"], ]
        mask = [1, ]
        for i in range(100):
            input_ids = np.array([ids])
            attention_mask = np.array([mask])
            decoder_output = self.decoder(input_ids=input_ids,
                                     encoder_hidden_states=encoder_output,
                                     attention_mask=attention_mask
                                     )
            pred = decoder_output[0][0]
            pred = pred.argmax(axis=1)
            if pred[-1] == self.vocab["</s>"]:
                break
            ids.append(pred[-1])
            mask.append(1)

        text = decode_text(ids, self.vocab, self.vocab_inp)
        return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='onxx model test')
    parser.add_argument('--model', type=str,
                            help="onnx 模型地址")
    parser.add_argument('--test_img', type=str, help="测试图像")

    args = parser.parse_args()
    model = onnxEncoderDecoder(args.model)
    img = cv2.imread(args.test_img)
    img = img[..., ::-1] ##BRG to RGB
    res = model.run(img)
    print(res)


