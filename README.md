# 基于trocr(beit+roberta)实现对中文场景文字识别
trocr原地址(https://github.com/microsoft/unilm/tree/master/trocr)
## 实现功能
- [x]  单行/多行文字/横竖排文字识别
- [x]  不规则文字（印章，公式等）
- [ ]  表格识别
- [ ]  模型蒸馏/DML(协作学习)
- [ ]  Prompt Learning
## 环境编译
```
docker build --network=host -t trocr-chinese:latest .
docker run --gpus all -it -v /tmp/trocr-chinese:/trocr-chinese trocr-chinese:latest bash

```
## 训练
### 初始化模型到自定义训练数据集
#### 字符集准备参考cust-data/vocab.txt
```
vocab.txt
1
2
...
a
b
c
```
```[python]
python gen_vocab.py \
       --dataset_dataset_path "dataset/cust-data/0/*.txt" \
       --cust_vocab ./cust-data/vocab.txt

```
### 初始化自定义数据集模型
#### 下载预训练模型trocr模型权重
链接: https://pan.baidu.com/s/1rARdfadQlQGKGHa3de82BA  密码: 0o65
```
python init_custdata_model.py \   
    --cust_vocab ./cust-data/vocab.txt \  
    --pretrain_model ./weights \
    --cust_data_init_weights_path ./cust-data/weights
    
## cust_vocab 词库文件   
## pretrain_model 预训练模型权重   
## cust_data_init_weights_path 自定义模型初始化模型权重保存位置   

```

### 训练模型
#### 数据准备,数据结构如下图所示
```
dataset/cust-data/0/0.jpg
dataset/cust-data/0/0.txt
...
dataset/cust-data/100/10000.jpg
dataset/cust-data/100/10000.txt
```

#### 训练模型
```
python train.py \
       --cut_data_init_weights_path ./cust-data/weights \
       --checkpoint_path ./checkpoint/trocr-custdata \
       --dataset_path "./dataset/cust-data/*/*.jpg" \
       --per_device_train_batch_size 8 \
       --CUDA_VISIBLE_DEVICES 1
```

#### 评估模型
##### 拷贝checkpoint/trocr-custdata训练完成的pytorch_model.bin 到 ./cust-data/weights 目录下

```[python]
python eval.py \
    --dataset_path "./data/cust-data/test/*/*.jpg" \
    --cust_data_init_weights_path ./cust-data/weights    
```

## 测试模型
```
## 拷贝训练完成的pytorch_model.bin 到 ./cust-data/weights 目录下
index = 2300 ##选择最好的或者最后一个step模型
cp ./checkpoint/trocr-custdata/checkpoint-$index/pytorch_model.bin ./cust-data/weights
python app.py --cust_data_init_weights_path ./cust-data/weights --test_img test/test.jpg
```

## 预训练模型
| 模型        | cer(字符错误率)           | acc(文本行)  | 下载地址  |训练数据来源 |训练耗时(GPU:3090) | 
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| hand-write(中文手写)      |0.011 | 0.940 |[hand-write](https://pan.baidu.com/s/19f7iu9tLHkcT_zpi3UfqLQ)  密码: punl |[数据集地址](https://aistudio.baidu.com/aistudio/datasetdetail/102884/0) |8.5h(10epoch)|
| seal(印章识别)      |- | - |- |- |
| im2latex(数学公式识别)      |- | - |- |[im2latex](https://zenodo.org/record/56198#.YkniL25Bx_S) ||
| TAL_OCR_TABLE(表格识别)     |- | - |- |[TAL_OCR_TABLE](https://ai.100tal.com/dataset) |
| TAL_OCR_MATH(小学低年级算式数据集)|- | - |- | [TAL_OCR_MATH](https://ai.100tal.com/dataset) |
| TAL_OCR_CHN(手写中文数据集)|- | - |- | [TAL_OCR_CHN](https://ai.100tal.com/dataset) ||
| HME100K(手写公式)|- | - |- | [HME100K](https://ai.100tal.com/dataset) |

备注:后续所有模型会开源在这个目录下链接,可以自由下载. https://pan.baidu.com/s/1uSdWQhJPEy2CYoEULoOhRA  密码: vwi2
### 模型调用 
#### 手写识别
![image](img/hand.png)
```
unzip hand-write.zip 
python app.py --cust_data_init_weights_path hand-write --test_img test/hand.png

## output: '醒我的昏迷,偿还我的天真。'
```

#### 打印公式识别
![image](img/im2latex.png)
```
unzip im2latex.zip 
python app.py --cust_data_init_weights_path im2latex --test_img test/im2latex.png

```


## 捐助
如果此项目给您的工作带来了帮忙，希望您能贡献自己微薄的爱心,
该项目的每一份收入将用着福利事业，每一季度在issues上公布捐赠明细!   
![image](img/chat.jpg)

