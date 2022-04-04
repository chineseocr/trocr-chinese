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
3
4
5
...
a
b
c
```
### 初始化自定义数据集模型
#### 下载预训练模型trocr模型权重
链接: https://pan.baidu.com/s/1rARdfadQlQGKGHa3de82BA  密码: 0o65
```
python init_custdata_model.py \   
    --cust_vocab ./cust-data/vocab.txt \  
    --pretrain_model ./weights \
    --cust_data_init_weights_path ./cust-data/weights
```
## cust_vocab 词库文件   
## pretrain_model 预训练模型权重   
## cut_data_init_weights_path 自定义模型初始化模型权重保存位置   

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

## 测试模型
```
## 拷贝训练完成的pytorch_model.bin 到 ./cust-data/weights 目录下
index = 2300
cp ./checkpoint/trocr-custdata/checkpoint-$index/pytorch_model.bin ./cust-data/weights
python app.py --test_img test/test.jpg
```

## 预训练模型
| 模型        | cer(字符错误率)           | acc(文本行)  | 下载地址  |训练数据来源 |训练耗时(GPU:3090) | 
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| hand-write(中文手写)      |0.011 | 0.940 |链接: https://pan.baidu.com/s/19f7iu9tLHkcT_zpi3UfqLQ  密码: punl |https://aistudio.baidu.com/aistudio/datasetdetail/102884/0 |8.5h|
| 印章识别      |- | - |- |- |
| im2latex(数学公式识别)      |- | - |- |https://zenodo.org/record/56198#.YkniL25Bx_S |
| 表格识别      |- | - |- |链接：https://pan.baidu.com/s/1V0NT2XmQDDb0mHQlw7V7_w 提取码：oo4a |

备注:后续所有模型会开源在这个目录下链接,可以自由下载. https://pan.baidu.com/s/1uSdWQhJPEy2CYoEULoOhRA  密码: vwi2
### 模型调用 
```
unzip hand-write.zip 
python app.py --cust_data_init_weights_path hand-write --test_img test/hand.png
```
## 捐助
如果此项目给您的工作带来了帮忙，希望您能贡献自己微薄的爱心,
该项目的每一份收入将用着福利事业，每一季度在issues上公布捐赠明细!   
![image](img/chat.jpg

