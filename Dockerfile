FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
LABEL version="0.1"  description="trocr-chinese" by="chineseocr"
ADD requirements.txt /trocr-chinese/requirements.txt
RUN pip install -r /trocr-chinese/requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
WORKDIR /trocr-chinese
CMD ["python", "app.py"]