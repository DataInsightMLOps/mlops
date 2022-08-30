FROM python:3.9-slim

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY timm  timm

COPY . .

ENTRYPOINT ["python3", "inference.py"]
