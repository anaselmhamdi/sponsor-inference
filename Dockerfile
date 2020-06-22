FROM python:3.6-slim
COPY app/main.py /deploy/
COPY app/models.py /deploy/
COPY app/train.py /deploy/
COPY app/utils.py /deploy/
COPY app/inference.py /deploy/
COPY requirements.txt /deploy/
RUN apt install -y wget
ADD wget https://anas-models.s3.amazonaws.com/full-model.pt /deploy/
WORKDIR /deploy/
RUN apt update
RUN apt install -y git
RUN apt install -y youtube-dl
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT uvicorn main:app --port 8080
