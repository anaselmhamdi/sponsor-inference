FROM python:3.6-slim
COPY main.py /deploy/
COPY models.py /deploy/
COPY train.py /deploy/
COPY utils.py /deploy/
COPY inference.py /deploy/
COPY requirements.txt /deploy/
RUN apt install -y wget
ADD https://anas-models.s3.amazonaws.com/tut7-model.pt /deploy/
WORKDIR /deploy/
RUN apt update
RUN apt install -y git
RUN apt install -y wget
RUN pip install -r requirements.txt
EXPOSE 8080
# ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port 8080
