FROM python:3.6-slim
RUN apt update
RUN apt install -y git
RUN wget https://anas-models.s3.amazonaws.com/tut7-model.pt
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT uvicorn main:app --host 0.0.0.0 --port 8080
