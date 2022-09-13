FROM python:3.9-slim

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN mkdir /vl-interpret
WORKDIR /vl-interpret

CMD python run_app.py -p 6006 -d example_database6