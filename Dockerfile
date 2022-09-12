FROM python:3.10-slim

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
COPY . ./
CMD python run_app.py -p 6006 -d example_database2