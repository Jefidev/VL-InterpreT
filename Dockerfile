FROM python:3.9-slim

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

RUN mkdir /vl-interpret
WORKDIR /vl-interpret

RUN pip install dash_bootstrap_components

# CMD python run_app.py -p 6006 -d example_database6 -m ApiMode
CMD python app.py