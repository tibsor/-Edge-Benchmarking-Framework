FROM python:3.7.13-slim-buster AS base

WORKDIR /inference/

RUN apt update
RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY ./src/conda_env.txt /inference/requirements.txt 
RUN pip install -r requirements.txt
ENV MAIN_APP=main.py
RUN pip install filprofiler
COPY  ./src/ /inference/

# debugger

# FROM base as debug

# RUN pip install ptvsd
# CMD python3 -m ptvsd --host 0.0.0.0 --port 5678 --wait --multiprocess main.py
# #ENTRYPOINT ["python3", "-m", "ptvsd", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m"]

# primary
FROM base as primary

CMD [ "python3", "main.py" ]
#RUN python3 main.py
