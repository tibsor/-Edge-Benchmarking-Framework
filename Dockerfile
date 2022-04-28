FROM python:3.7.13-slim-buster AS base

ENV MAIN_APP=main.py
# create virtual environment
ENV VIRTUAL_ENV=/opt/inference_env
RUN python3 -m venv /opt/inference_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# "apt" has unstable CLI, so we use apt-get instead
RUN apt-get update
# install necessary dependencies for python packages
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

WORKDIR /inference/
# get env packages & install them
COPY ./src/conda_env.txt /inference/requirements.txt 
RUN pip install -r requirements.txt

# to avoid caching problems for code while building, we will force remove anything left in the inference folder before copying the source code
RUN rm -rf /inference/
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
