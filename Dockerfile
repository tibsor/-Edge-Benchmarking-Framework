FROM python:3.7.13-slim-buster AS base

# create virtual environment
ENV VIRTUAL_ENV=/opt/benchmark_env
RUN python3 -m venv /opt/benchmark_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /benchmark/
COPY ./Mechanical-datasets/ /benchmark/Mechanical-datasets/
COPY ./MFPT_Fault_Data_Sets/ /benchmark/MFPT_Fault_Data_Sets/
# "apt" has unstable CLI, so we use apt-get instead
RUN apt update
# install necessary dependencies for python packages
RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6


# get env packages & install them
COPY ./src/conda_env.txt /benchmark/requirements.txt 
RUN pip install -r requirements.txt

# to avoid caching problems for code while building, we will force remove anything left in the inference folder before copying the source code
#RUN rm -rf /inference/
COPY  ./src/ /benchmark/
