FROM python:3.7.13-slim-buster AS base

# create virtual environment
ENV VIRTUAL_ENV=/opt/benchmark_env
RUN python3 -m venv /opt/benchmark_env
RUN /opt/benchmark_env/bin/python3 -m pip install --upgrade pip
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#set workingdir
WORKDIR /benchmark/

RUN apt update
# install necessary dependencies for python packages
RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
# get env packages & install them
COPY ./src/conda_env.txt /benchmark/requirements.txt 
RUN pip install -r requirements.txt


#### DATASET SECTION ####
# COPY ./Mechanical-datasets/ /benchmark/Mechanical-datasets/
# COPY ./MFPT_Fault_Data_Sets/ /benchmark/MFPT_Fault_Data_Sets/
# COPY ./CWRU/ /benchmark/CWRU
# COPY ./Paderborn/ /benchmark/Paderborn/
# COPY ./XJTU-SY_Bearing_Datasets /benchmark/XJTU-SY_Bearing_Datasets/
# COPY ./dataset_paderborn/ /benchmark/dataset_paderborn
#### DATASET SECTION END####

COPY  ./src/ /benchmark/
