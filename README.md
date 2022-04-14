# Edge Benchmarking Framework

## Getting started

This repository is under continuous development, currently being able to run a MLP model inference with around 64MB RAM. 

## First time use

`docker build -t *img_name*:*tag* .`

Follwed by: `bash seq_memory_limit.sh`


## Description
At the moment, given the image built from the present Dockerfile, the container will run 10 seeded random observations from the SEU evaluation dataset with an MLP model. By running the shell script after the image is built, it will recursively run the container, gradually decreasing memory size  



## Roadmap
* check function peak memory usage
* add other model types and compare memory usage results
* add other datasets


## License
For open source projects, say how it is licensed.

## Project status
Active. Following steps TBD
