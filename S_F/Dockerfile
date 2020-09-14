



# Waitress
## getting base image from python 3.6.8
#FROM python:3.6.8
#
#MAINTAINER rae wallace <raedjamw@gmail.com>
#
## This prevents Python from writing out pyc files
#ENV PYTHONDONTWRITEBYTECODE 1
## This keeps Python from buffering stdin/stdout
#ENV PYTHONUNBUFFERED 1
#
### copy project
#COPY . .
#
## install system dependencies
##RUN yum update \
##    && yum -y install gcc make \
##    && rm -rf /var/lib/yum/lists/*
#RUN apt-get update \
#    && apt-get -y install gcc make \
#    && rm -rf /var/lib/apt/lists/*
#
## install dependencies
#RUN pip3 install --no-cache-dir --upgrade pip
#RUN pip3 install  pandas numpy sklearn statsmodels
#RUN pip3 install waitress==1.44
#RUN pip3 install Flask==1.1.2
#
## set work directory
#WORKDIR /src/app
#
#COPY . .
#
## functions
#WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
#RUN python dataframe_shape_functions.py
#
## Unit Test
#WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
#RUN python test_Dataframes.py
#
## Functions
#WORKDIR /src/app/ML_Model/Test_Transformations_pickles
#RUN python transformation_functions.py
#
## Unit Test
#WORKDIR /src/app/ML_Model/Test_Transformations_pickles
#RUN python test_Transformations.py
#
## Generate pikle file
#WORKDIR /src/app/ML_Model
#RUN python glm_model_26_c.py
#
## set work directory
#WORKDIR /src/app
## set app port
#
#EXPOSE 8080
#
#CMD exec waitress serve --listen "*:8080" "app:app"













# raedjamw/sf_glm:26

## getting base image from python 3.6.8
#FROM centos/python-36-centos7:latest
#
##set the root user
#USER root
#
#MAINTAINER rae wallace <raedjamw@gmail.com>
#
## This prevents Python from writing out pyc files
#ENV PYTHONDONTWRITEBYTECODE 1
## This keeps Python from buffering stdin/stdout
#ENV PYTHONUNBUFFERED 1
#
## copy project
#COPY . .
#
## install dockerfile dependencies
#RUN yum -y update
#RUN pip install --no-cache-dir --upgrade pip
#RUN yum install python36 python36-pip
#
#
## set work directory
#WORKDIR /src/app
#
## install project requirements
#RUN pip install --no-cache-dir -Iv gunicorn pandas numpy sklearn statsmodels
#RUN pip install --no-cache-dir Flask==1.1.2
#
##Copy project
#COPY . .
#
## functions
#WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
#RUN python dataframe_shape_functions.py
#
## Unit Test
#WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
#RUN python test_Dataframes.py
#
## Functions
#WORKDIR /src/app/ML_Model/Test_Transformations_pickles
#RUN python transformation_functions.py
#
## Unit Test
#WORKDIR /src/app/ML_Model/Test_Transformations_pickles
#RUN python test_Transformations.py
#
## Generate pikle file
#WORKDIR /src/app/ML_Model
#RUN python glm_model_26_c.py
#
## set work directory
#WORKDIR /src/app
## set app port
#
#EXPOSE 8080
#
## Run app.py when the container launches
#CMD [ "gunicorn", "--bind", "0.0.0.0:8080", "app:app" ]



# raedjamw/sf_glm:22
# getting base image from python 3.6.8
FROM python:3.6.8

MAINTAINER rae wallace <raedjamw@gmail.com>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

## copy project
COPY . .

 #install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/app

# install project requirements
RUN pip install --no-cache-dir -Iv gunicorn pandas numpy sklearn statsmodels
RUN pip install --no-cache-dir Flask==1.1.2

#Copy project
COPY . .

# functions
WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
RUN python dataframe_shape_functions.py

# Unit Test
WORKDIR /src/app/ML_Model/Test_Dataframes_pickles
RUN python test_Dataframes.py

# Functions
WORKDIR /src/app/ML_Model/Test_Transformations_pickles
RUN python transformation_functions.py

# Unit Test
WORKDIR /src/app/ML_Model/Test_Transformations_pickles
RUN python test_Transformations.py

# Generate pikle file
WORKDIR /src/app/ML_Model
RUN python glm_model_26_c.py

# set work directory
WORKDIR /src/app
# set app port

EXPOSE 8080

# Run app.py when the container launches
CMD [ "gunicorn", "--bind", "0.0.0.0:8080", "app:app" ]




