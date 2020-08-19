# Docker Overview

## What is Docker?

Docker is an open platform for developers and sysadmins to *Build, Run, and Share* applications independently with the help of Containers.

## Docker Terminologies
<img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/11/1.png" width="30%" height="40%" alt="dockerimg"><br>

* When the Dockerfile is built, it becomes a Docker Image and when we run the Docker Image then it finally becomes a Docker Container.

#### Docker Images:
-It contains code, runtime, libraries, environment variables and configuratuon files which is needed to run an application in a container.<br>

#### Docker Container:
-Containers are instances of Docker images and each and every application runs on separate containers and has its own set of dependencies & libraries. 

<img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/11/Picture1-2-333x300.png" width="220px" height="220px">

#### Docker Hub:
-A registry service on the cloud that allows you to download and upload Docker images that are built.

## Basic Docker Commands
| Command | Description | Usage|
| ------ | ------ | ------ |
| docker ps| allows  to view all the containers that are running on the Docker Host. | `docker ps`  |
|docker start| starts any stopped container(s) |` docker start <CONTAINER ID/NAME> `  |
|docker stop| stops any running container(s).| `docker stop <CONTAINER ID/NAME>` |
|docker run| creates container from docker image| `docker run -it -d <image name> /bin/bash `  |
|docker rm| deletes the stopped container| `docker rm <CONTAINER ID/NAME>`  |
|docker build|build an image from a docker file|`<path to docker file> docker build .`|
| docker pull | To download a particular image |`docker pull <image name>` |
|docker exec | Access the running container | `docker exec -it <container id> /bin/bash` |
|docker commit|creates a new image of an edited container on the local system|`docker commit <conatainer id> <username/imagename>`|
|docker push|push an image to the docker hub repository|`docker push <username/image name>`|

## Docker Architecture
<img src="extras/dockerimg1.jpg" width="50%">




