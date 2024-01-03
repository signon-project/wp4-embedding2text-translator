#!/bin/bash

dokerComposeNetwork="signon-framework-docker-compose_default"
dockerContainerName=embedding2text-translation

docker run --rm -it \
--name $dockerContainerName \
--network=$dokerComposeNetwork \
 -v $(pwd):/$dockerContainerName \
python:3.8 \
bash -c "cd $dockerContainerName && pip install -r requirements.txt && bash"
