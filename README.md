# SignON Embedding-to-Text Machine Translation Component

This is the repository containing the code of the SignON embedding-to-text machine translation component.
This component is a web service built with Flask that receives a embedding representation from the SignON SLR componet (https://github.com/signon-project/wp3-slr-component) and machine translates it into an target language.

## Installation

This component is built to run in a Docker container (see `Dockerfile`).

## Testing

To test the component within the pipeline, you need to have the docker container running. For that, there are two options:

- Using the run_dockerized.sh script to create an interactive bash interpreter and run the server there:

```bash
./run_dockerized.sh
cd src; python slt_server.py ## To be executed inside the container
```

- Build and run the container in detach mode:
  
```bash
docker build -t embedding2text-translation .
docker run -v ${PWD}/model:/model --name embedding2text-translation -p 5001:5001 -d embedding2text-translation
```

# LICENSE

This code is licensed under the Apache License, Version 2.0 (LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
