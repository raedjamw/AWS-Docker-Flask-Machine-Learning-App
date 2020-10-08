#!/bin/bash

# $0 is the full path of the running script
scriptdir="$(dirname "$0")"
echo '0 is the full path of the running script'
cd "$scriptdir"
# ssh to the EC2 instance
ssh -i ./sf_glm.pem ec2-user@ec2-3-23-219-43.us-east-2.compute.amazonaws.com 'bash -i'  <<-'ENDSSH'
    # Pull the image
    sudo docker pull raedjamw/sf_glm_final:1.0
    # Run the image in the container
    sudo docker run --name Deploy_SF_Final -p 8080:8080 raedjamw/sf_glm_final:1.0


ENDSSH
