#!/bin/bash

# $0 is the full path of the running script
scriptdir="$(dirname "$0")"
echo '0 is the full path of the running script'
cd "$scriptdir"
# ssh to the EC2 instance
ssh -i ./ubuntu_3.pem ec2-user@ec2-3-23-219-43.us-east-2.compute.amazonaws.com 'bash -i'  <<-'ENDSSH'
    # Pull the image
    sudo docker pull raedjamw/sf_glm:46.0
    # Run the image in the container
    sudo docker run --name SF_MLE -p 8080:8080 raedjamw/sf_glm:46.0


ENDSSH


#    chmod +x api.sh
