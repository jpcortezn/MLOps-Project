#!/bin/bash

EC2_USER=ubuntu
EC2_HOST=3.149.65.240
PEM_PATH=./id_rsa

echo "Conectando a $EC2_HOST y desplegando..."
scp -i $PEM_PATH -r ./ $EC2_USER@$EC2_HOST:/home/$EC2_USER/emotion-api

ssh -i $PEM_PATH $EC2_USER@$EC2_HOST << 'EOF'
cd emotion-api
docker-compose down
docker-compose up -d --build
EOF
