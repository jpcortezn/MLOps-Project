#!/bin/bash

# Usuario correcto para Amazon Linux 2023 (ajusta si usas otra AMI)
EC2_USER=ec2-user
EC2_HOST=3.149.65.240
PEM_PATH=./id_rsa

echo "Conectando a $EC2_HOST y desplegando..."

# Copiar todo el proyecto a la instancia
scp -i $PEM_PATH -o StrictHostKeyChecking=no -r ./ $EC2_USER@$EC2_HOST:/home/$EC2_USER/emotion-api

# Conectarse y hacer limpieza + despliegue
ssh -i $PEM_PATH -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST << 'EOF'
echo "Limpiando contenedores y redes viejas..."
cd emotion-api

# Forzar liberación de puertos ocupados (8000, 9090, etc.)
sudo lsof -t -i:8000 -i:9090 -i:3000 | xargs --no-run-if-empty sudo kill -9 || true

# Bajar contenedores y limpiar
docker-compose down || true
docker container prune -f || true
docker network prune -f || true

# Desplegar servicios
echo "Levantando servicios con Docker Compose..."
docker-compose up -d --build
EOF
