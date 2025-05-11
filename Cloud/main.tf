provider "aws" {
  region = "us-east-2"
}

# ----------------------------
# Clave SSH
# ----------------------------
resource "tls_private_key" "deployer" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "local_file" "private_key" {
  filename        = "id_rsa"
  content         = tls_private_key.deployer.private_key_pem
  file_permission = "0600"
}

resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = tls_private_key.deployer.public_key_openssh
}

# ----------------------------
# Red VPC y Subred Pública
# ----------------------------
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# ----------------------------
# Security Group
# ----------------------------
resource "aws_security_group" "pods_security_group" {
  name_prefix = "pods-security-group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ----------------------------
# EC2 Instancias con EIP
# ----------------------------
resource "aws_instance" "slave" {
  count             = 4
  ami               = "ami-0100e595e1cc1ff7f"
  instance_type     = "t2.micro"
  key_name          = aws_key_pair.deployer.key_name
  subnet_id         = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.pods_security_group.id]

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave_micro_${count.index + 1}"
  }
}

resource "aws_eip" "slave_eip" {
  count      = 4
  instance   = aws_instance.slave[count.index].id
  domain = "vpc"

}

resource "aws_instance" "slave2" {
  count             = 2
  ami               = "ami-060a84cbcb5c14844"
  instance_type     = "t2.medium"
  key_name          = aws_key_pair.deployer.key_name
  subnet_id         = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.pods_security_group.id]

  root_block_device {
    volume_size = 32
  }

  tags = {
    Name = "slave_large_${count.index + 1}"
  }
}

resource "aws_eip" "slave2_eip" {
  count      = 2
  instance   = aws_instance.slave2[count.index].id
  domain = "vpc"
}

resource "aws_instance" "master" {
  ami               = "ami-0100e595e1cc1ff7f"
  instance_type     = "t2.micro"
  key_name          = aws_key_pair.deployer.key_name
  subnet_id         = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.pods_security_group.id]

  tags = {
    Name = "master"
  }
}

resource "aws_eip" "master_eip" {
  instance = aws_instance.master.id
  domain = "vpc"
}

# ----------------------------
# S3 Bucket
# ----------------------------
resource "random_id" "bucket_id" {
  byte_length = 4
}

resource "aws_s3_bucket" "data_storage" {
  bucket        = "facial-recognition-data-${random_id.bucket_id.hex}"
  force_destroy = true
}

# ----------------------------
# IAM para Lambda
# ----------------------------
resource "aws_iam_role" "lambda_exec" {
  name = "lambda-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attach" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

# ----------------------------
# Lambda (sin archivo zip, usando inline code temporal)
# ----------------------------
resource "aws_lambda_function" "handler" {
  function_name = "ml-inference"
  role          = aws_iam_role.lambda_exec.arn
  runtime       = "python3.12"
  handler       = "index.lambda_handler"

  filename         = "lambda_placeholder.zip"
  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  # Este archivo debes crearlo localmente para evitar error de ejecución
}

# ----------------------------
# API Gateway -> Lambda
# ----------------------------
resource "aws_apigatewayv2_api" "http_api" {
  name          = "ml-http-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.http_api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.handler.invoke_arn
}

resource "aws_apigatewayv2_route" "default_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default_stage" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
}

# ----------------------------
# Outputs
# ----------------------------
output "master_ip" {
  value = aws_eip.master_eip.public_ip
}

output "slave_ips" {
  value = [for eip in aws_eip.slave_eip : eip.public_ip]
}

output "slave_ips2" {
  value = [for eip in aws_eip.slave2_eip : eip.public_ip]
}

output "s3_bucket_name" {
  value = aws_s3_bucket.data_storage.bucket
}

output "lambda_function_name" {
  value = aws_lambda_function.handler.function_name
}

output "api_endpoint" {
  value = aws_apigatewayv2_api.http_api.api_endpoint
}


///////////////////////////     Usuarios IAM          ///////////////////////


# Grupo con permisos específicos para los servicios requeridos
resource "aws_iam_group" "mlops_team" {
  name = "mlops-team"
}

# Política personalizada con acceso total a S3, API Gateway, Lambda y EC2
resource "aws_iam_policy" "mlops_services_policy" {
  name = "mlops-services-full-access"
  description = "Acceso total a S3, API Gateway, Lambda y EC2"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          # S3 Full Access
          "s3:*",
          # API Gateway Full Access
          "apigateway:*",
          # Lambda Full Access
          "lambda:*",
          # EC2 Full Access
          "ec2:*",
          # IAM para Lambda (necesario para crear/asignar roles)
          "iam:PassRole",
          "iam:GetRole",
          "iam:CreateRole",
          "iam:PutRolePolicy",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:DeleteRole",
          "iam:DeleteRolePolicy",
          # VPC para EC2 (necesario para crear/gestionar recursos de red)
          "vpc:*",
          # EC2 Instance Connect (necesario para conectarse desde la consola)
          "ec2-instance-connect:SendSSHPublicKey"
        ]
        Resource = "*"
      }
    ]
  })
}

# Adjuntar la política personalizada al grupo
resource "aws_iam_group_policy_attachment" "mlops_team_services_access" {
  group      = aws_iam_group.mlops_team.name
  policy_arn = aws_iam_policy.mlops_services_policy.arn
}

# Usuario IAM
resource "aws_iam_user" "mlops_user" {
  name = "mlops_user"
  force_destroy = true
}

# Añadir el usuario al grupo
resource "aws_iam_user_group_membership" "mlops_user_membership" {
  user = aws_iam_user.mlops_user.name
  groups = [aws_iam_group.mlops_team.name]
}

# NOTA: El login profile se creará pero sin contraseña definida
# Deberás configurar la contraseña manualmente desde la consola AWS
resource "aws_iam_user_login_profile" "mlops_user_login" {
  user     = aws_iam_user.mlops_user.name
  # Esta línea obliga al usuario a cambiar la contraseña en el primer inicio
  password_reset_required = true
}

# Credenciales de acceso programático
resource "aws_iam_access_key" "mlops_user_key" {
  user = aws_iam_user.mlops_user.name
}

# Outputs para ver los datos generados
output "mlops_console_user" {
  value = aws_iam_user.mlops_user.name
}

output "mlops_access_key_id" {
  value     = aws_iam_access_key.mlops_user_key.id
  sensitive = true
}

output "mlops_secret_access_key" {
  value     = aws_iam_access_key.mlops_user_key.secret
  sensitive = true
}

# Instrucciones posteriores a la ejecución
output "post_deployment_instructions" {
  value = <<EOF
IMPORTANTE: Después de ejecutar terraform apply, debes:

1. Ir a la consola AWS (IAM)
2. Buscar el usuario: ${aws_iam_user.mlops_user.name}
3. En la pestaña "Security credentials"
4. Hacer clic en "Create console password"
5. Establecer la contraseña: pozole64@
6. Desmarcar "Require password reset" si no quieres que cambie la contraseña

Luego podrá iniciar sesión con:
Usuario: ${aws_iam_user.mlops_user.name}
Contraseña: la que configures manualmente
EOF
}


# ----------------------------
# Security Engineer (Marco)
# ----------------------------
# Grupo de seguridad para el Security Engineer
resource "aws_iam_group" "security_engineers" {
  name = "security-engineers"
}

# Política personalizada para el Security Engineer
resource "aws_iam_policy" "security_engineer_policy" {
  name        = "security-engineer-policy"
  description = "Permisos para el Security Engineer"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # IAM Management
      {
        Effect = "Allow"
        Action = [
          "iam:Get*", 
          "iam:List*",
          "iam:Create*",
          "iam:Update*",
          "iam:Delete*",
          "iam:Attach*",
          "iam:Detach*",
          "iam:Put*",
          "iam:Generate*",
          "iam:Tag*",
          "iam:Untag*",
          "iam:SimulatePrincipalPolicy",
          "iam:SimulateCustomPolicy"
        ]
        Resource = "*"
      },
      # KMS for encryption
      {
        Effect = "Allow"
        Action = [
          "kms:Create*",
          "kms:Describe*",
          "kms:Enable*",
          "kms:List*",
          "kms:Put*",
          "kms:Update*",
          "kms:Revoke*",
          "kms:Disable*",
          "kms:Get*",
          "kms:Delete*",
          "kms:ScheduleKeyDeletion",
          "kms:CancelKeyDeletion",
          "kms:GenerateDataKey",
          "kms:TagResource",
          "kms:UntagResource"
        ]
        Resource = "*"
      },
      # Security audit permissions
      {
        Effect = "Allow"
        Action = [
          "cloudtrail:Describe*",
          "cloudtrail:Get*",
          "cloudtrail:List*",
          "cloudtrail:LookupEvents",
          "cloudtrail:StartLogging",
          "cloudtrail:StopLogging",
          "cloudtrail:CreateTrail",
          "cloudtrail:UpdateTrail",
          "cloudtrail:DeleteTrail",
          "cloudtrail:AddTags",
          "cloudtrail:RemoveTags",
          "cloudtrail:PutEventSelectors"
        ]
        Resource = "*"
      },
      # Config for compliance monitoring
      {
        Effect = "Allow"
        Action = [
          "config:*"
        ]
        Resource = "*"
      },
      # CloudWatch Logs for audit logs
      {
        Effect = "Allow"
        Action = [
          "logs:Describe*",
          "logs:Get*",
          "logs:List*",
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:TestMetricFilter",
          "logs:FilterLogEvents"
        ]
        Resource = "*"
      },
      # Read access to services being secured
      {
        Effect = "Allow"
        Action = [
          "s3:Get*",
          "s3:List*",
          "lambda:Get*", 
          "lambda:List*",
          "apigateway:GET",
          "ec2:Describe*"
        ]
        Resource = "*"
      },
      # Security Hub
      {
        Effect = "Allow"
        Action = [
          "securityhub:*"
        ]
        Resource = "*"
      },
      # GuardDuty
      {
        Effect = "Allow"
        Action = [
          "guardduty:*"
        ]
        Resource = "*"
      },
      # AWS WAF
      {
        Effect = "Allow"
        Action = [
          "waf:*",
          "waf-regional:*",
          "wafv2:*"
        ]
        Resource = "*"
      }
    ]
  })
}

# Adjuntar la política al grupo
resource "aws_iam_group_policy_attachment" "security_engineer_policy_attach" {
  group      = aws_iam_group.security_engineers.name
  policy_arn = aws_iam_policy.security_engineer_policy.arn
}

# Usuario IAM para Marco (Security Engineer)
resource "aws_iam_user" "SE_Marco" {
  name = "SE_Marco"
  force_destroy = true
  
  tags = {
    Role = "Security Engineer"
    Description = "Usuario para Marco - Encargado de seguridad y controles de acceso"
  }
}

# Añadir el usuario al grupo
resource "aws_iam_user_group_membership" "marco_security_membership" {
  user = aws_iam_user.SE_Marco.name
  groups = [aws_iam_group.security_engineers.name]
}

# Perfil de acceso a la consola
resource "aws_iam_user_login_profile" "SE_Marco_login" {
  user                    = aws_iam_user.SE_Marco.name
  password_reset_required = true
}

# Credenciales de acceso programático
resource "aws_iam_access_key" "SE_Marco_key" {
  user = aws_iam_user.SE_Marco.name
}

# Outputs para SE_Marco
output "SE_Marco_console_user" {
  value = aws_iam_user.SE_Marco.name
}

output "SE_Marco_access_key_id" {
  value     = aws_iam_access_key.SE_Marco_key.id
  sensitive = true
}

output "SE_Marco_secret_access_key" {
  value     = aws_iam_access_key.SE_Marco_key.secret
  sensitive = true
}

output "SE_Marco_console_instructions" {
  value = <<EOF
INSTRUCCIONES PARA SECURITY ENGINEER (MARCO):

1. Ir a la consola AWS: https://console.aws.amazon.com
2. Iniciar sesión con:
   - Usuario: ${aws_iam_user.SE_Marco.name}
   - Contraseña: [La contraseña temporal que se te enviará por correo electrónico]
3. La primera vez que accedas, se te pedirá que cambies la contraseña

RESPONSABILIDADES:
- Implementar políticas de seguridad y controles de acceso usando AWS IAM
- Asegurar la encriptación de datos en reposo y en tránsito usando AWS KMS
- Auditar el acceso a la infraestructura y servicios
- Asegurar el acceso a las APIs usando mecanismos de autenticación
EOF
}