# Default VPC + its (public) subnets — no NAT gateways, no paid networking.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Cloudflare publishes its egress ranges; only Cloudflare may reach the task,
# so the origin can't be hit directly even though it has a public IP.
data "http" "cloudflare_ips_v4" {
  url = "https://www.cloudflare.com/ips-v4"
}

locals {
  cloudflare_ipv4_cidrs = [
    for line in split("\n", trimspace(data.http.cloudflare_ips_v4.response_body)) :
    trimspace(line) if trimspace(line) != ""
  ]
}

resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-app-"
  description = "Streamlit task: HTTP from Cloudflare only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP from Cloudflare proxy ranges"
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = local.cloudflare_ipv4_cidrs
  }

  egress {
    description = "All outbound (Gemini API, ECR pull, SSM, logs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}
