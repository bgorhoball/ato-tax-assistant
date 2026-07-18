variable "project_name" {
  description = "Name prefix for all resources. Reused as the ECR repo / cluster / service name."
  type        = string
  default     = "ato-rag"
}

variable "aws_region" {
  description = "us-east-1: all prices in ARCHITECTURE.md were verified for this region."
  type        = string
  default     = "us-east-1"
}

variable "github_repo" {
  description = "GitHub repo allowed to deploy via OIDC, e.g. \"brian/ato-tax-assistant\"."
  type        = string
}

variable "app_hostname" {
  description = "Public hostname served via Cloudflare, e.g. \"tax.example.com\"."
  type        = string
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID that owns app_hostname (not a secret; the API token is)."
  type        = string
}

variable "alert_email" {
  description = "Email for the monthly budget alarm."
  type        = string
}

variable "container_port" {
  description = "Must stay on Cloudflare's proxied HTTP port list (80/8080/8880/...). 8501 is NOT proxied."
  type        = number
  default     = 80
}

variable "task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU, the priced config)."
  type        = number
  default     = 256
}

variable "task_memory" {
  description = "Fargate task memory in MiB. Bump to 2048 when the PII/NER layer lands."
  type        = number
  default     = 1024
}

variable "log_retention_days" {
  type    = number
  default = 14
}
