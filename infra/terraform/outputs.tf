output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.app.name
}

output "container_name" {
  description = "Container name inside the task definition (needed by the CI render step)."
  value       = var.project_name
}

output "github_deploy_role_arn" {
  description = "Set this as the AWS_ROLE_ARN secret in the GitHub repo."
  value       = aws_iam_role.github_deploy.arn
}

output "google_api_key_param" {
  description = "SSM parameter to fill with the real Gemini API key."
  value       = aws_ssm_parameter.google_api_key.name
}

output "cloudflare_token_param" {
  description = "SSM parameter to fill with the Cloudflare API token (Zone.DNS edit)."
  value       = aws_ssm_parameter.cloudflare_api_token.name
}

output "app_url" {
  value = "https://${var.app_hostname}"
}
