resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "dns_updater" {
  name              = "/aws/lambda/${var.project_name}-dns-updater"
  retention_in_days = var.log_retention_days
}
