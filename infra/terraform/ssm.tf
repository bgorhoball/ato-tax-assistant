# Secrets live in SSM Parameter Store (free tier), injected into the task at
# start. Terraform creates placeholders only — set real values out-of-band:
#   aws ssm put-parameter --name /<project>/google-api-key --type SecureString \
#       --value '<real key>' --overwrite
# ignore_changes keeps terraform from ever writing the real value back to state.

resource "aws_ssm_parameter" "google_api_key" {
  name  = "/${var.project_name}/google-api-key"
  type  = "SecureString"
  value = "CHANGEME"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "cloudflare_api_token" {
  name  = "/${var.project_name}/cloudflare-api-token"
  type  = "SecureString"
  value = "CHANGEME"

  lifecycle {
    ignore_changes = [value]
  }
}
