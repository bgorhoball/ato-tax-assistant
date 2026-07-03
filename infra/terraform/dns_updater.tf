data "archive_file" "dns_updater" {
  type        = "zip"
  source_file = "${path.module}/lambda/dns_updater.py"
  output_path = "${path.module}/lambda/dns_updater.zip"
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "dns_updater" {
  name               = "${var.project_name}-dns-updater"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "dns_updater_logs" {
  role       = aws_iam_role.dns_updater.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "dns_updater" {
  name = "dns-updater"
  role = aws_iam_role.dns_updater.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ec2:DescribeNetworkInterfaces"]
        Resource = "*" # EC2 Describe* does not support resource-level scoping
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = [aws_ssm_parameter.cloudflare_api_token.arn]
      },
    ]
  })
}

resource "aws_lambda_function" "dns_updater" {
  function_name    = "${var.project_name}-dns-updater"
  role             = aws_iam_role.dns_updater.arn
  runtime          = "python3.12"
  handler          = "dns_updater.handler"
  filename         = data.archive_file.dns_updater.output_path
  source_code_hash = data.archive_file.dns_updater.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      CF_ZONE_ID     = var.cloudflare_zone_id
      CF_RECORD_NAME = var.app_hostname
      CF_TOKEN_PARAM = aws_ssm_parameter.cloudflare_api_token.name
    }
  }

  depends_on = [aws_cloudwatch_log_group.dns_updater]
}

# Fire on our service's tasks reaching RUNNING.
resource "aws_cloudwatch_event_rule" "task_running" {
  name = "${var.project_name}-task-running"

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    detail-type   = ["ECS Task State Change"]
    detail = {
      clusterArn    = [aws_ecs_cluster.main.arn]
      group         = ["service:${aws_ecs_service.app.name}"]
      lastStatus    = ["RUNNING"]
      desiredStatus = ["RUNNING"]
    }
  })
}

resource "aws_cloudwatch_event_target" "dns_updater" {
  rule = aws_cloudwatch_event_rule.task_running.name
  arn  = aws_lambda_function.dns_updater.arn
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.dns_updater.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.task_running.arn
}
