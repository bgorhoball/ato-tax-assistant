resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE" # CI pushes :latest alongside :<sha>

  image_scanning_configuration {
    scan_on_push = true
  }

  force_delete = true # side project: allow terraform destroy to remove images
}

# Keep image storage pennies-small: only the 5 most recent images.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}
