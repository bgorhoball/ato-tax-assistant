# Deployment Runbook

Operational steps only. *Why* the stack looks like this → [ARCHITECTURE.md](../ARCHITECTURE.md);
budget/goals → [DEPLOY_BRIEF.md](../DEPLOY_BRIEF.md).

## Stack at a glance

- **App:** Streamlit container (port **80** — Cloudflare only proxies a fixed port list; 8501 is not on it), ChromaDB index baked into the image.
- **Compute:** ECS Fargate **Spot**, 0.25 vCPU / 1 GB, us-east-1, single task, public IP (no ALB).
- **Front door:** Cloudflare free tier (proxied A record, Flexible SSL). A Lambda (EventBridge, ECS task-state change) re-points the A record whenever the task gets a new IP.
- **Secrets:** SSM Parameter Store → injected via task definition. No AWS keys in GitHub (OIDC role).
- **Guardrail:** AWS Budget $15/mo, email at 80% actual / 100% forecast.
- **Expected cost:** ~$7–8/mo (see ARCHITECTURE.md cost table).

## First-time setup (in order)

1. **Tooling:** install Terraform ≥ 1.5; `aws configure` with your credentials.
2. **Variables:** `cd infra/terraform && cp terraform.tfvars.example terraform.tfvars`, fill in
   `github_repo`, `app_hostname`, `cloudflare_zone_id`, `alert_email`.
3. **Provision:** `terraform init && terraform apply`. Note the outputs.
4. **Real secrets** (Terraform only creates `CHANGEME` placeholders):
   ```bash
   aws ssm put-parameter --name /ato-rag/google-api-key       --type SecureString --value '<Gemini key>' --overwrite
   aws ssm put-parameter --name /ato-rag/cloudflare-api-token --type SecureString --value '<CF token>'   --overwrite
   ```
   The Cloudflare token needs only **Zone → DNS → Edit** on the one zone.
5. **Cloudflare dashboard:** set SSL/TLS mode to **Flexible** (Cloudflare→origin is HTTP:80).
   Don't create the A record — the Lambda does.
6. **GitHub:** push the repo (`chroma_db/` must be committed — the CI build bakes it in);
   add repo secret `AWS_ROLE_ARN` = terraform output `github_deploy_role_arn`.
7. **First deploy:** push to `master`. The ECS service will flap until the first image lands in
   ECR — expected; the workflow's deploy step stabilises it.
8. **Verify:** open `https://<app_hostname>`. If broken, check logs (below).

## Day-to-day

| Action | Command |
|---|---|
| Deploy app change | `git push origin master` (CI does build → ECR → rolling deploy → DNS follows) |
| Change infra | `cd infra/terraform && terraform apply` (CI ignores `infra/**` on purpose) |
| Rotate a secret | `aws ssm put-parameter ... --overwrite`, then force redeploy: `aws ecs update-service --cluster ato-rag --service ato-rag --force-new-deployment` |
| Re-ingest corpus | run ingestion locally → commit new `chroma_db/` → push (new image = new index) |
| Tear down | `terraform destroy` (ECR has `force_delete`; images go too) |

## Troubleshooting

- **App logs:** CloudWatch `/ecs/ato-rag`
- **DNS updater:** CloudWatch `/aws/lambda/ato-rag-dns-updater` — first place to look if the
  hostname points at a dead IP after a Spot interruption.
- **Task won't start:** usually the SSM param is still `CHANGEME`, or the image doesn't exist yet.
  `aws ecs describe-services --cluster ato-rag --services ato-rag` → events list says which.
- **522/timeout via Cloudflare:** task IP changed and Lambda didn't fire/failed, or SSL mode
  isn't Flexible. Direct-to-IP access is blocked by design (security group allows Cloudflare
  ranges only).

## Known limitations (accepted trade-offs)

- Single Spot task: ~1 min blip on interruption, chat history in open sessions is lost.
- Terraform state is local (`*.tfstate` gitignored). Losing it means re-import — acceptable for
  a one-person stack.
- `terraform validate` has not been run on this config yet (no terraform on the dev box);
  expect the first `terraform plan` to be the real syntax check.
- Cloudflare Flexible SSL = browser→Cloudflare encrypted, Cloudflare→origin plaintext HTTP.
  Fine for public ATO documents; upgrade path is ALB + ACM (see ARCHITECTURE.md).

## Reusing for the RegTech repo

Copy `Dockerfile`, `.dockerignore`, `infra/`, `.github/workflows/deploy.yml`; change
`project_name` in tfvars and the five `env:` values at the top of `deploy.yml`. Everything else
is name-agnostic.
