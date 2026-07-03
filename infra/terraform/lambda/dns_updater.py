"""Upsert the Cloudflare A record when the ECS task gets a new public IP.

Fargate Spot replaces the task on interruption/deploy and the new task gets a
new public IP (no ALB in this stack, by budget). EventBridge fires this on
every task reaching RUNNING; we look up the task ENI's public IP and point the
proxied Cloudflare record at it.

Env vars: CF_ZONE_ID, CF_RECORD_NAME, CF_TOKEN_PARAM (SSM SecureString name).
"""

import json
import logging
import os
import urllib.request

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CF_API = "https://api.cloudflare.com/client/v4"

_token_cache = None


def _cf_token():
    global _token_cache
    if _token_cache is None:
        ssm = boto3.client("ssm")
        _token_cache = ssm.get_parameter(
            Name=os.environ["CF_TOKEN_PARAM"], WithDecryption=True
        )["Parameter"]["Value"]
    return _token_cache


def _cf_request(method, path, body=None):
    req = urllib.request.Request(
        f"{CF_API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {_cf_token()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read())
    if not payload.get("success"):
        raise RuntimeError(f"Cloudflare API error: {payload.get('errors')}")
    return payload["result"]


def _task_public_ip(detail):
    eni_id = None
    for attachment in detail.get("attachments", []):
        for kv in attachment.get("details", []):
            if kv.get("name") == "networkInterfaceId":
                eni_id = kv["value"]
    if not eni_id:
        return None

    ec2 = boto3.client("ec2")
    enis = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    association = enis["NetworkInterfaces"][0].get("Association", {})
    return association.get("PublicIp")


def handler(event, _context):
    detail = event.get("detail", {})
    if detail.get("lastStatus") != "RUNNING":
        return {"skipped": "not RUNNING"}

    ip = _task_public_ip(detail)
    if not ip:
        logger.warning("No public IP on task ENI yet; event=%s", json.dumps(event))
        return {"skipped": "no public ip"}

    zone = os.environ["CF_ZONE_ID"]
    name = os.environ["CF_RECORD_NAME"]

    existing = _cf_request("GET", f"/zones/{zone}/dns_records?type=A&name={name}")
    record = {"type": "A", "name": name, "content": ip, "ttl": 1, "proxied": True}

    if existing:
        if existing[0]["content"] == ip:
            logger.info("Record %s already points at %s", name, ip)
            return {"unchanged": ip}
        _cf_request("PUT", f"/zones/{zone}/dns_records/{existing[0]['id']}", record)
        logger.info("Updated %s -> %s", name, ip)
    else:
        _cf_request("POST", f"/zones/{zone}/dns_records", record)
        logger.info("Created %s -> %s", name, ip)

    return {"updated": ip}
