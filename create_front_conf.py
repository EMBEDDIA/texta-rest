#! /usr/bin/env python3
import json
import os


def parse_bool_env(env_name: str, default: bool):
    value = os.getenv(env_name, str(default)).lower()
    if value in ["true"]:
        return True
    else:
        return False


# Parse env variables
TEXTA_API_URL = os.getenv("TEXTA_API_URL", "http://localhost")
TEXTA_HOSTED_FILE_FIELD = os.getenv("TEXTA_HOSTED_FILE_FIELD", "properties.hosted_filepath")
TEXTA_USE_UAA = parse_bool_env("TEXTA_USE_UAA", False)
TEXTA_UAA_URL = os.getenv("TEXTA_UAA_URL", "http://localhost:8080/uaa")
TEXTA_UAA_REDIRECT_URI = os.getenv("TEXTA_UAA_REDIRECT_URI", "http://localhost/api/v1/uaa/callback")
TEXTA_UAA_CLIENT_ID = os.getenv("TEXTA_UAA_CLIENT_ID", "login")

# Generate config
config = {
  "apiHost": TEXTA_API_URL,
  "apiBasePath": "/api/v1",
  "apiBasePath2": "/api/v2",
  "logging": True,
  "fileFieldReplace": TEXTA_HOSTED_FILE_FIELD,
  "useCloudFoundryUAA": TEXTA_USE_UAA,
  "uaaConf":{
    "uaaURL": f"{TEXTA_UAA_URL}/oauth/authorize",
    "redirect_uri": TEXTA_UAA_REDIRECT_URI,
    "client_id": TEXTA_UAA_CLIENT_ID,
    "scope":"openid",
    "response_type":"code"
  }
}

# print output
print(json.dumps(config))
