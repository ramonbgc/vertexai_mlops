import os
import google.auth

from kfp.registry import RegistryClient

PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    creds, PROJECT_ID = google.auth.default()
REGION = os.environ["REGION"]

REPO_NAME= "mlops-vivo-repo"

host_uri = "https://us-central1-kfp.pkg.dev/{}/{}".format(PROJECT_ID,REPO_NAME)
client = RegistryClient(host=host_uri)

client.upload_pipeline(file_name='pipeline.yaml', tags=['V1','latest'])

l = client.list_packages()
print(l)
