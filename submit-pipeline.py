import os

from google.cloud import aiplatform
import google.auth

PROJECT_ID = os.getenv("PROJECT_ID")
if not PROJECT_ID:
    creds, PROJECT_ID = google.auth.default()

REGION = os.environ["REGION"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
PIPELINE_NAME = os.environ["PIPELINE_NAME"]
ENABLE_CACHING = os.getenv("CACHE_PIPELINE", 'true').lower() in ('true', '1', 't')

aiplatform.init(project=PROJECT_ID, location=REGION)
sync_pipeline = os.getenv("SUBMIT_PIPELINE_SYNC", 'False').lower() in ('true', '1', 't')

job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path="pipeline.yaml",
        location=REGION,
        pipeline_root = "gs://{}/pipeline_root/pipeline".format(BUCKET_NAME),
        enable_caching=True,
    )

print(f"Submitting pipeline {PIPELINE_NAME} in experiment {EXPERIMENT_NAME}.")

job.submit(experiment=EXPERIMENT_NAME)

if sync_pipeline:
    job.wait()
