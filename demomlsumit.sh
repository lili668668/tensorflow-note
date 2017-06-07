gcloud ml-engine jobs submit training demo19 \
    --job-dir gs://learn_talk/demo17 \
    --runtime-version 1.0 \
    --module-name tfb.trytfb \
    --package-path tfb/ \
    --region us-central1 \

