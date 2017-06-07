gcloud ml-engine jobs submit training demo1 \
    --job-dir gs://learn_talk/demo1 \
    --runtime-version 1.0 \
    --module-name tfb.trytfb \
    --package-path tfb/ \
    --region us-central1 \

gcloud ml-engine versions create v1 \
    --model deml1 \
    --origin $MODEL_BINARIES \
    --runtime-version 1.0

