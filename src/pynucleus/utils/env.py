# Global env tweaks
import os
# Disable PostHog telemetry & HF advisory warnings
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("DO_NOT_TRACK", "1") 