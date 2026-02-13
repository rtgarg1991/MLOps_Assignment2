import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Mock google cloud modules for unit tests.
google = types.ModuleType("google")
google_cloud = types.ModuleType("google.cloud")
google_cloud_storage = types.ModuleType("google.cloud.storage")


class DummyBlob:
    def __init__(self, name=""):
        self.name = name

    def exists(self):
        return False

    def upload_from_filename(self, *args, **kwargs):
        return None

    def download_to_filename(self, *args, **kwargs):
        return None


class DummyBucket:
    def blob(self, name):
        return DummyBlob(name)

    def copy_blob(self, *args, **kwargs):
        return None


class DummyStorageClient:
    def __init__(self, *args, **kwargs):
        self.project = "dummy-project"

    def bucket(self, *args, **kwargs):
        return DummyBucket()

    def list_blobs(self, *args, **kwargs):
        return []


google_cloud_storage.Client = DummyStorageClient

sys.modules["google"] = google
sys.modules["google.cloud"] = google_cloud
sys.modules["google.cloud.storage"] = google_cloud_storage
