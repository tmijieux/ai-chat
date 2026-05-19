import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OllamaModelResolver:
    """
    Resolves an Ollama model name to its underlying GGUF blob file.
    Works with local Ollama installations.
    """

    def __init__(self, ollama_dir: str|None = None):
        self.ollama_dir = Path(
            ollama_dir or Path.home() / ".ollama" / "models"
        )

        self.manifests_dir = self.ollama_dir / "manifests"
        self.blobs_dir = self.ollama_dir / "blobs"

    # -----------------------------
    # Step 1: find manifest file
    # -----------------------------
    def find_manifest(self, model_name: str):
        """
        Model name format usually:
        llama3, mistral, phi3, etc.
        """

        # Ollama stores manifests under registry path
        registry_path = self.manifests_dir
        print("registry_path=",registry_path)
        print("model_name=",model_name)

        if ":" in model_name:
            split = model_name.rsplit(":",maxsplit=1)

            model_name = ":".join(split[:-1])
            tag = split[-1]
        else: 
            tag = None

        for path in registry_path.rglob("*"):
            if path.is_file() and model_name in str(path):
                logger.critical("found manifest for %s at %s", model_name, path)
                return path

        raise FileNotFoundError(f"Manifest not found for model: {model_name}")

    # -----------------------------
    # Step 2: parse manifest JSON
    # -----------------------------
    def load_manifest(self, manifest_path: Path):
        with open(manifest_path, "r") as f:
            return json.load(f)

    # -----------------------------
    # Step 3: extract blob digest
    # -----------------------------
    def extract_blobs(self, manifest_json):
        """
        Manifest contains layers with digests like:
        sha256:abcd1234...
        """
        blobs = []
        for layer in manifest_json.get("layers", []):
            if layer["mediaType"] != "application/vnd.ollama.image.model":
                continue
            digest = layer.get("digest")
            if digest and digest.startswith("sha256:"):
                blobs.append(digest.replace("sha256:", "sha256-"))

        return blobs

    # -----------------------------
    # Step 4: resolve blob file path
    # -----------------------------
    def resolve_blob_path(self, blob_hash: str):
        blob_path = self.blobs_dir / blob_hash

        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_hash}")

        return blob_path

    # -----------------------------
    # Main entry
    # -----------------------------
    def resolve_model(self, model_name: str):
        manifest_path = self.find_manifest(model_name)
        manifest = self.load_manifest(manifest_path)

        blobs = self.extract_blobs(manifest)

        blob_paths = [self.resolve_blob_path(b) for b in blobs]

        return {
            "model": model_name,
            "manifest": str(manifest_path),
            "blobs": [str(p) for p in blob_paths],
        }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    resolver = OllamaModelResolver()

    result = resolver.resolve_model("llama3")

    print(json.dumps(result, indent=2))