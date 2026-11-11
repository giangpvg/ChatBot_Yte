import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medical_chatbot.api.main import app

def test_api_health_and_index():
    print("Testing FastAPI Health & Index endpoints...")
    with TestClient(app) as client:
        # 1. Test Root Page (Web UI)
        res_index = client.get("/")
        assert res_index.status_code == 200, f"Expected 200, got {res_index.status_code}"
        assert res_index.headers["content-type"].startswith("text/html")
        assert "<title>MedAI" in res_index.text, "Web title should be present"
        assert '<div id="root"></div>' in res_index.text, "React root should be present"
        print("✅ Root Web UI loaded successfully!")

        # 2. Test Health Endpoint
        res_health = client.get("/api/v1/health")
        assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
        data = res_health.json()
        assert data["status"] == "healthy", f"Health status should be healthy, got {data}"
        assert data["vector_db_ready"] is True
        assert data["bm25_ready"] is True
        print(f"✅ Health Check passed: {data}")

if __name__ == "__main__":
    test_api_health_and_index()
