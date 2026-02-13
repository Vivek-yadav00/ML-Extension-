import requests
import time

def test_backend():
    base_url = "http://127.0.0.1:8000"
    
    print(f"Testing root endpoint: {base_url}/")
    try:
        response = requests.get(base_url)
        print(f"Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print(f"\nTesting analyze endpoint: {base_url}/api/analyze")
    try:
        # Sending empty body to trigger 400 validation error (confirming route exists)
        response = requests.post(f"{base_url}/api/analyze", json={})
        print(f"Status: {response.status_code}, Response: {response.json()}")
        if response.status_code == 422: # Pydantic validation error or 400
             print("Route exists and is responding to validation.")
        elif response.status_code == 200:
             print("Route exists and returned success.")
        else:
             print(f"Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"Failed to connect to analyze: {e}")

if __name__ == "__main__":
    test_backend()
