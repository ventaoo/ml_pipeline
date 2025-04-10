import requests
import pytest

@pytest.fixture
def base_url():
    return "http://localhost:8000"

def test_predict_endpoint(base_url):
    # 正常测试用例
    response = requests.get(f"{base_url}/predict/good product")
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], int)

    # 异常测试用例
    response = requests.get(f"{base_url}/predict/!@#$%")
    assert response.status_code == 200  # 根据你的输入校验逻辑调整