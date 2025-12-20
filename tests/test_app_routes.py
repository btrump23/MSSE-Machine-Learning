from app import app

def test_index_loads():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200

def test_predict_csv_requires_file():
    client = app.test_client()
    r = client.post("/predict-csv", data={})
    assert r.status_code in (400, 415)
