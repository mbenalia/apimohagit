import unittest
from fastapi.testclient import TestClient
from apimohagit import app  # Assurez-vous que 'apimoha' est le nom correct de votre fichier

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Créer un client de test
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "API is up and running!"})

    def test_predict_endpoint(self):
        # Remplacez les valeurs par des exemples valides pour votre modèle
        test_data = {
            "sepal_width": 3.5,
            "petal_length": 1.5,
            "petal_width": 0.2
        }
        response = self.client.post("/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_sepal_length", response.json())

if __name__ == "__main__":
    unittest.main()
