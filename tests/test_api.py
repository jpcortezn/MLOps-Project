import io
import os
import logging
import pytest
import requests
from PIL import Image

# Configuración de logging para los tests
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# URL base de tu API (ajusta si usas otro puerto o host)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def create_test_image(format="JPEG") -> io.BytesIO:
    """
    Genera en memoria una imagen 100×100 px en blanco.
    """
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf


def test_predict_valid_image():
    logger.info("Iniciando test: test_predict_valid_image")
    buf = create_test_image()
    files = {"file": ("test.jpg", buf, "image/jpeg")}
    resp = requests.post(f"{API_URL}/predict", files=files)
    logger.info(f"Status code: {resp.status_code}")
    logger.info(f"Response JSON: {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in {
        "Enojo", "Disgusto", "Miedo", "Felicidad",
        "Neutral", "Tristeza", "Sorpresa"
    }
    logger.info("test_predict_valid_image completado exitosamente")


def test_predict_no_file():
    logger.info("Iniciando test: test_predict_no_file")
    resp = requests.post(f"{API_URL}/predict", data={})
    logger.info(f"Status code: {resp.status_code}")
    assert resp.status_code == 422
    logger.info("test_predict_no_file completado exitosamente")


@pytest.mark.parametrize("filename,content,type_", [
    ("test.pdf", b"%PDF-1.4 contenido falso", "application/pdf"),
    (
        "test.docx",
        b"PK\x03\x04 contenido falso",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
])
def test_predict_invalid_binary_file(filename, content, type_):
    logger.info(f"Iniciando test: test_predict_invalid_binary_file con archivo {filename}")
    buf = io.BytesIO(content)
    buf.seek(0)
    files = {"file": (filename, buf, type_)}
    resp = requests.post(f"{API_URL}/predict", files=files)
    logger.info(f"Status code: {resp.status_code}")
    try:
        logger.info(f"Response JSON: {resp.json()}")
    except Exception:
        logger.warning("No se pudo parsear JSON de la respuesta")
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    logger.info(f"test_predict_invalid_binary_file para {filename} completado exitosamente")


def test_predict_get_not_allowed():
    logger.info("Iniciando test: test_predict_get_not_allowed")
    resp = requests.get(f"{API_URL}/predict")
    logger.info(f"Status code: {resp.status_code}")
    assert resp.status_code == 405
    logger.info("test_predict_get_not_allowed completado exitosamente")


def test_metrics_endpoint():
    logger.info("Iniciando test: test_metrics_endpoint")
    resp = requests.get(f"{API_URL}/metrics")
    logger.info(f"Status code: {resp.status_code}")
    text = resp.text
    logger.info(f"Metrics response snippet: {text[:200]}...")
    assert resp.status_code == 200
    assert "process_cpu_seconds_total" in text or "python_gc_objects_collected_total" in text
    logger.info("test_metrics_endpoint completado exitosamente")
