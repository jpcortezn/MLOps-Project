from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
sys.path.append("../src/models")
from model import ResidualEmotionCNN

app = FastAPI()

model = ResidualEmotionCNN()
model.load_state_dict(torch.load("../models/final_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

EMOTION_CLASSES = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Felicidad",
    4: "Neutral",
    5: "Tristeza",
    6: "Sorpresa"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            _, prediction = torch.max(output, 1)
        emotion = EMOTION_CLASSES.get(prediction.item(), "Desconocido")
        return {"prediction": emotion}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
