from django.shortcuts import render
from django.http import JsonResponse
import json
from .mlp_model import MLP
import torch

# Create your views here.
model = MLP()
model.load_state_dict(torch.load("ml_api/mnist_mlp.pth", map_location="cpu"))
model.eval()

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def predict(request):
    if request.method == "POST":
        data = json.loads(request.body)
        pixels = data["pixels"]
        pixels_tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)  # shape [1,
        with torch.no_grad():
            output = model.forward(pixels_tensor)
            prediction = torch.argmax(output).item()
            return JsonResponse({"message": prediction})

