import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 모델 로드 함수 정의
def load_model(path):
    model = torch.load(path)
    model.eval()  # 평가 모드로 설정
    return model

# 이미지 분류 함수 정의
def classify_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델이 기대하는 입력 크기로 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 배치 차원을 추가
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 모델 로드
model_paths = [
    './model/aram_model1.pt',
    './model/aram_model2.pt',
    './model/aram_model3.pt',
    './model/aram_model4.pt',
    './model/aram_model5.pt',
    './model/aram_model6.pt'
]

models = [load_model(path) for path in model_paths]

st.title("Image Classification with Multiple Models")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # 이미지를 열고 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # 각 모델에 이미지를 입력하여 결과를 얻음
    st.write("Classifying...")

    results = [classify_image(image, model) for model in models]

    for i, result in enumerate(results):
        st.write(f"Model {i+1}: {result}")
