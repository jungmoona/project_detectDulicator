import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# 이미지 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 모델 불러오기
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# 진행률 계산 함수 정의
def calculate_progress(current_index, total_files):
    progress = ((current_index + 1) / total_files) * 100
    return f'{progress:.2f}%'

# 타겟 폴더 및 하위 폴더의 이미지 경로를 재귀적으로 읽어오는 함수 정의
def find_images_in_folder(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 가속 환경 설정
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()

# 유사한 이미지를 찾을 함수 정의
def find_similar_images(target_folder, similarity_threshold=0.8):
    # 타겟 폴더 내의 이미지 파일들의 경로 리스트 생성
    #image_paths = [os.path.join(target_folder, filename) for filename in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, filename))]
    image_paths = find_images_in_folder(target_folder)
    # 유사한 이미지 저장할 리스트
    similar_images = []
    print("start!")
    # 이미지들 간의 유사도 계산
    for i, image_path_i in enumerate(image_paths):
        target_image = Image.open(image_path_i)
        target_tensor = preprocess(target_image).unsqueeze(0).to(device)
        with torch.no_grad():
            target_features = model(target_tensor)

        for j, image_path_j in enumerate(image_paths[i+1:]):  # 자기 자신과는 비교하지 않음
            compared_image = Image.open(image_path_j)
            compared_tensor = preprocess(compared_image).unsqueeze(0).to(device)
            with torch.no_grad():
                compared_features = model(compared_tensor)
            similarity_score = torch.nn.functional.cosine_similarity(target_features, compared_features).item()
            if similarity_score >= similarity_threshold:
                similar_images.append((image_path_i, image_path_j, similarity_score))
            print(calculate_progress(j, len(image_paths)),end='\n\r')
            print(f'current {image_path_i} target:{image_path_j} similarity_score:{similarity_score}', end='\r' )


    return similar_images

# 타겟 폴더 설정
target_folder = "d:/서영이사진"

# 유사한 이미지 찾기
similar_images = find_similar_images(target_folder)
for image_i, image_j, similarity in similar_images:
    if similarity >= 0.97:
        print(f'Image {image_i} 와 Image {image_j}의 유사도: {similarity}')
print("complete")