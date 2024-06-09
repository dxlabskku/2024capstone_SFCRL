import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioStream(nn.Module):
    def __init__(self):
        super(AudioStream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(5, 7), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1)))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride = (2, 2)))
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer7 = nn.Linear(13312, 512)
        self.layer8 = nn.Linear(512, 128)
    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
    
class VisualStream(nn.Module):
    def __init__(self):
        super(VisualStream, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=96, kernel_size=(10, 7, 7), stride=(1,2,2), padding=(0,0,0)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=96, out_channels=256, kernel_size=(1, 5, 5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1)))
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3),padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride = (1,2,2)))
        self.layer6 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(1, 6, 6), padding=(0,0,0)),
            nn.BatchNorm3d(512),
            nn.ReLU())
        self.layer7 = nn.Linear(512, 512)
        self.layer8 = nn.Linear(512, 128)
    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.flatten(x, 1)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

class LipSync(nn.Module):
    def __init__(self, audio_model, visual_model):
        super().__init__()

        self.audio = audio_model
        self.visual = visual_model

        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        self.lay_audio1 = nn.Linear(128, 256)
        self.lay_audio2 = nn.Linear(256, 256)

        self.lay_visual1 = nn.Linear(128, 256)
        self.lay_visual2 = nn.Linear(256, 512)

        self.criterion = nn.CrossEntropyLoss()
        self.cosine_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

        
    def cross_modal_loss(self, audio_features, visual_features):
        audio_features = F.normalize(audio_features, dim=1)
        visual_features = F.normalize(visual_features, dim=1)
        similarities = audio_features @ visual_features.T
        similarities = torch.exp(self.w * similarities + self.b)
        loss = -torch.sum(torch.log(torch.diag(similarities)/torch.sum(similarities)))/similarities.shape[0]
        return loss
    
    def forward(self, mel, mel_feature, image, img_feature):
        audio_feature = self.audio(mel)
        visual_feature = self.visual(image)
        av_loss = self.cross_modal_loss(audio_feature, visual_feature)

        a_x = F.relu(self.lay_audio1(audio_feature))
        a_x = self.lay_audio2(a_x)

        v_x = F.relu((self.lay_visual1(visual_feature)))
        v_x = self.lay_visual2(v_x)

        
        mel_loss = self.criterion(a_x, mel_feature)
        visual_loss = self.criterion(v_x, img_feature)

        return audio_feature, visual_feature, a_x, v_x, av_loss, mel_loss, visual_loss
    
def train_transform():
    transform_list = [
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.mel_path = f"{data_folder}/mel"
        self.wav_feature_path = f"{data_folder}/wav_feature"
        self.image_path = f"{data_folder}/image"
        self.image_feature_path = f"{data_folder}/img_feature_concat"
        audio_files = os.listdir(self.mel_path)
        self.file_names = [p[:-4] for p in audio_files]
        self.transform = train_transform()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        mel_path = f"{self.mel_path}/{self.file_names[idx]}.npy"
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel).unsqueeze(0)

        mel_feature_file = f"{self.wav_feature_path}/{self.file_names[idx]}.npy"
        mel_feature = np.load(mel_feature_file)
        mel_feature = torch.from_numpy(mel_feature)
        
        img_list = glob.glob(f"{self.image_path}/{self.file_names[idx]}*")
        visual = torch.load(f'{self.image_path}/{self.file_names[idx]}.pt')
        visual_feature = torch.load(f'{self.image_feature_path}/{self.file_names[idx]}.pt')

        return mel, mel_feature, visual, visual_feature
    
# Define customdataset
print("DATASET...")
dataset = CustomDataset("/datapath")

# Set dataloader
batch_size = 100
data_loader = DataLoader(dataset, batch_size=batch_size)

print("MODEL LOAD...")
audio_model = AudioStream().to(device)
visual_model = VisualStream().to(device)
model = LipSync(audio_model, visual_model).to(device)

# Setting parameters
learning_rate = 1e-4
num_epochs = 50
optimizer = torch.optim.AdamW([
                {'params': model.parameters()}
            ], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)

print("MODEL TRAIN...")
av_weight = 1.0
mel_weight = 1.0
visual_weight = 1.0

for e in range(num_epochs):
    epoch_loss = 0
    batch_num = 0
    for batch_idx, (mel, mel_feature, visual, visual_feature) in enumerate(data_loader):
        mel, mel_feature, image, img_feature = mel.to(device), mel_feature.to(device), visual.to(device), visual_feature.to(device)
        audio_feature, visual_feature, a_x, v_x, av_loss, mel_loss, visual_loss = model(mel, mel_feature, image, img_feature)
        loss = av_weight*av_loss + mel_weight*mel_loss + visual_weight*visual_loss
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_num += 1
    print(f"EPOCH : {e+1} / {num_epochs} - Loss : {epoch_loss/num_epochs}")

print("MODEL SAVE..")
PATH = '/modelsavepath'
torch.save(model, PATH + 'model.pt') 
torch.save(model.state_dict(), PATH + 'model_state_dict.pt')
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'all.tar')

##########################
## Downstream Task
##########################

# Load mel data
mel_path ='/meldatapath'
mel = np.load(mel_path)
mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
output = audio_model(mel)

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(vgg16_base.output)
x = Dense(4096, activation='relu')(x)
x = Dense(11, activation='linear')(x)
img_model = Model(inputs=vgg16_base.input, outputs=x)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).resize((224,224))
    image_array = np.array(image) / 255.0  
    if image_array.shape[-1] == 4: 
        image_array = image_array[..., :3]
    return image_array

image_paths = [
    '/imagepath1',
    '/imagepath2',
    '/imagepath3',
    '/imagepath4',
    '/imagepath5'
]

image_vectors = []
for image_path in image_paths:
    image_array = load_and_preprocess_image(image_path)
    image_vector = img_model.predict(np.expand_dims(image_array, axis=0))[0]
    image_vectors.append(image_vector)

def find_most_similar_image(audio_vector, image_vectors):
    similarities = []
    
    for img_vector in image_vectors:
        audio_np = audio_vector.detach().numpy()
        img_vector = img_vector.reshape(1,-1)
        similarity = cosine_similarity(audio_np, img_vector)
        similarities.append(similarity[0][0])
    
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[most_similar_index]

most_similar_index, similarity_score = find_most_similar_image(output, image_vectors)
print(f"Most similar image index: {most_similar_index}")
print(f"Similarity score: {similarity_score}")







