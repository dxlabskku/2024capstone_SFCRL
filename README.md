# SF-CRL: Speech-Facial Contrastive Representation Learning for Speaker Feature Extraction

Recently, several studies have shown that various modalities can be used to extract features from audio data. For example, based on the CLIP methodology, the pre-trained AudioCLIP model achieved state-of-the-art (SOTA) performance on the ESC dataset by extracting generalized features from audio with text and image.
However, most research focuses on general sounds such as rain or animal noises. In this reason, this study aims to extract unique features of individual voices using a dataset of human speech. Several previous studies have demonstrated the close correlation between human speech and facial features such as the jawline and oral structure. Leveraging this correlation, we performed cross-modal contrastive learning between pairs of human face images and speech.
Recent advancements in multimodal learning have highlighted the potential of integrating various modalities to enhance feature extraction from audio data. However, most research has focused on non-human sounds. This study addresses this gap by extracting unique features of individual voices using human speech datasets.
We propose SF-CRL (Speech-Facial Contrastive Representation Learning), a model leveraging the correlation between speech and facial features. 

## Model

![model-pretrain](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/99e568d1-9201-4263-9263-d136ffb5bd60)

The image shows the overall architecture of SF-CRL. Utilizing mel-spectrograms of speech and corresponding facial images, our model employs modified VGG-M architectures for audio and image encoders and incorporates a custom loss function to maximize the similarity between audio and visual features. Both the audio and image encoders utilized a modified VGG-M architecture. We implemented our model to perform cross-modal contrastive learning between audio and visual features. This model utilizes a custom loss function to maximize the similarity between corresponding audio and visual features. To train the model, run the following code.

```
python model.py
```

## Experiment

For Experiment, we calculated speech-face matching accuracy by cosine similarity. Also, we performed retrieving image from audio. We assessed whether the model could match a randomly generated image among five images to a given speech on unseen data (GRID dataset). This approach aims to extract distinctive features of individual voices, potentially enhancing applications in speaker recognition and other areas requiring precise voice analysis.

Evaluations on the LRS3 and GRID datasets demonstrate that SF-CRL outperforms benchmark models such as Resemblyzer, Wav2Vec 2.0, and AudioCLIP in speaker similarity and cross-modal retrieval tasks. Our approach effectively captures distinctive voice features, with potential applications in speaker recognition and biometric authentication. 

- Evaluation Result
  
  ![image](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/d7d33fea-02ab-4f78-a89e-dbea752a9796)


- Speaker Scatter Plot

  ![image](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/af353c93-a64c-44fa-8cba-aab7771ef566)


- Encoder Attention Map
  - Audio feature extractor
    
    ![image](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/fef30fe1-6fb0-499a-8d20-d184c5794a01)

  - Visual feature extractor
    
    ![image](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/46d8c405-2be9-469c-9ece-8ec8a480b2ab)

  
- Ablation Study for Feature-Matching Loss
  
  ![image](https://github.com/dxlabskku/2024capstone_SFCRL/assets/149983937/9b8d3afe-dce2-430e-b69b-59d249f4c967)



