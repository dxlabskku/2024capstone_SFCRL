# SF-CRL: Speech-Facial Contrastive Representation Learning for Speaker Feature Extraction

Recently, several studies have shown that various modalities can be used to extract features from audio data. For example, based on the CLIP methodology, the pre-trained AudioCLIP model achieved state-of-the-art (SOTA) performance on the ESC dataset by extracting generalized features from audio with text and image.
However, most research focuses on general sounds such as rain or animal noises. In this reason, this study aims to extract unique features of individual voices using a dataset of human speech. Several previous studies have demonstrated the close correlation between human speech and facial features such as the jawline and oral structure. Leveraging this correlation, we performed cross-modal contrastive learning between pairs of human face images and speech.

The dataset consists of mel-spectrograms from human speech and corresponding facial images.
Both the audio and image encoders utilized a modified VGG-M architecture. We implemented our model to perform cross-modal contrastive learning between audio and visual features. This model utilizes a custom loss function to maximize the similarity between corresponding audio and visual features.

For Experiment, we calculated speech-face matching accuracy by cosine similarity. Also, we performed retrieving image from audio. We assessed whether the model could match a randomly generated image among five images to a given speech on unseen data (GRID dataset). This approach aims to extract distinctive features of individual voices, potentially enhancing applications in speaker recognition and other areas requiring precise voice analysis.

Recent advancements in multimodal learning have highlighted the potential of integrating various modalities to enhance feature extraction from audio data. However, most research has focused on non-human sounds. This study addresses this gap by extracting unique features of individual voices using human speech datasets.

We propose SF-CRL (Speech-Facial Contrastive Representation Learning), a model leveraging the correlation between speech and facial features. Utilizing mel-spectrograms of speech and corresponding facial images, our model employs modified VGG-M architectures for audio and image encoders and incorporates a custom loss function to maximize the similarity between audio and visual features.

Evaluations on the LRS3 and GRID datasets demonstrate that SF-CRL outperforms benchmark models such as Resemblyzer, Wav2Vec 2.0, and AudioCLIP in speaker similarity and cross-modal retrieval tasks. Our approach effectively captures distinctive voice features, with potential applications in speaker recognition and biometric authentication. Future work will aim to extend the model to diverse languages and facial expressions, enhancing its robustness and versatility.
