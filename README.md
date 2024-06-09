# SF-CRL: Speech-Facial Contrastive Representation Learning for Speaker Feature Extraction

Recently, several studies have shown that various modalities can be used to extract features from audio data. For example, based on the CLIP methodology, the pre-trained AudioCLIP model achieved state-of-the-art (SOTA) performance on the ESC dataset by extracting generalized features from audio with text and image. However, most research focuses on general sounds such as rain or animal noises. In this reason, this study aims to extract unique features of individual voices using a dataset of human speech.

Several previous studies have demonstrated the close correlation between human speech and facial features such as the jawline and oral structure. Leveraging this correlation, we performed cross-modal contrastive learning between pairs of human face images and speech. Both the audio and image encoders utilized a modified VGG-M architecture. We used Resemblyzer to verify speech-face matching accuracy by comparing cosine similarity.

For the downstream task, we performed classification. We assessed whether the model could match a randomly generated image among five images to a given speech on unseen data.

This approach aims to extract distinctive features of individual voices, potentially enhancing applications in speaker recognition and other areas requiring precise voice analysis.
