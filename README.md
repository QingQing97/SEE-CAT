# SEE-CAT
Our code framework references the publicly available repository https://github.com/TaoShi1998/MultiEMO-ACL2023.git

Stage1

Text
1. python text_code/baselineTrain.py
2. python text_code/meld_pkltrans.py
3. python text_code/iemocap_pkltrans.py

Audio
1. python audio_code/video2audio.py
2. Ultimate Vocal Remover
3. python audio_code/audio2feature_meld.py
4. python audio_code/audio2feature_iemocap.py
5. python audio_code/csv2pkl_meld.py
6. python audio_code/csv2pkl_iemocap.py


Visual
1. python Models\FrameFeatureExtractionSaver_EfficientNet_frozen.py
2. python Models\Speaker_selection.py
3. python Models\Speaker_encoder.py
4. python Train\Train_VisualEncoder_EfficientNet_frozen_DGAF_batch.py

Stage2

python Train\TrainSEECAT_ComplexGatedMultimodalUnit_HNM_LossBalanceLog2.py


Other details
1. The random seeds are 983, 247, 615, 34, 872. 
2. Different part of the pipeline are conducted on NVIDIA A100 (Text), 5090D V2 (Inference) and RTX 3090 GPUs (Other parts), respectively.
3. For feature extraction, the average utterance time costs are approximately 7.4 ms for text, 9.7 s for audio, and 8.1 s for visual. 
