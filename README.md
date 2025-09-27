# Person-Re-Identification

Person Re-identification with a Siamese CLIP-based Network
This repository contains the implementation of a Person Re-identification (ReID) model using a Siamese network architecture with a CLIP backbone. The project aims to identify and match individuals across different camera views, a fundamental task in computer vision and surveillance.

Introduction
Person Re-identification is the task of recognizing a person from a gallery of images or videos, captured by different cameras and at different times. Our approach leverages a Siamese network, an architecture designed to learn a similarity metric between two inputs, to determine if two images belong to the same person.

To enhance the model's ability to learn robust, representative features, we use OpenAI's CLIP (Contrastive Language-Image Pre-training) as the backbone. By leveraging CLIP's powerful image encoder, we extract high-quality, general-purpose features that are highly effective for the ReID task.

Architecture Details
The core architecture consists of a Siamese network with shared weights, where each branch processes an input image.

Backbone: The CLIP image encoder extracts a feature vector from each person's image.

Projection Layer: A dense layer is added on top of the fixed CLIP backbone to project the features into a more discriminative embedding space, optimized specifically for the ReID task.

Metric Learning: The network is trained using a triplet loss function, which minimizes the distance between embeddings of the same person (positives) and maximizes the distance between embeddings of different people (negatives).

A variant, UNet+CLIP+fc, was also implemented to incorporate segmentation information, although the results favored the base CLIP approach (see Results section).

Datasets
The model was trained and evaluated on four widely used person re-identification benchmark datasets:

Market-1501: Large-scale dataset captured from 6 cameras.

CUHK03: Dataset from 2 cameras, featuring both manually labeled and detected bounding boxes.

DukeMTMC: Large-scale re-ID dataset collected from 8 cameras.

Occluded-Duke: A challenging variation of the DukeMTMC dataset that specifically includes occluded pedestrian images, testing robustness.

Results
The following tables show the model's performance in terms of Rank-1 accuracy (identifying the correct match as the top result) and mAP (mean Average Precision) across the test sets. Results are reported for the case where same-camera retrievals are included.

Rank-1 Accuracy
(model)

Market1501

CUHK03

DukeMTMC

Occluded-Duke

CLIP+fc

98.87%

95.36%

93.00%

96.79%

UNet+CLIP+fc

97.98%

94.50%

92.10%

96.20%

mAP (mean Average Precision)
(model)

Market1501

CUHK03

DukeMTMC

Occluded-Duke

CLIP+fc

81.72%

79.71%

72.21%

50.13%

UNet+CLIP+fc

78.91%

73.02%

70.68%

49.03%

The superior performance of the CLIP+fc model suggests that the robust, generalized features learned by the pre-trained CLIP backbone are highly effective and require minimal adaptation for this specific task.

How to Get Started
Clone the repository:

git clone [https://github.com/SahibParmar/Person-Re-Identification.git](https://github.com/SahibParmar/Person-Re-Identification.git)
cd Person-Re-Identification

Install dependencies:
(Add your specific installation commands here, e.g., using a requirements.txt file or conda environment)

Prepare datasets:
(Add instructions on dataset structure/preprocessing here)

Run Training / Evaluation:
(Add example commands to start training or evaluation here)
