# Pet-Nose-Localization
Processing and analyzing images through a custom dataset and a deep learning model based on a modified ResNet architecture.

This GitHub repository introduces a comprehensive framework for processing and analyzing images through a custom dataset and a deep learning model based on a modified ResNet architecture. The project is tailored for a specific dataset that requires custom loading, preprocessing, and labeling mechanisms, encapsulating the core functionalities within a CustomDataset class and a neural network model named CustomNet.

The CustomDataset class is designed to manage image data efficiently, handling operations such as reading image files, applying transformations, and parsing label information from a designated annotation file. It supports both training and evaluation modes, making it versatile for various machine learning tasks.

The CustomNet model extends the capabilities of the standard ResNet model by integrating additional layers and mechanisms to cater to the specific needs of the task at hand. It emphasizes on processing encoded features through a custom frontend network, aiming to optimize performance for the target application.

This setup demonstrates a practical application of deep learning techniques for custom image analysis tasks, providing a foundation for further experimentation and development in similar domains.
