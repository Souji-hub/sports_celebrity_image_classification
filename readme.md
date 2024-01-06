# Sports Celebrity Image Classification Project

This project aims to classify sports celebrity images into five categories: 'lionel_messi', 'maria_sharapova', 'roger_federer', 'serena_williams', and 'virat_kohli'. The classification model is based on Support Vector Machines (SVM), and its performance has been evaluated using GridSearchCV, with SVM yielding the best results.

## Project Overview

- **Categories:**
  - 'lionel_messi': 0
  - 'maria_sharapova': 1
  - 'roger_federer': 2
  - 'serena_williams': 3
  - 'virat_kohli': 4

- **Model Performance:**
  - Test Score: 0.914286
  - Average Score on Test and Validation: 0.871429

## Data Cleaning

The images were preprocessed using the following steps:

1. **Face and Eyes Recognition:**
   - Images were cropped using Haarcascade for face and eyes recognition.
   - The goal was to focus on the regions containing faces and ensure that each image has at least two eyes for better classification.

2. **Wavelet Transform:**
   - The model was tuned using wavelet transform.
   - The feature vector 'x' consists of 4096 features, including both raw cropped image features and wavelet transform features.

## Model Training

- **Model Used:**
  - Support Vector Machines (SVM)

- **Evaluation Method:**
  - GridSearchCV was employed to evaluate and compare the performance of different models.
  - SVM showed the best results with a test score of 0.914286 and an average score of 0.871429 on test and validation sets.

## File Structure

- **`classifier.py`:**
  - Python script containing functions for image preprocessing, model loading, and image classification.

- **`app.py`:**
  - Flask server script for providing an API endpoint to classify images.

- **`model/`:**
  - Directory containing the SVM model file (`model.pkl`), Haarcascade XML files, and other artifacts.

- **`server/`:**
  - Directory containing additional server-related files, including the `artifacts/` directory with class dictionaries.

## Usage

1. **Setup:**
   - Ensure required dependencies are installed (`joblib`, `numpy`, `opencv-python`, `scikit-learn`, etc.).

2. **Run Flask Server:**
   - Execute `python app.py` to run the Flask server.

3. **Make Classification Requests:**
   - Send image files or base64-encoded strings to the `/classify` endpoint using POST requests.

## How to Contribute

If you would like to contribute to the project, feel free to fork the repository, create a branch, and submit a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

