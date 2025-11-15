# Deploying Leaf Classifier on Streamlit

## Local Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser at `http://localhost:8501`

## Streamlit Cloud Deployment

1. **Push code to GitHub:**
   - Ensure `app.py`, `requirements.txt`, and `custom_model.pth` are in your repo

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `Messaoudi-Belkacem/leaf-classifier`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Important Notes:**
   - Your model file (`custom_model.pth`) must be in the repository
   - If the file is too large for GitHub (>100MB), consider using Git LFS or storing it elsewhere

## Troubleshooting

- **Model file too large:** Use Git LFS or host the model on cloud storage
- **Memory issues:** Consider using a smaller model or optimize the deployment settings
- **Class names mismatch:** Update `CLASS_NAMES` in `app.py` to match your dataset