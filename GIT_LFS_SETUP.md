# Git LFS Setup Guide for Leaf Classifier Model

This guide will help you set up Git Large File Storage (Git LFS) to handle the large PyTorch model file (`best_model.pth`, ~180MB) for deployment to Streamlit Cloud.

## What is Git LFS?

Git LFS (Large File Storage) is an extension for Git that replaces large files with text pointers inside Git, while storing the file contents on a remote server. This is essential for managing large model files efficiently.

## Storage Limits

- **Maximum file size**: 2GB per file
- **GitHub free tier**: 1GB storage + 1GB bandwidth per month
- **Additional storage**: Can be purchased from GitHub if needed

## Prerequisites

- Git installed on your system
- Repository cloned locally
- The `best_model.pth` file generated from training

## Installation Instructions

### Windows

**Option 1: Using Git for Windows Installer**
```bash
# If you installed Git for Windows, LFS is likely already included
git lfs version
```

**Option 2: Using Chocolatey**
```bash
choco install git-lfs
```

**Option 3: Manual Download**
1. Download the installer from: https://git-lfs.github.com/
2. Run the installer
3. Open Git Bash or Command Prompt

### macOS

**Option 1: Using Homebrew (Recommended)**
```bash
brew install git-lfs
```

**Option 2: Using MacPorts**
```bash
port install git-lfs
```

**Option 3: Manual Download**
1. Download from: https://git-lfs.github.com/
2. Run the installer

### Linux

**Ubuntu/Debian**
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

**Fedora/RHEL/CentOS**
```bash
sudo dnf install git-lfs
# or
sudo yum install git-lfs
```

**Arch Linux**
```bash
sudo pacman -S git-lfs
```

**Manual Installation (Any Linux)**
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## Setup Git LFS in Your Repository

### 1. Initialize Git LFS

After installing Git LFS, initialize it for your user account (only needed once per machine):

```bash
git lfs install
```

You should see:
```
Updated Git hooks.
Git LFS initialized.
```

### 2. Track Model Files

The repository already includes a `.gitattributes` file that configures Git LFS to track `*.pth` files. This file contains:

```
*.pth filter=lfs diff=lfs merge=lfs -text
best_model.pth filter=lfs diff=lfs merge=lfs -text
```

If you need to add more file patterns, use:
```bash
git lfs track "*.pth"
git lfs track "best_model.pth"
```

### 3. Verify LFS Tracking

Check which files are being tracked by LFS:

```bash
git lfs track
```

You should see:
```
Listing tracked patterns
    *.pth (.gitattributes)
    best_model.pth (.gitattributes)
Listing excluded patterns
```

### 4. Add and Commit the Model File

After training your model and generating `best_model.pth`:

```bash
# Add the model file
git add best_model.pth

# Verify it will be tracked by LFS
git lfs ls-files

# Commit the changes
git commit -m "Add trained model via Git LFS"
```

### 5. Push to GitHub

```bash
git push origin main
```

Git LFS will automatically upload the large file to the LFS server while only storing a pointer in your Git repository.

## Verifying Git LFS Setup

### Check LFS Status
```bash
git lfs status
```

### List LFS Files
```bash
git lfs ls-files
```

This should show your `best_model.pth` file.

### Verify File on GitHub
1. Navigate to your repository on GitHub
2. Click on the `best_model.pth` file
3. You should see a file with LFS metadata instead of binary content
4. The file size should be displayed with an "LFS" badge

### Check Local LFS Cache
```bash
git lfs env
```

This shows your LFS configuration and cache location.

## Deployment to Streamlit Cloud

### Important Notes

1. **Streamlit Cloud supports Git LFS automatically** - No additional configuration needed on the Streamlit side.

2. **Deployment steps**:
   - Push your code with the LFS-tracked model to GitHub
   - Connect your repository to Streamlit Cloud
   - Streamlit will automatically pull LFS files during deployment

3. **First deployment may take longer** - Streamlit needs to download the LFS files.

4. **Bandwidth considerations**: 
   - Each deployment pulls the LFS file, consuming bandwidth
   - GitHub provides 1GB free bandwidth per month
   - Monitor your LFS bandwidth usage in GitHub settings

### Streamlit App Configuration

Create a `requirements.txt` file if you haven't already:
```txt
torch
torchvision
streamlit
numpy
Pillow
```

Create a simple Streamlit app (e.g., `app.py`):
```python
import streamlit as st
import torch
from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Load the model
@st.cache_resource
def load_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
st.write("Model loaded successfully!")
```

## Troubleshooting

### Issue: "Git LFS is not installed"
**Solution**: Follow the installation instructions above for your operating system.

### Issue: "This exceeds GitHub's file size limit of 100 MB"
**Solution**: This means Git LFS is not properly configured. Run:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Issue: "Exceeded LFS bandwidth limit"
**Solution**: 
- Wait until your bandwidth resets (monthly)
- Purchase additional bandwidth from GitHub
- Consider alternative storage solutions (see below)

### Issue: Files not being tracked by LFS
**Solution**: 
1. Check `.gitattributes` exists and contains the correct patterns
2. Run `git lfs track` to verify
3. If files were already committed without LFS:
   ```bash
   git rm --cached best_model.pth
   git add best_model.pth
   git commit -m "Migrate model to LFS"
   ```

### Issue: "Smudge error" or "Filter error"
**Solution**: 
```bash
git lfs install --force
git lfs pull
```

### Issue: Clone is slow or fails
**Solution**: 
Clone without LFS files first, then pull them:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
cd <repo-name>
git lfs pull
```

## Alternative Solutions

If Git LFS doesn't meet your needs, consider these alternatives:

### 1. Cloud Storage Services
- **Google Drive**: Upload model, share with public link, download in app
- **Dropbox**: Similar to Google Drive
- **AWS S3**: Professional option with more control
- **Hugging Face Hub**: Specifically designed for ML models

Example with Hugging Face:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/leaf-classifier",
    filename="best_model.pth"
)
```

### 2. DVC (Data Version Control)
- Specifically designed for ML projects
- Similar to Git LFS but more features
- Supports various cloud storage backends

### 3. Git LFS Alternatives
- **Git Annex**: More complex but powerful
- **Git Fat**: Lightweight alternative

### 4. Model Compression
- Quantize the model to reduce size
- Use pruning techniques
- Convert to ONNX format with optimization

## Best Practices

1. **Don't commit the model file until it's final** - Each commit uses bandwidth
2. **Use `.gitignore` for intermediate models** - Only track the best/final model
3. **Monitor your LFS usage** - Check GitHub Settings → Billing → Git LFS Data
4. **Tag releases** - Tag commits with final models for easy reference
5. **Document model versions** - Keep track of which commit has which model version

## Additional Resources

- [Git LFS Official Documentation](https://git-lfs.github.com/)
- [GitHub LFS Tutorial](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [DVC for ML Projects](https://dvc.org/)

## Quick Reference Commands

```bash
# Install LFS
git lfs install

# Track files
git lfs track "*.pth"

# Check tracked files
git lfs ls-files

# Check LFS status
git lfs status

# Verify tracking patterns
git lfs track

# Migrate existing file to LFS
git lfs migrate import --include="*.pth"

# Pull LFS files
git lfs pull

# Check LFS configuration
git lfs env
```

## Support

If you encounter issues not covered in this guide:
1. Check the [Git LFS Issues page](https://github.com/git-lfs/git-lfs/issues)
2. Review [Streamlit Community Forums](https://discuss.streamlit.io/)
3. Open an issue in this repository

---

**Note**: This repository is already configured with Git LFS. The `.gitattributes` file is set up to track all `.pth` files automatically. You just need to install Git LFS on your machine and push the model file.
