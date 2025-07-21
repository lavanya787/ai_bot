"""
Setup script for AI Document Assistant
Run this script to set up the application dependencies and initial configuration.
"""

import subprocess
import sys
import os
import nltk
import spacy
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def download_spacy_model():
    """Download spaCy English model"""
    print("🧠 Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading spaCy model: {e}")
        print("You can manually install it with: python -m spacy download en_core_web_sm")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "saved_models",
        "uploads",
        "temp",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created successfully!")
    return True

def create_env_file():
    """Create .env file template"""
    env_content = """# AI Document Assistant Configuration
DOMAIN=general
DEBUG=False

# Database Configuration (Optional - using Supabase)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Model Configuration
DEFAULT_MODEL_PATH=checkpoint.pt
DEFAULT_TOKENIZER_PATH=uml_tokenizer.pkl
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
    else:
        print("ℹ️  .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Document Assistant...")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Install requirements
    if install_requirements():
        success_count += 1
    
    # Step 2: Download NLTK data
    if download_nltk_data():
        success_count += 1
    
    # Step 3: Download spaCy model
    if download_spacy_model():
        success_count += 1
    
    # Step 4: Create directories
    if create_directories():
        success_count += 1
    
    # Step 5: Create env file
    create_env_file()
    success_count += 1
    
    print("=" * 50)
    print(f"Setup completed: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("✅ Setup completed successfully!")
        print("\n🚀 To run the application:")
        print("   streamlit run main.py")
    else:
        print("⚠️  Setup completed with some issues. Please check the errors above.")
        print("\n🔧 You may need to manually install missing dependencies.")

if __name__ == "__main__":
    main()