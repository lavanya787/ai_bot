import nltk
import os
import shutil

def fix_nltk_stopwords():
    try:
        # Get nltk_data directories
        nltk_paths = nltk.data.path
        found = False

        for path in nltk_paths:
            stopwords_dir = os.path.join(path, 'corpora', 'stopwords')
            if os.path.exists(stopwords_dir):
                print(f"🧹 Deleting corrupted stopwords at: {stopwords_dir}")
                shutil.rmtree(stopwords_dir)
                found = True

        if not found:
            print("✅ No corrupted stopwords folder found, proceeding to download.")

        # Redownload stopwords
        print("⬇️ Downloading fresh stopwords corpus...")
        nltk.download("stopwords")

        print("✅ Stopwords fixed and downloaded successfully.")

    except Exception as e:
        print(f"❌ Error while fixing stopwords: {e}")

# Run the function
fix_nltk_stopwords()
