import os

def create_project_structure():
    # Main directories
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "notebooks",
        "configs",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create initial files
    files = {
        "README.md": "# Facial Emotion Recognition\n\nMLOps project for emotion recognition from facial expressions.",
        "requirements.txt": "torch>=2.0.0\ntorchvision>=0.15.0\nnumpy>=1.23.0\npandas>=1.5.0\nmatplotlib>=3.5.0\nPillow>=9.0.0\nscikit-learn>=1.0.0\nmlflow>=2.3.0\ndvc>=2.0.0",
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/features/__init__.py": "",
        "src/models/__init__.py": "",
        "src/visualization/__init__.py": ""
    }
    
    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content)
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()