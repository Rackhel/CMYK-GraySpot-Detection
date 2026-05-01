# CMYK Printer Project

## Setup Instructions

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Build Docker Image
docker build -t cmyk-project .

# Create a data folder if needed(for storing the dataset)
mkdir -p data

# Run with volume mount (dataset will persist on your computer) (remove --rm if you want to keep container)
docker run --rm -v ${PWD}/data:/app/data cmyk-project
```