# Nova Lite Image Processor

This Python script processes images with the Amazon Bedrock Nova Lite model to extract OCR text and generate captions.

## Prerequisites

- Python 3.6+
- Required Python packages: boto3, argparse
- AWS credentials configured in your environment

## AWS Credentials

The script uses AWS credentials from your shell environment. Make sure the following environment variables are set:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_SESSION_TOKEN="your-session-token"  # if using temporary credentials
```

## Usage

```bash
python nova_lite_processor.py -i image_list.txt -o output_dir -p "your-inference-profile-arn"
```

Parameters:
- `-i, --image-list`: Path to the image list file (required)
- `-o, --output-dir`: Output directory (default: ./output)
- `-r, --region`: AWS region (default: us-west-2)
- `-d, --duration`: Duration to run in seconds (default: 300)
- `-t, --threads`: Number of concurrent threads (default: 2)
- `-l, --log-file`: Log file path (default: ./nova_lite_errors.log)
- `-p, --profile-arn`: Inference profile ARN (required)

## Image List Format

The image list file should contain one image path per line:

```
/path/to/image1.jpg
/path/to/image2.png
/path/to/image3.jpg
```

## Output

For each image, the script will create:
- A text file with the extracted OCR text and captions
- A JSON file with the full API response

All output files will be saved in the specified output directory.