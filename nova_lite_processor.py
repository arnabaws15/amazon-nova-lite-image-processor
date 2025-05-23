#!/usr/bin/env python3
# Script to process images with Bedrock Nova Lite model
# Usage: python nova_lite_processor.py -i image_list.txt -o output_dir -d 300

import argparse
import base64
import boto3
import json
import os
import time
import mimetypes
import sys
from concurrent.futures import ThreadPoolExecutor

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process images with Bedrock Nova Lite model')
parser.add_argument('-i', '--image-list', required=True, help='File containing list of image paths')
parser.add_argument('-o', '--output-dir', default='./output', help='Output directory')
parser.add_argument('-r', '--region', default='us-west-2', help='AWS region')
parser.add_argument('-d', '--duration', type=int, default=300, help='Duration to run in seconds')
parser.add_argument('-t', '--threads', type=int, default=2, help='Number of concurrent threads')
parser.add_argument('-l', '--log-file', default='./nova_lite_errors.log', help='Log file path')
parser.add_argument('-p', '--profile-arn', required=True, help='Inference profile ARN')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Initialize log file
with open(args.log_file, 'w') as log:
    log.write(f"{time.ctime()}: Starting nova-lite image processing script\n")
    log.write(f"{time.ctime()}: Using region: {args.region}\n")
    
    # Log AWS credential environment variables presence (not their values)
    log.write(f"{time.ctime()}: AWS_ACCESS_KEY_ID present: {'AWS_ACCESS_KEY_ID' in os.environ}\n")
    log.write(f"{time.ctime()}: AWS_SECRET_ACCESS_KEY present: {'AWS_SECRET_ACCESS_KEY' in os.environ}\n")
    log.write(f"{time.ctime()}: AWS_SESSION_TOKEN present: {'AWS_SESSION_TOKEN' in os.environ}\n")

# Create a Bedrock Runtime client
# This will automatically use AWS credentials from environment variables:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
client = boto3.client("bedrock-runtime", region_name=args.region)

# The prompt to be used for all requests
PROMPT = '''You are an AI assistant specialized in analyzing screenshots from Zoom meeting calls. Your task is to extract OCR text, generate captions, and provide this information in a structured XML format.
Please follow these steps to analyze the screenshot and generate your response:

1. Extract OCR Text:
- Carefully examine the screenshot and identify all visible text.
- Include text from tables, charts, images, and any other elements present.
- This text will be used to extract keywords and provide context for your analysis.

2. Determine Screenshot Type:
- Identify whether the screenshot is of an interactive session (e.g., coding environment, terminal) or a document/slide.

3. Generate Captions:
- Create two captions: a detailed long caption and a concise short caption.
- Adjust the level of detail based on the complexity of the screenshot.

For interactive sessions:
- Provide a brief, high-level description (1-3 sentences).
- Focus on the environment or tools being used (e.g., IDE, terminal).
- Avoid describing specific content or sensitive information.

For documents or slides:
- For text-heavy slides:
    * Describe key elements such as titles, dates, names, and terms visible in the slide.
    * Include section headers, bullet points, agenda items, project names, timelines, deliverables, performance metrics, recommendations, meeting objectives, key terms, calls to action, presenter or team information, citations, and status updates.
- For slides with visual elements (graphs, charts, tables):
    * Describe all visible elements in detail, including chart titles, axis labels, legends, data points, and visual highlights.
    * Include exact numbers when relevant.
    * If there are multiple figures or graphs, describe each one in detail.

4. Format Your Response:
Present your analysis and output in the following XML structure:

<ocr>
[All extracted OCR text]
</ocr>

<caption>
[Detailed caption describing the screenshot content]
</caption>

<short_caption>
[Concise summary of the screenshot (1-2 sentences)]
</short_caption>

Study the following examples carefully. They illustrate the expected level of detail and format for different types of screenshots. Use these examples as a guide for your own analysis and output generation.'''

def process_image(image_path):
    """Process a single image with the Nova Lite model"""
    try:
        # Log processing start
        with open(args.log_file, 'a') as log:
            mime_type = mimetypes.guess_type(image_path)[0] or 'application/octet-stream'
            log.write(f"{time.ctime()}: Processing image: {image_path} (MIME type: {mime_type})\n")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_string = base64.b64encode(binary_data).decode("utf-8")
        
        # Determine image format
        image_format = "jpeg"
        if mime_type and "png" in mime_type:
            image_format = "png"
        
        # Create the request
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": base64_string}
                        }
                    },
                    {
                        "text": PROMPT
                    }
                ]
            }
        ]
        
        # Configure inference parameters
        inf_params = {"maxTokens": 3000, "topP": 0.1, "topK": 20, "temperature": 0.3}
        
        # Create the full request
        request = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "inferenceConfig": inf_params
        }
        
        # Invoke the model
        timestamp = int(time.time())
        output_file = os.path.join(args.output_dir, f"{os.path.basename(image_path)}_{timestamp}.txt")
        response_file = f"{output_file}_response.json"
        
        response = client.invoke_model(
            modelId=args.profile_arn,
            body=json.dumps(request)
        )
        
        # Process the response
        model_response = json.loads(response["body"].read())
        
        # Save the full response
        with open(response_file, 'w') as f:
            json.dump(model_response, f, indent=2)
        
        # Extract and save the text content
        content_text = model_response["output"]["message"]["content"][0]["text"]
        with open(output_file, 'w') as f:
            f.write(content_text)
        
        print(f"Response saved to: {output_file}")
        return True
        
    except Exception as e:
        # Log any errors
        with open(args.log_file, 'a') as log:
            log.write(f"{time.ctime()}: Error processing {image_path}: {str(e)}\n")
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_all_images():
    """Process all images in the list using multiple threads"""
    # Read the image list
    with open(args.image_list, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(image_paths)} images in list")
    
    # Process images with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        executor.map(process_image, image_paths)

# Main execution loop
print(f"Starting processing loop for {args.duration} seconds...")
end_time = time.time() + args.duration

while time.time() < end_time:
    print(f"Starting new batch at {time.ctime()}")
    process_all_images()
    print(f"Batch completed at {time.ctime()}")

print(f"Processing completed after running for {args.duration} seconds. Results saved in {args.output_dir}")
