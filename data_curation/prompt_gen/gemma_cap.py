#Taken from: https://github.com/huggingface/diffusers/discussions/7953
import os
import logging
import argparse
import requests
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

logger = logging.getLogger("Captioner")

# Function to load PaliGemma model and processor
def load_pali_gemma_model(args):
    model_id = args.model_path
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(torch.float32).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_caption_with_pali_gemma(image_path, processor, model, query_strings, do_sample=False, temperature=0.7):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    #NOTE: it can take mupliple query strings, for dev.
    model_inputs = processor(text=query_strings, images=[image] * len(query_strings), return_tensors="pt")
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs, 
            max_new_tokens=100, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=0.9, 
            top_k=50
        )
        outputs = []
        for _generation in generation:
            decoded = processor.decode(_generation[input_len:], skip_special_tokens=True)
            outputs.append(decoded)
    return outputs

def process_image(args, image_path, alt_text, model, processor):
    query_string=(
    f"The image presented came from a web"
    f"and had the alt-text: {alt_text}. Please describe what is in the image "
    f"using the alt-text and the page title as a guide to ground your response. "
    f"For example, if the alt-text contains a specific brand name, use that brand "
    f"name in your output. Please be descriptive but concise. DO NOT make things up. "
    f"If you can't tell something with certainty in the image, simply don't say "
    f"anything about it.\ncaption en"
    )
    result= generate_caption_with_pali_gemma(image_path, processor, model, query_string)
    return result

def process_directory(args, image_dir, output_parquet, model, processor):
    records = []
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    print(f"Parquet: {parquet_path}")
    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            logging.info(f"Found directory to traverse: {full_filepath}")
            process_directory(args, full_filepath, output_parquet, model, processor)
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                logging.info(f"Attempting to load image: {filename}")

                base_name = os.path.splitext(filename)[0]
                alt_text_path = os.path.join(image_dir, f"{base_name}.txt")
                if os.path.isfile(alt_text_path):
                    with open(alt_text_path, "r") as f:
                        alt_text = f.read().strip()
                    logging.info(f"Found alt-text for {filename}: {alt_text[:80]}...")
                else:
                    alt_text = ""
                    logging.info(f"No alt-text file for {filename}, using empty alt-text")

                result = process_image(args, full_filepath, alt_text, model, processor)
                if isinstance(result, list):
                    result = result[0]
                logging.info(f"Best match for {filename}: {result}")

                with Image.open(full_filepath) as img_file:
                    image_bytes = img_file.tobytes()

                records.append({
                    "filename": filename,
                    "caption": result,
                    "image": image_bytes
                })

            except Exception as e:
                import traceback
                logging.error(f"Error processing {filename}: {str(e)}, traceback: {traceback.format_exc()}")
                if "CUDA error" in str(e):
                    import sys
                    sys.exit(1)

    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, engine="pyarrow")
    logging.info(f"Processed Parquet file saved to {parquet_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process images and generate captions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to the output Parquet dataset.")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="fp16", help=("Precision for loading the model. Default: fp16"))
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-448", help=("Model path to load. Default: google/paligemma-3b-mix-448"))

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    model, processor = load_pali_gemma_model(args)
    process_directory(args, args.input_dir, args.output_parquet, model, processor)

if __name__ == "__main__":
    main()