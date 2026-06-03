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

from dotenv import load_dotenv

logger = logging.getLogger("Captioner")

# Function to load PaliGemma model and processor
def load_pali_gemma_model(args):
    load_dotenv()
    model_id = args.model_path
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                              device_map="auto",
                                                              torch_dtype=torch.float32,
                                                              low_cpu_mem_usage=True,
                                                              cache_dir=args.model_dir,
                                                            #   attn_implementation="flash_attention_2"
                                                              )                                                           
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_caption_with_pali_gemma(image_path, processor, model, query_strings, do_sample=False, temperature=0.3):
    if image_path.startswith("http://") or image_path.startswith("https://"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    if 'A' in image.mode:
        image= image.convert("RGB")
    #NOTE: it can take mupliple query strings, for dev.
    model_inputs = processor(text=query_strings, images=[image] * len(query_strings), return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs, 
            max_new_tokens=100, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=0.9, 
            top_k=10
        )
        outputs = []
        for _generation in generation:
            decoded = processor.decode(_generation[input_len:], skip_special_tokens=True)
            outputs.append(decoded)
    return outputs

def process_image(args, image_path, alt_text, model, processor):
    if alt_text:
        # query_string = f"The image contain the alt_text: {alt_text} as a guide to ground your response. Please be concise. <image>caption en\n"
        query_string=(
    f"The image presented came from a web"
    f"and had the alt-text: {alt_text}. Please describe what is in the image "
    f"using the alt-text and the page title as a guide to ground your response. "
    f"For example, if the alt-text contains a specific brand name, use that brand "
    f"name in your output. Please be descriptive but concise. DO NOT make things up. "
    f"If you can't tell something with certainty in the image, simply don't say "
    f"anything about it.\n<image>caption en"
    )
    else:
        query_string = f"<image>caption en"
    result= generate_caption_with_pali_gemma(image_path, processor, model, [query_string])
    return result


def generate_captions_batch(image_paths, alt_texts, model, processor):
    """Process multiple images+queries in a single forward pass."""
    images = []
    for path in image_paths:
        if path.startswith("http://") or path.startswith("https://"):
            img = Image.open(requests.get(path, stream=True).raw)
        else:
            img = Image.open(path)
        if 'A' in img.mode:
            img = img.convert("RGB")
        images.append(img)

    queries = []
    for alt_text in alt_texts:
        if alt_text:
            query = (
                f"The image presented came from a web"
                f"and had the alt-text: {alt_text}. Please describe what is in the image "
                f"using the alt-text and the page title as a guide to ground your response. "
                f"For example, if the alt-text contains a specific brand name, use that brand "
                f"name in your output. Please be descriptive but concise. DO NOT make things up. "
                f"If you can't tell something with certainty in the image, simply don't say "
                f"anything about it.\n<image>caption en"
            )
        else:
            query = "<image>caption en"
        queries.append(query)

    model_inputs = processor(text=queries, images=images, return_tensors="pt", padding=True)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=False, temperature=0.3, top_p=0.9, top_k=10)
    outputs = []
    for i, gen in enumerate(generation):
        # input_len = model_inputs["attention_mask"][i].sum().item()
        input_len = model_inputs["input_ids"].shape[-1] # image + text tokens length
        outputs.append(processor.decode(gen[input_len:], skip_special_tokens=True).strip())
    return outputs


def process_directory(args, image_dir, output_parquet, model, processor):
    records = []
    batch_size = getattr(args, "batch_size", 1)
    parquet_path = f"{output_parquet}.{os.path.basename(image_dir)}.parquet"
    print(f"Parquet: {parquet_path}")

    # First pass: collect all image files (recursively handling subdirs)
    image_files = []
    for filename in sorted(os.listdir(image_dir)):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            logging.info(f"Found directory to traverse: {full_filepath}")
            process_directory(args, full_filepath, output_parquet, model, processor)
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(filename)

    if not image_files:
        logging.info(f"No image files found in {image_dir}, skipping.")
        return

    # Second pass: process images in batches
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(0, len(image_files), batch_size), desc="Processing Batches", total=num_batches):
        batch_filenames = image_files[batch_idx:batch_idx + batch_size]
        batch_paths = []
        batch_alt_texts = []

        for filename in batch_filenames:
            full_filepath = os.path.join(image_dir, filename)
            logging.info(f"Attempting to load image: {filename}")

            base_name = os.path.splitext(filename)[0]
            alt_text_path = os.path.join(image_dir, f"{base_name}.txt")
            if os.path.isfile(alt_text_path):
                with open(alt_text_path, "r") as f:
                    alt_text = f.read().strip()
                logging.info(f"Found alt-text for {filename}: {alt_text[:80]}...")
            else:
                alt_text = None
                logging.info(f"No alt-text file for {filename}, using empty alt-text")

            batch_paths.append(full_filepath)
            batch_alt_texts.append(alt_text)

        try:
            results = generate_captions_batch(batch_paths, batch_alt_texts, model, processor)

            for filename, alt_text, result in zip(batch_filenames, batch_alt_texts, results):
                logging.info(f"Caption for {filename}: {result}")
                records.append({
                    "filename": filename,
                    "alt_text": alt_text,
                    "caption": result,
                })

            # Write incrementally after each batch so progress is saved
            df = pd.DataFrame(records)
            df.to_parquet(parquet_path, engine="pyarrow")
            logging.info(f"Batch saved — {len(records)} records written to {parquet_path}")

        except Exception as e:
            import traceback
            logging.error(f"Error processing batch: {str(e)}, traceback: {traceback.format_exc()}")
            if "CUDA error" in str(e):
                import sys
                sys.exit(1)
        # break
    logging.info(f"All done — {len(records)} total records in {parquet_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process images and generate captions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to the output Parquet dataset.")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16"], default="fp16", help=("Precision for loading the model. Default: fp16"))
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-448", help=("Model path to load. Default: google/paligemma-3b-mix-448"))
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model.")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size for inference. Default: 1.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    model, processor = load_pali_gemma_model(args)
    process_directory(args, args.input_dir, args.output_parquet, model, processor)

if __name__ == "__main__":
    main()