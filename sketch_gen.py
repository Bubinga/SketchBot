from openai import OpenAI
import requests
import json
from datetime import datetime
from io import BytesIO
from PIL import Image

# Set your OpenAI API key
def load_api_key():
    with open("secrets.json") as f:
        secrets = json.load(f)
        return secrets.get("openai_api_key")

def generate_image_url(prompt):
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    image_url = response.data[0].url
    print("Image URL:")
    print(image_url)
    return image_url

def download_image(url, filename):
    try:
        # Send a GET request to the URL to download the image
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for invalid responses

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))
        image.save(filename)
        print("Image saved successfully as", filename)

    except Exception as e:
        print("Error:", e)

def generate_image(prompt, filename = "images/newest-image.png"):
    # Generate and display image
    image_url = generate_image_url(prompt)
    if filename is None:
        filename = f"images/generated_image-{datetime.now().strftime('%m-%d_%H-%M-%S')}.png"
    download_image(image_url, filename)
    return image_url

def main():
    # Example prompt
    prompt = "A detailed image of a dog drawn with a single continuous line without picking up the pen"
    generate_image(prompt)

if __name__ == "__main__":
    main()