from openai import OpenAI
import openai
import requests
import json
from datetime import datetime
from io import BytesIO
import base64
import io
from PIL import Image

# Set your OpenAI API key
def load_api_key():
    with open("secrets.json") as f:
        secrets = json.load(f)
        return secrets.get("openai_api_key")

def generate_image(prompt):
    client = OpenAI(api_key=openai.api_key)

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

# Set your OpenAI API key
openai.api_key = load_api_key()

prompt = "An image of a dog drawn with a single continuous line without picking up the pen"

# Generate and display image
# image_url = generate_image(prompt)
image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-BHeGyeohREjILCYGYY9lf7dT/user-lIlP4QmHSmfHPDW2lRLPDVSX/img-Z9oJ54dfIeqasP60XNjqQ0Ny.png?st=2024-04-23T20%3A21%3A10Z&se=2024-04-23T22%3A21%3A10Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-04-22T23%3A52%3A03Z&ske=2024-04-23T23%3A52%3A03Z&sks=b&skv=2021-08-06&sig=VJ1NxuCVAlHAotALDleARAM2adjxrr5VWey96/ysjuc%3D"
filename = f"images/generated_image-{datetime.now().strftime('%m-%d_%H-%M-%S')}.png"
download_image(image_url, filename)
