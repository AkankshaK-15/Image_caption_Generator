#importing necessary libraries
import torch
from PIL import Image
from transformers import *
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai


# Setting for the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Defining the objects Pre-processor,model and tokenizer for pretrained Hugging Face model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")


# Defining the API key
openai.api_key = "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"
# Defining object for OpenAi model
openai_model = "text-davinci-002"


# Function to generate description of image using the pretrained models of Hugging Face
def generate_caption(image):
    # Creating object for an image
    img = Image.open(image)
    # Displaying the image
    st.image(img)
    # Preprocessing the image
    f_img = processor(img, return_tensors='pt').to(device)
    # Generating the encoded description of image
    output = model.generate(**f_img)
    # Decoding the description
    description = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # Giving the output
    return description

# Function to generate creative captions using OpenAi model
def generate_more_captions(des):
    # Generating the prompt
    caption_prompt = ('''generate 3 unique captions for a photo that shows'''+des+'''the captions should be creative
    captions_
    1.
    2.
    3.
    ''')

    response= openai.Completion.create(
    engine = openai_model,
    prompt = caption_prompt,
    max_tokens = (150*3),
    n = 1,
    stop = None,
    temperature = 0.7,
    )

    # generating creative captions
    caption = response.choices[0].text.strip().split("\n")
    return caption


def upload():
    with st.form("Uploader"):
        image = st.file_uploader("Upload Images")
        #  Adding the button to take the input image generating the description of image
        submit = st.form_submit_button("Generate")
        # Adding Button to generate creative captions using openai
        more_captions = st.form_submit_button("Creative Captions")

        if submit:
            st.subheader("Description of the Image_")
            # calling the function to generate the description of image
            description = generate_caption(image)
            # Showing the description output on webpage
            st.write(description)
        if more_captions:
            # Taking the description as input to create more creative captions
            description = generate_caption(image)
            # Giving the sub-header
            st.subheader("More Creative Captions for Image")
            # Creating the captions using openai model
            more_caption = generate_more_captions(description[0])
            for caption in more_caption:
                st.write(caption)

def main():
    # setting the page title
    st.set_page_config(page_title="Image Caption Generator")
    # setting the title
    st.title("Image Caption Generator")
    # setting the sub-header
    st.subheader("Get Captions for your Image")
    # calling the upload() function to generate the output
    upload()


if __name__ == '__main__':
    main()
