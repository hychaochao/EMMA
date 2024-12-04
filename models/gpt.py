import logging
import re
import base64
from io import BytesIO

from openai import OpenAI


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def create_message(sample):
    query = sample['query']
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    for i, fragment in enumerate(split_text):
        all_contents.extend([
            {"type": "text", "text": fragment}
        ])
        if i < len(matches):
            if sample[matches[i]]:
                img_base64 = encode_image_to_base64(sample[matches[i]])
                all_contents.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ])
            else:
                logging.error(
                    f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")

    messages = [
        {
            "role": "system",
            "content": f"You are a {sample['subject']} expert."
        },
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages


# build gpt class
class GPT_Model:
    def __init__(
            self,
            client: OpenAI,
            model="chatgpt-4o-latest",
            temperature=0,
            max_tokens=1024
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, sample):

        messages = create_message(sample)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return None
