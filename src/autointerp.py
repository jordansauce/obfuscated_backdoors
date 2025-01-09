import multiprocessing as mp
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List

from openai import OpenAI

# Constants
SYSTEM_PROMPT = """You are a system designed to automatically interpret internal concepts that activate within a transformer model. These concepts are detected using machine learning algorithms for a paper. You will be given examples where the concept activates highly. Words in each example where the concept activates will be surrounded by [[ ]].

Note a couple of things:
1. Find some common properties among all of the examples where the feature activates.  Dont just look at the tokens inside the [[ ]], but also the ones outside of it.
2. The tokens surrounded by [[ ]] are the ones where the feature associated with the concept activates.  Is there anything particular about how these tokens are used in the examples?
3. The next token prediction is the token that the model predicts will come next after the tokens where the concept activates.  This can give you an idea of what the model thinks is important after the concept activates.
4. The explanation at the end is a brief summary of the feature's behavior across the examples.  This can help you understand the feature's role in the model.
5. Sometimes, the last few examples will be formatted in chat format.  These are examples of how the feature might be used in a conversation.  You should note the difference in the feature activations between harmless and harmful conversations.

You must first think about the role of the feature/concept first, before writing an explanation.  List all the activating tokens, and reason about their common properties, including about whether or not the majority are harmful. Reason through the properties, while keeping in mind what I wrote above. 
However, you must conclude with: Explanation: This feature is about ______.
Keep the final explanation very brief and concise, but you must note whether or not the concept is harmful.
Absolutely make sure you write the string "Explanation: " before the final explanation.  This is how my parser will know where the explanation is."""


class GPTClient:
    def __init__(
        self,
        api_key=None,
        completion_model="gpt-4-turbo-2024-04-09",
        embedding_model="text-embedding-3-small",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set it as an environment variable or pass it to the constructor."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.completion_model = completion_model
        self.embedding_model = embedding_model

    def get_completion(self, messages):
        # Create a chat completion using the specified model
        try:
            response = self.client.chat.completions.create(
                model=self.completion_model,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred during chat completion: {e}")
            return None

    def get_completions_batch(self, message_lists, max_workers=None):
        # Process multiple message lists simultaneously for chat completions using threading
        if max_workers is None:
            max_workers = min(32, threading.active_count() * 5)  # Adjust as needed

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_messages = {
                executor.submit(self.get_completion, messages): messages
                for messages in message_lists
            }
            for future in as_completed(future_to_messages):
                results.append(future.result())

        return results

    def get_embedding(self, text):
        # Get the embedding for a given text using the OpenAI embeddings API
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"An error occurred while getting embedding: {e}")
            return []

    def get_embeddings_batch(self, texts, max_workers=None):
        # Get embeddings for multiple texts in parallel using threading
        if max_workers is None:
            max_workers = min(32, threading.active_count() * 5)  # Adjust as needed

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(self.get_embedding, text): text for text in texts
            }
            for future in as_completed(future_to_text):
                results.append(future.result())

        return results


def convert_feature_to_text(feature, extra_examples):
    # Return depicting the activations of the feature across contexts
    max_activation_examples = feature.get_max_activating(15)
    examples = [
        ex.get_tokens_feature_lists(feature)
        for ex in max_activation_examples + extra_examples
    ]

    def convert_example_to_string(tokens, values):
        opened = False
        string = ""
        for i, t in enumerate(tokens):
            if values[i] != 0 and not opened:
                string = string + " [[" + tokens[i].lstrip().rstrip()
                opened = True
            elif values[i] == 0 and opened:
                string = string.rstrip() + "]]" + tokens[i]
                opened = False
            else:
                string = string + tokens[i]
        return string.rstrip()

    example_strings = [
        f"Example {i+1}:" + convert_example_to_string(tokens[1:], values[1:])
        for i, (tokens, values) in enumerate(examples)
    ]
    return "\n".join(example_strings)


def autointerp_features(features, extra_examples=[]):
    # Return the explanations of all the features passed in
    if not isinstance(features, list):
        features = [features]
    feature_texts = {
        (feature.hook_name, feature.feature_id): convert_feature_to_text(
            feature, extra_examples
        )
        for feature in features
    }

    # Create messages for each feature
    feature_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": feature_texts[(feature.hook_name, feature.feature_id)],
            },
        ]
        for feature in features
    ]

    # Get completions for each feature
    client = GPTClient()
    batch_responses = client.get_completions_batch(feature_messages)
    batch_responses = [
        (feature, response.split("Explanation: ")[-1].strip())
        for feature, response in zip(features, batch_responses)
    ]
    if len(batch_responses) == 1:
        return batch_responses[0]
    return batch_responses


def get_text_embeddings_for_features(features):
    # Get text embeddings for the feature explanations
    if not isinstance(features, list):
        features = [features]
    feature_texts = {
        (feature.hook_name, feature.feature_id): convert_feature_to_text(feature)
        for feature in features
    }
    feature_texts = list(feature_texts.values())

    client = GPTClient()
    embeddings = client.get_embeddings_batch(feature_texts)
    return embeddings
