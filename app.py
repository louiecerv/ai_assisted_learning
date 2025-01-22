import streamlit as st
from openai import OpenAI
import streamlit as st
import os

api_key = os.getenv("NVIDIA_API_KEY")
MODEL_ID = "meta/llama-3.1-405b-instruct"

# Check if the API key is found
if api_key is None:
    st.error("NVIDIA_API_KEY environment variable not found.")
else:
    # Initialize the OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )


def get_ai_response(prompt):
    """Generates a response from an AI model

    Args:
    prompt: The prompt to send to the AI model.

    Returns:
    response from the AI model.
    """
    try:
        # Access conversation_history from session state
        messages = [
            {
                "role": "system",
                "content": "You are a programming assistant focused on providing \
                accurate, clear, and concise answers to technical questions. \
                Your goal is to help users solve programming problems efficiently, \
                explain concepts clearly, and provide examples when appropriate. \
                Use a professional yet approachable tone. Use explicit markdown \
                format for code for all codes in the output."
            }
        ]

        messages.append({
            "role": "user",
            "content": prompt
        })

        completion = client.chat.completions.create(
            model=MODEL_ID,
            temperature=0.5,  # Adjust temperature for creativity
            top_p=1,
            max_tokens=1024,
            messages=messages,
            stream=False
        )

        model_response = completion.choices[0].message.content
        return model_response
        
    except Exception as e:
        st.error(f"Error handling AI response: {e}")
        return None

def main():
    # Streamlit App
    st.title("Create an AI App using the Nvidia AI Model")

    # Step 1: Platform selection
    platform = st.selectbox(
        "Choose the platform:",
        ("Streamlit", "Gradio")
    )

    # Step 2: Task selection
    task = st.selectbox(
        "Select a task:",
        ("Get NVIDIA API Key", 
         "Code the Program on the select platform",
         "Deploy and test the App")
    )

    detailed_task = ""

    if task == "Get NVIDIA API Key":
        detailed_task = f"""
Search the web for information how to obtain an API key from Nvidia NGC and 
give detailed instruction on to how to setup a huggingface space to host 
a {platform} app that uses the Nvidia API. """
        
    elif task == "Code the Program on the select platform":
        detailed_task = f"""
Create a {platform} app that gives the user an intuitive interface using a text 
area to prompt the user for an input prompt.  Provide a button that send the 
input to the AI model.  The app displays the response to the page 
Give me the full python code for this app."""
        
    elif task == "Deploy and test the App":
        detailed_task = f"""
Give detailed instruction on how to deploy and test a {platform} app on hugging face.
        """

    # Display the generated prompt
    st.write("Generated AI Prompt:", detailed_task)

    prompt = detailed_task

    # Step 4: Get AI Response button
    if st.button("Get AI Response"):
        with st.spinner("Thinking..."):
            # Get the AI response (dummy function)
            response = get_ai_response(prompt)
            # Display the AI response
            st.write("AI Response:", response)

if __name__ == "__main__":
    main()