from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@csrf_exempt
def my_playground(request):
    if request.method == "POST":
        try:
            # Parse the JSON data from the request body
            data = json.loads(request.body)
            
            # Extract parameters from the request data
            model_name = data.get('model', 'mixtral-8x7b-32768')  # Default to mixtral if not specified
            temperature = float(data.get('temperature', 0.5))
            max_tokens = int(data.get('max_tokens', 8064))
            stream = data.get('stream') == 'on'
            user_input = data.get('user_input')
            system_prompt = data.get('system_prompt', '')

            # Validate input
            if not user_input:
                return JsonResponse({"error": "User input is required"}, status=400)

            # Get the API key from environment variables
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment variables")
                return JsonResponse({"error": "API key not configured"}, status=500)

            # Initialize the Groq model
            try:
                llm = ChatGroq(
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            except Exception as e:
                logger.error(f"Error initializing Groq model: {str(e)}")
                return JsonResponse({"error": "Failed to initialize AI model"}, status=500)

            # Set up the chat prompt template
            prompt = ChatPromptTemplate([
                ("system", "{system_prompt}"),
                ("human", "{user_input}")
            ])

            # Create the chain and invoke it
            try:
                chain = prompt | llm
                response = chain.invoke({
                    "user_input": user_input,
                    "system_prompt": system_prompt
                })
                return JsonResponse({"response": response.content})
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return JsonResponse({"error": "Failed to generate response"}, status=500)

        except json.JSONDecodeError:
            logger.error("Invalid JSON in request body")
            return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({"error": "An unexpected error occurred"}, status=500)
    else:
        # For GET requests, render the playground template
        return render(request, 'playground.html')
