import os
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data
from dotenv import load_dotenv

load_dotenv()

KEY=os.getenv("HUGGINGFACE_API_KEY")

model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    provider= "featherless-ai",
    max_new_tokens=1500,
    stop=["\n[1]", "\n\n["],
    repetition_penalty=1.03,
    huggingfacehub_api_token=KEY
)

llm = ChatHuggingFace(llm=model, temperature=0.3)

TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, create EXACTLY {number} multiple choice questions for {subject} students in {tone} tone.
Respond with ONLY the JSON object below, filled in — no explanations, no citations, no text before or after the JSON.
Every single question, MUST include a "correct" key indicating the right answer (a, b, c, or d). A question without a "correct" key is invalid and unacceptable.
Make sure the questions are not repeated and check all the questions for grammar and spelling mistakes and to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \

### RESPONSE_JSON
{response_json}
"""

quiz_generator_template = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template= TEMPLATE
)

quiz_chain= quiz_generator_template|llm | StrOutputParser()

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

review_chain = quiz_evaluation_prompt | llm | StrOutputParser()

generate_evaluate_chain = (
    RunnablePassthrough.assign(quiz=quiz_chain) | RunnablePassthrough.assign(review=review_chain)
)