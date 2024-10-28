import warnings
import argparse
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

# Setting up environment
load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core._api.deprecation")

# Main code
llm = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="python")
parser.add_argument("--task", type=str, help="return a list of numbers")
args = parser.parse_args()

code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}."
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code: \n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>> GENERATED CODE:")
print(result["code"])

print(">>>>> GENERATED TEST:")
print(result["test"])
