from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

template = """You are a helpful assistant. Given a topic, generate a comma separated list of 5 key points about that topic.
Topic: {topic}
List:""" 

prompt = ChatPromptTemplate.from_template(template)

chain = LLMChain(
    llm=ChatOpenAI(), 
    prompt=prompt,
    output_parser=CommaSeparatedListOutputParser()
)

topic = input("What should I explain?")
response = chain.invoke(topic)
print(response)
