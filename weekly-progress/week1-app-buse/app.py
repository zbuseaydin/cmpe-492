from flask import Flask, render_template, request
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatCohere(model="command-r-plus")
parser = StrOutputParser()

system_template = "Determine the language of the following text and translate it into {language}. Give the output in this format: <language_of_the_text> -- <translation>"
user_template = "Text:{text}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)
chain = prompt_template | model | parser


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def generate_response():
    language = request.form["language"]
    text = request.form["text"]

    response = chain.invoke({"language": language, "text": text})
    response = response.split("--")
    response_lang = response[0]
    response_translation = response[1]
    return render_template('index.html', text=text, language=response_lang, translation=response_translation, translated_lang=language)

if __name__ == '__main__':
    app.run(debug=True)