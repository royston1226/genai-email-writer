import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile",
                            temperature=0,
                            groq_api_key = os.getenv("GROQ_API_KEY"))


    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing only the
            following keys: `role`, `experience`, `skills` and `description`.
            Only please return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm

        res = chain_extract.invoke(input={'page_data': cleaned_text})

        try:
            json_parser = JsonOutputParser()

            json_res = json_parser.parse(res.content)

        except OutputParserException:
            raise OutputParserException("Output too big")

        return json_res if isinstance(json_res, list) else [json_res]


    def write_mail(self, job):

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Royston, a university student studying Computer Science and Data Science at the University of Wisconsin–Madison. 
            You are reaching out to this company to express your interest in working with them, offering your skills in software development, AI/ML, or backend engineering.

            Your job is to write a cold email to the client based on the job description above. 
            The email should clearly express your interest, highlight your technical background, and communicate how you can contribute value to their team. 
            Also add any soft skills that may be required or help with the job

            Keep the tone polite, professional, and concise.
            Do not include a preamble or system message.

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm

        res = chain_email.invoke({"job_description": str(job)})

        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))