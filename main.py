import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(os.environ.get("OPENAI_API_KEY"))
    information = """Bhimrao Ramji Ambedkar[a] (14 April 1891 – 6 December 1956) was an Indian jurist, economist, social reformer and politician who chaired the committee that drafted the Constitution of India based on the debates of the Constituent Assembly of India and the first draft of Sir Benegal Narsing Rau.[1][2][3][4][5] Ambedkar served as Law and Justice minister in the first cabinet of Jawaharlal Nehru. He later renounced Hinduism and converted to Buddhism, inspiring the Dalit Buddhist movement.[6]
                     After graduating from Elphinstone College, University of Bombay, Ambedkar studied economics at Columbia University and the London School of Economics, receiving doctorates in 1927 and 1923, respectively, and was among a handful of Indian students to have done so at either institution in the 1920s. During his time at Columbia University, Ambedkar came under the influence of John Dewey and his philosophy of pragmatism.[7][8][9] He also trained in the law at Gray's Inn, London. In his early career, he was an economist, professor, and lawyer. His later life was marked by his political activities; he became involved in campaigning and negotiations for partition, publishing journals, advocating political rights and social freedom for Dalits, and contributing to the establishment of the state of India. In 1956, he converted to Buddhism, initiating mass conversions of Dalits.[10]
                     In 1990, the Bharat Ratna, India's highest civilian award, was posthumously conferred on Ambedkar. The salutation Jai Bhim (lit. "Hail Bhim") is used by his followers to honour him. He is also referred to by the honorific Babasaheb, meaning "Respected Father"."""
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template= summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information })
    print(response.content)

if __name__ == "__main__":
    main()
