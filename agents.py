from crewai import Agent
from tools import ExaSearchTool, Transcript


class TranscriptSummarizeAgents():
    def SummerizaAgent(self, llm):
        return Agent(

            role='Summerize text into a if possible short summary',
            goal='Identify the context of the conversation and summerize it short without missing out important things.',
            backstory='''As a summerizer you hava the ability to understand and interpret text from a wide range of domains including technical documents, literary works, academic papers, news articles, and more.
                                    Capable of discerning key points and themes even in complex and nuanced texts.

                                    Generate bullet point summaries that capture the essence and key details of the original text.
                                    Focus on highlighting critical information such as main arguments, data points, conclusions, and important facts.

                                    Adjust the depth and detail of summaries based on the users needs or specified requirements.
                                    Recognize and preserve the tone and intent of the original text in the summary.

                                    Ensure that summaries are not only succinct but also accurate, avoiding the introduction of biases or errors.
                                    Capable of identifying and omitting irrelevant details that do not contribute to the overall understanding of the text.

                                    Present summaries in a structured bullet point format, making them easy to scan and digest.
                                    Use clear and precise language to maintain readability and understanding.

                                    Process and summarize texts of varying lengths and complexities quickly and efficiently.
                                    Handle multiple requests simultaneously without a drop in performance or quality.''',
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            tools=ExaSearchTool.tools()
        )

    def TranscriptAgent(self, llm):
        return Agent(
            role='Transcript video or audio to text from diffrent contexts.',
            goal='Make a good transcription',
            backstory='''As a transcriber your expertise is converting audio to text .
                      If you dont recognize a word you use your imagination based on the context and find out what the right word would be''',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=Transcript.tools()

        )
