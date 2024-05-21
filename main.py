from dotenv import load_dotenv
from crewai import Crew
from tasks import SumAi
from agents import TranscriptSummarizeAgents
from langchain_groq import ChatGroq
import os


def main():
    load_dotenv()

    print('''
            Welcome to the sumAI-crew!
            ''')
    print()
    groq_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(model='llama3-70b-8192',
                   api_key=groq_key)

    # input('Enter file path to start: ')
    # file = '/Users/tomasrydenstam/Desktop/sarascurry.m4a'
    file = '/Users/tomasrydenstam/Desktop/sarascurry.m4a'
    # input('Enter context: ')
    context = 'casual conversation about asian food'
    language = 'sv'  # input('Enter the spoken language: ')

    # import tasks
    tasks = SumAi()

    # Import agents
    agents = TranscriptSummarizeAgents()

    # transcription-agent
    transcribe_agent = agents.TranscriptAgent(llm)

    # transcript-tasks
    transcribe_task = tasks.transcribe(transcribe_agent, file, language)

    # transcript-Crew
    crew_transcribe = Crew(
        agents=[transcribe_agent],
        tasks=[transcribe_task]
    )

    # Starts the transcription
    transcript = crew_transcribe.kickoff()

    print(transcript)
    # Summerize agent
    summarize_agent = agents.SummerizaAgent(llm=llm)

    # Summerize task
    summerize_task = tasks.summarize(
        summarize_agent, context, transcript, language)

    # sum crew
    crew_summarize = Crew(
        agents=[summarize_agent],
        tasks=[summerize_task]
    )

    summary = crew_summarize.kickoff()
    print(summary)


if __name__ == '__main__':
    main()
