import os
from exa_py import Exa
from langchain.agents import tool
from dotenv import load_dotenv


class ExaSearchTool:
    @tool
    def search(query: str):
        """Search for a webpage based on the query."""
        return ExaSearchTool._exa().search(
            f"{query}", use_autoprompt=True, num_results=3
        )

    @tool
    def find_similar(url: str):
        """Search for webpages similar to a given URL.
        The url passed in should be a URL returned from `search`.
        """
        return ExaSearchTool._exa().find_similar(url, num_results=3)

    @tool
    def get_contents(ids: str):
        """Get the contents of a webpage.
        The ids must be passed in as a list, a list of ids returned from `search`.
        """

        print("ids from param:", ids)

        ids = eval(ids)
        print("eval ids:", ids)

        contents = str(ExaSearchTool._exa().get_contents(ids))
        print(contents)
        contents = contents.split("URL:")
        contents = [content[:1000] for content in contents]
        return "\n\n".join(contents)

    def tools():
        return [
            ExaSearchTool.search,
            ExaSearchTool.find_similar,
            ExaSearchTool.get_contents,
        ]

    def _exa():
        load_dotenv()
        return Exa(api_key=os.getenv('EXA_API_KEY'))


class Transcript():

    @tool
    def transcriber(audio_url, language):
        '''Transcribe audio or film to text, the spoken language must be given. Return the text in the given language'''
        import assemblyai as aai

        load_dotenv()

        aai.settings.api_key = os.getenv('AAI_KEY')

        config = aai.TranscriptionConfig(language_code=language,
                                         speech_model=aai.SpeechModel.nano,
                                         # speaker_labels=True
                                         )

        transcriber = aai.Transcriber(config=config)

        transcript = transcriber.transcribe(audio_url)

        return transcript.text

    def tools():
        return [Transcript.transcriber]


if __name__ == '__main__':
    pass
