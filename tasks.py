from crewai import Task


class SumAi():

    def summarize(self, agent, text_context: str, transcript: str, language='en', depth='superficially'):

        return Task(
            description=f'''summarize this text: '{transcript}' Return the text in the same langauage as given.
                        This context is {text_context}

                        Generate bullet point summaries that capture the essence and key details of the original text.
                        Focus on highlighting critical information such as main arguments, data points, conclusions, and important facts.

                        make a {depth} detailed summary.
                        Recognize and preserve the tone and intent of the original text in the summary.

                        Ensure that summaries are not only succinct but also accurate, avoiding the introduction of biases or errors.
                        Capable of identifying and omitting irrelevant details that do not contribute to the overall understanding of the text.

                        Present summaries in a structured bullet point format, making them easy to scan and digest.
                        Use clear and precise language to maintain readability and understanding.

                        Process and summarize texts of varying lengths and complexities quickly and efficiently.
                        Handle multiple requests simultaneously without a drop in performance or quality.
                        if any words dosent fit in the context use the ´search´ tool to find the right word that fits in the context. Do note over use the tool ´search´
                        return the summary in {language}
                        ''',
            agent=agent,
            expected_output='Short summary in point form.  Return the text in the same langauage as given'
        )

    def transcribe(self, agent, audio_file, language):
        return Task(
            description=f'''Transcribe this audio file '{
                audio_file}' to text by using the tool `transcriber`. input {language} to the tool aswell''',
            agent=agent,
            expected_output='plain text with text from the transcriber in the laguage that was given'

        )
