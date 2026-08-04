TABLE_PROMPT = (
    'You are given a table in Markdown from a university handbook. Write ONE short '
    'sentence, in the same language as the table, summarising what the table contains. '
    'Return only the sentence, no preamble.'
)

VISION_PROMPT = (
    'This image is from a university handbook. Respond in the language of the image. '
    'First, describe in 1-2 sentences what the image shows (including any chart or '
    'diagram and its meaning). Then, on new lines, transcribe verbatim all text visible '
    'in the image. If there is no text, omit the transcription.'
)
