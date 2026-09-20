# jev-rerank

## Task

The current project is a TypeSafe's Jev-based reranker tool.

We are aiming at tuning this tool to be more general-purpose:

- Good sensible defaults for ranking texts, be it either coming from SQLite database table fields or plain text documents.
- The `--query` arg should be improved in terms of the values it can accept:
    - Add yes/no extensions, that would replace Noul criteria, like e.g.
    ```
        -- query 'a song about missing love | yes: the song talks about missing romantic love | no: the song talks about something non-romantic, whatever it is like e.g. love for books or music'
    ```
    - Same thing but for inline json, like e.g.
    ```
        -- query '{"query": "a song about missing love", "yes": "the song talks about missing romantic love", "no": "the song talks about something non-romantic, whatever it is like e.g. love for books or music"}'
    ```
    - For both of them, admit multi-line texts.
        - for the "|"-separators one:
            - consider the "|" as the separator char, no spaces around required, 
            - same goes "yes:" and "no:", consider them as mandatory prefixes tagging the text that follows. 
        - for both of them: if either "yes" or "no" value is present, then the other "no", "yes" must also be present.

## Golden Rules (MANDATORY)

- **Do Not Reinvent the Wheel!:** install useful python packages if that would save work and/or make better code.
- **KISS:** do not overcomplicate things!
- **Simple Code:** so it will be easy to **understand**.
- **Familiar Code:** so it will be easy to **learn**.
- **Idiomatic Code:** so it will show **how it is done**.
- **Annotated for other agents to follow:** add docstrings and comments for other agents to read and understand the code.
- **TDD:** follow Test Driven Development good practices, create high-quality, well scoped tests.
- **Use** the `typesafe-ai` skill for this task.
- **Index** for TypeSafe-related information: `https://docs.typesafe.ai/llms.txt`
- Consider **querying TypeSafe API** in order to **improve** your **decision-making** process.
