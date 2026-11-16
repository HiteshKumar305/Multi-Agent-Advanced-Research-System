
import re
import time

from agents import writer_chain, critic_chain
from tools import web_search, scrape_url


def update_progress(callback, step, message):
    print(f"{step.upper()}: {message}")

    if callback:
        callback(step, message)


def invoke_with_retry(chain, inputs, stage, callback=None, attempts=3):
    """Retry temporary Gemini/API failures with exponential backoff."""

    for attempt in range(1, attempts + 1):
        try:
            return chain.invoke(inputs)

        except Exception as exc:
            error_text = str(exc).lower()

            temporary_errors = [
                "503", "429", "500", "502", "504",
                "unavailable", "resource_exhausted",
                "deadline_exceeded", "timeout",
                "high demand", "overloaded"
            ]

            is_temporary = any(
                error in error_text for error in temporary_errors
            )

            if attempt == attempts or not is_temporary:
                raise

            wait_seconds = 2 ** attempt

            update_progress(
                callback,
                stage,
                f"Temporary API failure. Retrying in {wait_seconds} seconds "
                f"({attempt}/{attempts - 1})..."
            )

            time.sleep(wait_seconds)


def create_fallback_report(topic, search_results, scraped_content, sources):
    """Create a transparent source-based report without an LLM."""

    source_list = "\n".join(
        f"- {url}" for url in sources
    ) or "- No source URLs were successfully extracted."

    return f"""# Research Report: {topic}

> **Fallback mode:** Gemini was unavailable during this execution.
> This report is compiled from the live search results and extracted webpage
> content. It is not an AI-generated analysis.

## 1. Introduction

This report presents the available web research material on **{topic}**.
The findings below are based on the search results and webpage content
retrieved during this execution.

## 2. Web Search Findings

{search_results[:10000] if search_results else "No search results available."}

## 3. Extracted Source Content

{scraped_content[:12000] if scraped_content else "No readable webpage content was extracted."}

## 4. Limitations / Evidence Gaps

- Gemini-based analysis was unavailable during this execution.
- The material has not been independently fact-checked.
- Search snippets and extracted webpage text may be incomplete.
- No unsupported statistics or conclusions have been added.

## 5. Conclusion

The retrieved sources provide preliminary material for further research
on **{topic}**. A full analytical report should be generated after the
AI service becomes available and the source claims have been verified.

## 6. Sources

{source_list}
"""


def create_fallback_feedback(report):
    return """## Critic Review — Fallback Mode

**Status:** Automated AI review unavailable.

**Strengths:**
- The report identifies the research topic.
- Retrieved search results and source material are included.
- Limitations are explicitly disclosed.

**Areas to Improve:**
- Generate a full analytical report using the AI service.
- Verify important claims against the original sources.
- Add comparative analysis and evidence-backed conclusions.

**Evidence and Citation Check:**
- Source URLs are listed where available.
- The retrieved material has not been independently verified.

**One-line Verdict:**
Preliminary research output is available, but it requires full AI analysis
and human verification.
"""


def run_research_pipeline(topic, progress_callback=None):

    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    topic = topic.strip()

    state = {
        "topic": topic,
        "sources": [],
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "execution_mode": "LIVE_AI",
    }

    # STEP 1: SEARCH
    update_progress(
        progress_callback,
        "search",
        "Searching the web using Tavily..."
    )

    search_results = web_search.invoke({"query": topic})
    state["search_results"] = search_results

    # STEP 2: READ / SCRAPE
    update_progress(
        progress_callback,
        "reader",
        "Extracting URLs and reading webpages..."
    )

    urls = re.findall(r"https?://[^\s]+", search_results)

    urls = list(dict.fromkeys(
        url.rstrip(".,);]") for url in urls
    ))[:3]

    scraped_pages = []

    for url in urls:
        content = scrape_url.invoke({"url": url})

        if content and not content.startswith("Scraping failed"):
            scraped_pages.append(content)

    state["sources"] = urls
    state["scraped_content"] = "\n\n".join(scraped_pages)

    research = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    # STEP 3: WRITER
    update_progress(
        progress_callback,
        "writer",
        "Generating the research report using Gemini..."
    )

    try:
        state["report"] = invoke_with_retry(
            writer_chain,
            {
                "topic": topic,
                "research": research,
            },
            "writer",
            progress_callback,
        )

    except Exception as exc:
        state["execution_mode"] = "FALLBACK"

        update_progress(
            progress_callback,
            "writer",
            f"Gemini unavailable. Using source-based fallback. Details: {exc}"
        )

        state["report"] = create_fallback_report(
            topic,
            state["search_results"],
            state["scraped_content"],
            state["sources"],
        )

    # STEP 4: CRITIC
    update_progress(
        progress_callback,
        "critic",
        "Reviewing the research report..."
    )

    if state["execution_mode"] == "LIVE_AI":

        try:
            state["feedback"] = invoke_with_retry(
                critic_chain,
                {"report": state["report"]},
                "critic",
                progress_callback,
            )

        except Exception as exc:
            state["execution_mode"] = "PARTIAL_FALLBACK"

            update_progress(
                progress_callback,
                "critic",
                f"AI review unavailable. Using fallback review. Details: {exc}"
            )

            state["feedback"] = create_fallback_feedback(
                state["report"]
            )

    else:
        state["feedback"] = create_fallback_feedback(
            state["report"]
        )

    update_progress(
        progress_callback,
        "complete",
        f"Research completed. Mode: {state['execution_mode']}"
    )

    return state


if __name__ == "__main__":

    research_topic = input("Enter a research topic: ").strip()

    try:
        output = run_research_pipeline(research_topic)

        print("\n========== EXECUTION MODE ==========\n")
        print(output["execution_mode"])

        print("\n========== FINAL REPORT ==========\n")
        print(output["report"])

        print("\n========== CRITIC FEEDBACK ==========\n")
        print(output["feedback"])

    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
