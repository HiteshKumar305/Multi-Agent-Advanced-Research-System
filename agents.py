
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    max_retries=2,
    timeout=60,
)

# WRITER AGENT
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a careful research writer.
        Produce a structured, evidence-aware research report.
        Distinguish sourced facts from interpretation.
        Do not invent statistics, quotations, dates, or URLs.
        If evidence is insufficient, clearly mention it."""
    ),
    (
        "human",
        """Write a detailed research report on this topic:

Topic: {topic}

Research Material:
{research}

Use the following structure:

# Research Report: [Topic]

## 1. Introduction
Introduce the research topic and explain its importance.

## 2. Key Findings
Explain at least three important findings based on the available evidence.

## 3. Detailed Analysis
Analyse the findings and discuss their implications.

## 4. Limitations / Evidence Gaps
Identify missing information and limitations in the available sources.

## 5. Conclusion
Summarise the major findings and provide a concluding perspective.

## 6. Sources
List the source URLs supplied in the research material.

Use clear, professional language.
Do not claim that a source supports a statement unless its supplied content
actually supports it."""
    ),
])

# CRITIC AGENT
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict and constructive research quality reviewer.
        Evaluate accuracy, evidence, relevance, structure, completeness,
        and unsupported claims."""
    ),
    (
        "human",
        """Review the research report below.

Report:
{report}

Return this exact structure:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

Evidence and Citation Check:
- ...

One-line Verdict:
..."""
    ),
])

# Create LangChain chains
writer_chain = writer_prompt | llm | StrOutputParser()

critic_chain = critic_prompt | llm | StrOutputParser()
