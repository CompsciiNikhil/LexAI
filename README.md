# LexAI - Agentic Legal Document Intelligence

> Understand any legal document in plain English. Built with Google ADK, MCP, and Gemini.

## What it does
LexAI is a multi-agent AI system that:
- Explains legal documents in plain English
- Flags risky clauses with severity ratings (HIGH / MEDIUM / LOW)
- Answers your questions about the document
- Suggests concrete negotiation rewrites for HIGH / MEDIUM risk clauses

## Built for
- Kaggle AI Agents Capstone (Agents for Business track)
- Google GenAI Exchange Hackathon (Legal Documents track)

## Tech Stack
- Google Agent Development Kit (ADK)
- Gemini 2.0 Flash
- Custom MCP Server (document tools)
- Google Cloud Run
- Python 3.11

## Setup
```bash
git clone <repo-url>
cd lexai
cp .env.example .env  # fill in your API keys
pip install -r requirements.txt
python main.py
```

## Architecture
```
User (Browser)
      |
[Orchestrator Agent]
            |
    ┌───────┼────────────┬──────────────────┐
    |       |            |                  |
[Explainer] [Risk] [Q&A]   [Negotiation Suggester]
      |                       (also via /negotiate)
 [MCP Server]
 - parse_document
 - extract_clauses
 - search_clause
      |
 [PDF / Text Input]
```

## Disclaimer
LexAI provides legal clarity, not legal advice. Always consult a qualified lawyer for important decisions.
