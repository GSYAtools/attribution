## Pilot evaluator models

Two general-purpose LLMs from different providers are used as experimental evaluators.

- OpenAI: `gpt-5.4-mini-2026-03-17`
- Google Gemini: `gemini-3.5-flash`

The OpenAI dated model identifier is used to reduce model-version drift and improve experimental reproducibility. The Gemini model is selected from the stable models available through the experimental API account.

The models are not assumed to be equivalent in capability. Model identity is retained explicitly as an experimental factor.

The purpose of using multiple model families is to assess whether observed effects of the evaluative conditions are specific to one model implementation or appear across distinct LLM families.
