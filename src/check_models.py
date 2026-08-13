from openai import OpenAI
from google import genai

from src.utils import get_env


def check_openai() -> None:
    print("\n=== OPENAI ===")

    client = OpenAI(
        api_key=get_env("OPENAI_API_KEY")
    )

    models = client.models.list()

    model_ids = sorted(
        model.id
        for model in models.data
    )

    for model_id in model_ids:
        if "gpt-5" in model_id.lower():
            print(model_id)


def check_gemini() -> None:
    print("\n=== GEMINI ===")

    client = genai.Client(
        api_key=get_env("GEMINI_API_KEY")
    )

    found = []

    for model in client.models.list():
        name = getattr(model, "name", "")

        if "gemini-3" in name.lower():
            found.append(name)

    for name in sorted(found):
        print(name)


def main() -> None:
    check_openai()
    check_gemini()


if __name__ == "__main__":
    main()
