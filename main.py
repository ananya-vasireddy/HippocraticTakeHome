import os
import json
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""
=====================================
Before submitting the assignment, describe here in a few sentences 
what you would have built next if you spent 2 more hours on this project:
=====================================

If I spent two more hours on this project, I would have built out a more 
comprehensive parent/adult-facing interface. Rather than simply displaying
the judge's feedback and safety notes, I could also include a short summary &
analysis of the story's moral and age-appropriateness meant for an adult audience.
Additionally, I could implement more robust safety checks (i.e. a second safety-focused
judge) to ensure the generated stories are always appropriate for children, especially
if the initial user request is potentially risky or ambiguous. Finally, I could 
incorporate multi-pass revision. Currently, if the judge determines that a story needs 
revision, the system performs a single revision pass and re-checks it once with the judge. 
However, even if the judge's score of this second, revised story is low, the system returns 
the result and does not attempt further revisions to avoid infinite looping/cost concerns.
With more time, I’d experiment with a bounded multi-pass loop (e.g., up to 2–3 revision rounds) 
that continues revising the story until it reaches the target quality score, while still 
enforcing strict limits on cost and latency. Although this case is rare, it could help ensure 
higher-quality stories in edge cases.

The block diagram is attached as a PNG!

=====================================
Hippocratic AI Bedtime Story Generator
=====================================

Features:
- Reading level selector (ages 5–6, 7–8, 9–10)
- Storyteller + LLM judge pattern
- Clear, scene-based story structure:
    - Named main character
    - Specific, concrete conflict in a scene
    - Supporting named characters
    - Dialogue
    - Explicit moral at the end
- Post-story customization menu (max 3 customizations):
    1) I'm happy with this story (exit)
    2) Add suggestions to modify the existing story
    3) Change the reading level & regenerate
    4) Change the original story request & regenerate
- Optional judge info display (for parents/adults)
"""

# OpenAI Client
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your .env file or environment."
        )
    return OpenAI(api_key=api_key)

client = get_client()

# LLM Call Wrapper
def call_chat(messages, max_tokens: int = 1500, temperature: float = 0.3) -> str:
    """
    Thin wrapper around the OpenAI chat completions API.
    """
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# Reading Level Helpers
def get_reading_level_label(level: str) -> str:
    """
    Human-readable label for each reading level.
    """
    mapping = {
        "1": "very simple language for ages 5–6, with short sentences (6–10 words) and very basic everyday vocabulary",
        "2": "simple language for ages 7–8, with mostly short sentences (8–14 words) and easy everyday vocabulary",
        "3": "clear spoken language for ages 9–10, still gentle, with mostly short sentences and no literary or complex phrasing",
    }
    return mapping.get(level, mapping["2"])


def get_spoken_style_instructions(level: str) -> str:
    """
    Extra constraints to make the language feel like an adult speaking to a child.
    """
    if level == "1":
        return (
            "- Pretend you are talking out loud to a 5-year-old.\n"
            "- Use very short, simple sentences (6–10 words).\n"
            "- Use only familiar everyday words.\n"
            "- Keep the tone warm, gentle, and playful.\n"
        )
    elif level == "2":
        return (
            "- Pretend you are talking to a 7-year-old.\n"
            "- Use short sentences (8–14 words) and simple vocabulary.\n"
            "- Avoid complex clauses.\n"
            "- Keep the tone friendly and clear.\n"
        )
    else:
        return (
            "- Pretend you are talking to a 9-year-old.\n"
            "- Use mostly short, clear spoken sentences.\n"
            "- Avoid overly poetic or complex language.\n"
            "- Keep the tone friendly and easy to follow.\n"
        )


def get_length_instructions(level: str) -> str:
    """
    Paragraph + sentence-based length instructions.
    """
    if level == "1":
        return (
            "Formatting and length:\n"
            "- Write the story in 3 to 5 paragraphs.\n"
            "- Each paragraph should have at least 3 short sentences.\n"
            "- Do NOT end the story after only 1 or 2 paragraphs.\n"
        )
    elif level == "2":
        return (
            "Formatting and length:\n"
            "- Write the story in 4 to 6 paragraphs.\n"
            "- Each paragraph should have 3–5 sentences.\n"
            "- Do NOT end the story after only 1 or 2 paragraphs.\n"
        )
    else:
        return (
            "Formatting and length:\n"
            "- Write the story in 5 to 7 paragraphs.\n"
            "- Each paragraph should have 3–6 sentences.\n"
        )


# Story Generator Agent
def generate_story(user_request: str, reading_level: str) -> str:
    """
    Generate a bedtime story given the user request and reading level.
    Enforces:
    - named main character
    - specific conflict scene
    - supporting named characters
    - dialogue
    - explicit moral
    """
    level_description = get_reading_level_label(reading_level)
    spoken_style = get_spoken_style_instructions(reading_level)
    length_instructions = get_length_instructions(reading_level)

    system_prompt = (
        "You are a warm, imaginative children's storyteller who tells bedtime "
        "stories for kids ages 5 to 10.\n"
        f"Target style: {level_description}\n"
        f"{spoken_style}"
        "Story structure (you must follow this):\n"
        "1) Beginning:\n"
        "   - Introduce the main character with a specific name.\n"
        "   - Describe one clear setting (e.g., school hallway, playground, bedroom).\n"
        "\n"
        "2) Build-up:\n"
        "   - Show the main character trying something new or facing a small challenge.\n"
        "   - Introduce 1–2 supporting characters with names.\n"
        "\n"
        "3) Specific Conflict (the climax):\n"
        "   - Include ONE concrete problem that happens in a clear, visual scene.\n"
        "   - Examples: going to the wrong classroom, losing a game,\n"
        "     getting into a conflict with another character, misunderstanding instructions.\n"
        "   - The problem must actually happen, not just be mentioned as a feeling.\n"
        "\n"
        "4) Resolution:\n"
        "   - Show how the character(s) solve the problem together.\n"
        "   - Use simple cause-and-effect (\"because she... then he...\" or \"with help from...\").\n"
        "\n"
        "5) Cozy Ending:\n"
        "   - End with a calm, sleepy moment (bedtime, peaceful thought).\n"
        "   - State the moral in ONE short, simple sentence a child can repeat.\n"
        "\n"
        f"{length_instructions}"
        "Additional content rules:\n"
        "- The story must be safe and appropriate.\n"
        "- No violence, horror, or disturbing content.\n"
        "- Gentle conflict only with a warm resolution.\n"
        "- Use concrete actions, simple dialogue, and simple sensory details.\n"
        "- Include at least two lines of dialogue with quotation marks.\n"
        "- At least two characters must have names.\n"
        "- The moral should clearly match what happens in the story.\n"
        "\n"
        "Before you begin writing, silently check:\n"
        "- Do you have a named main character?\n"
        "- Do you have a specific, concrete problem that happens in a scene?\n"
        "- Do you include dialogue in at least two places?\n"
        "- Do you have a clear, simple moral at the end?\n"
        "Only begin writing once all four conditions are satisfied.\n"
        "\n"
        "Write the story now. Separate paragraphs with blank lines.\n"
    )

    user_message = (
        f"The user requested this bedtime story:\n\"{user_request}\"\n\n"
        "Please tell the story now. Return ONLY the story text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    return call_chat(messages, max_tokens=1500, temperature=0.7)


# Judge Agent
def judge_story(user_request: str, story: str, reading_level: str) -> Dict[str, Any]:
    """
    Judge checks:
    - clear plot with specific conflict scene
    - named main character + supporting characters
    - presence of dialogue
    - explicit moral at the end
    - language simplicity and safety
    """
    level_description = get_reading_level_label(reading_level)
    spoken_style = get_spoken_style_instructions(reading_level)

    system_prompt = (
        "You are a children's book editor reviewing a bedtime story.\n"
        f"Target reading level: {level_description}\n"
        f"{spoken_style}"
        "You must evaluate these criteria:\n"
        "1) Plot and conflict:\n"
        "   - Is there a clear beginning, build-up, specific conflict scene, resolution, and happy ending?\n"
        "   - Does ONE concrete problem actually happen in a scene (not just general feelings)?\n"
        "\n"
        "2) Characters and dialogue:\n"
        "   - Is there a named main child character?\n"
        "   - Are there supporting named characters (at least one, ideally two)?\n"
        "   - Is there dialogue (quoted speech) in at least two places?\n"
        "\n"
        "3) Moral and tone:\n"
        "   - Is there a clear, short moral sentence at or near the end, and does it match the story?\n"
        "   - Is the tone gentle, calm, and bedtime-friendly?\n"
        "   - Is all content safe for ages 5–10?\n"
        "\n"
        "4) Language simplicity:\n"
        "   - Are sentences short and easy to follow for the target reading level?\n"
        "   - Are words common and child-friendly (no overly fancy or academic words)?\n"
        "\n"
        "Return ONLY JSON in this format:\n"
        "{\n"
        '  \"score\": number,  // 0-10 overall quality\n'
        '  \"needs_revision\": boolean,\n'
        '  \"feedback_for_author\": string,\n'
        '  \"safety_warnings\": string\n'
        "}\n"
        "In feedback_for_author, explicitly mention if any of the following are missing:\n"
        "- specific conflict scene,\n"
        "- named main character,\n"
        "- dialogue,\n"
        "- clear moral sentence,\n"
        "- or if the language is too complex.\n"
    )

    user_message = f"User request:\n{user_request}\n\nStory:\n{story}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    raw = call_chat(messages, max_tokens=800, temperature=0.0)

    try:
        parsed = json.loads(raw)
        parsed.setdefault("score", 0)
        parsed.setdefault("needs_revision", False)
        parsed.setdefault("feedback_for_author", "")
        parsed.setdefault("safety_warnings", "")
        return parsed
    except Exception:
        return {
            "score": 0,
            "needs_revision": True,
            "feedback_for_author": (
                "Could not parse judge response. Recommend clearer plot with a specific conflict, "
                "named characters, dialogue, simpler language, and an explicit moral."
            ),
            "safety_warnings": "Parsing failed.",
        }


# Revision Agent
def revise_story(
    user_request: str,
    original_story: str,
    judge_feedback: Dict[str, Any],
    reading_level: str,
    user_modification_instruction: Optional[str] = None,
) -> str:
    """
    Revise the story using the judge's feedback and an optional extra instruction.
    Keeps reading level constraints and structure requirements.
    """
    level_description = get_reading_level_label(reading_level)
    spoken_style = get_spoken_style_instructions(reading_level)
    length_instructions = get_length_instructions(reading_level)

    system_prompt = (
        "Revise the bedtime story using the editor's feedback and the user's suggestions.\n"
        f"Keep the reading level: {level_description}\n"
        f"{spoken_style}"
        "Required structure (do not change this):\n"
        "- Named main child character.\n"
        "- 1–2 supporting named characters.\n"
        "- One specific, concrete problem in a scene (the climax).\n"
        "- Dialogue in at least two places.\n"
        "- Cozy ending with a clear, short moral sentence.\n"
        f"{length_instructions}"
        "When revising:\n"
        "- Strengthen the plot into clear steps (beginning, build-up, conflict, resolution, cozy ending).\n"
        "- Simplify language to sound like calm bedtime talk.\n"
        "- Use short sentences and common words.\n"
        "- Maintain safe, gentle tone.\n"
        "Return ONLY the revised story text.\n"
    )

    extra_instr = user_modification_instruction or "No extra user requests."

    user_message = (
        f"Original request:\n{user_request}\n\n"
        f"Original story:\n{original_story}\n\n"
        f"Editor feedback:\n{judge_feedback.get('feedback_for_author', '')}\n\n"
        f"User refinements:\n{extra_instr}\n\n"
        "Please revise the story to satisfy all requirements."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    return call_chat(messages, max_tokens=1500, temperature=0.7)


# Generate, Judge, & (Optionally) Revise Story
def generate_story_with_judge(user_request: str, reading_level: str) -> Tuple[str, Dict[str, Any]]:
    story = generate_story(user_request, reading_level)
    verdict = judge_story(user_request, story, reading_level)

    if verdict.get("needs_revision", False):
        story = revise_story(user_request, story, verdict, reading_level)
        verdict = judge_story(user_request, story, reading_level)

    return story, verdict

# Customization Menu
def customization_menu(
    initial_request: str,
    initial_story: str,
    initial_verdict: Dict[str, Any],
    reading_level: str,
    show_judge_info: bool,
    max_customizations: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """
    Interactive loop allowing the user to tweak the story with 4 options:
      1) I'm happy with this story (exit)
      2) Add suggestions to modify the existing story
      3) Change the reading level and regenerate the story
      4) Change the original story request & regenerate the story

    - Reading level is preserved unless explicitly changed (option 3).
    - At most `max_customizations` customizations total (choices 2–4).
    """
    story = initial_story
    verdict = initial_verdict
    current_request = initial_request
    customizations_used = 0

    while True:
        remaining = max_customizations - customizations_used
        print("\nWhat would you like to do next?")
        print("1. I'm happy with this story! (exit)")
        print("2. Add suggestions to improve this story.")
        print("3. Change the reading level & regenerate.")
        print("4. Change the original story request & regenerate.")
        print(f"You have {remaining} customization(s) remaining.")

        if remaining <= 0:
            print("\nYou’ve reached the maximum number of customizations.")
            print("Keeping the current story as your final version. 🌟")
            return story, verdict

        choice = input("Enter 1–4: ").strip()

        if choice == "1":
            return story, verdict

        if choice not in {"2", "3", "4"}:
            print("Please enter a valid option (1, 2, 3, or 4).")
            continue

        # Choices 2–4 count as a customization
        customizations_used += 1

        if choice == "2":
            print(
                "\nHow would you like to change this story?\n"
                "For example: 'make it sillier', 'shorter', 'add a dragon friend', etc.\n"
            )
            mod = input("Your suggestions (or press Enter to cancel): ").strip()
            if not mod:
                print("No suggestions provided. Keeping the story as is.")
                customizations_used -= 1
                continue

            print("\nRevising your story based on your suggestions...\n")
            story = revise_story(
                current_request,
                story,
                verdict,
                reading_level,
                user_modification_instruction=mod,
            )
            verdict = judge_story(current_request, story, reading_level)

        elif choice == "3":
            print("\nOkay! Let's pick a new reading level.\n")
            reading_level = select_reading_level()
            print("\nRegenerating the story with the new reading level...\n")
            story, verdict = generate_story_with_judge(current_request, reading_level)

        elif choice == "4":
            print(
                "\nWhat would you like your new main story request to be?\n"
                "You can completely change it or build on your previous idea.\n"
            )
            new_req = input("New main request (or press Enter to cancel): ").strip()
            if not new_req:
                print("No new request provided. Keeping the previous request.")
                customizations_used -= 1
                continue

            current_request = new_req
            print("\nRegenerating the story with your new request...\n")
            story, verdict = generate_story_with_judge(current_request, reading_level)

        print("\n========== UPDATED BEDTIME STORY ==========\n")
        print(story)
        print("\n==========================================\n")

        if show_judge_info:
            print("Judge Summary:")
            print(f"- Score: {verdict.get('score')}")
            print(f"- Needs revision? {verdict.get('needs_revision')}")
            print(f"- Safety notes: {verdict.get('safety_warnings')}")
            print(f"- Feedback: {verdict.get('feedback_for_author')}")


# UI Helpers
def select_reading_level() -> str:
    print("Choose a reading level:")
    print("1. Very simple (ages 5–6)")
    print("2. Simple (ages 7–8)")
    print("3. Standard (ages 9–10)")
    while True:
        lvl = input("Enter 1, 2, or 3: ").strip()
        if lvl in {"1", "2", "3"}:
            return lvl
        print("Please enter a valid choice (1, 2, or 3).")


def ask_show_judge_info() -> bool:
    """
    Ask the user (likely a parent/adult) if they want to see judge info.
    """
    print(
        "\nWould you like to see behind-the-scenes judge ratings and safety notes?"
    )
    print("This is mostly for parents or adults. (y/n)")
    while True:
        ans = input("> ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please type 'y' or 'n'.")


# Main
def main():
    print("🌙 Welcome to the Hippocratic Bedtime Story Generator! 🌙\n")
    print("This tool creates safe, cozy bedtime stories for kids ages 5–10.\n")

    user_request = input("What kind of bedtime story would you like? ")
    reading_level = select_reading_level()
    show_judge = ask_show_judge_info()

    print("\nGenerating your story with our storyteller and judge...\n")
    story, verdict = generate_story_with_judge(user_request, reading_level)

    print("========== YOUR BEDTIME STORY ==========\n")
    print(story)
    print("\n========================================\n")

    if show_judge:
        print("Judge Summary:")
        print(f"- Score: {verdict.get('score')}")
        print(f"- Needs revision? {verdict.get('needs_revision')}")
        print(f"- Safety notes: {verdict.get('safety_warnings')}")
        print(f"- Feedback: {verdict.get('feedback_for_author')}")

    # Customization loop (max 3 tweaks)
    story, verdict = customization_menu(
        user_request,
        story,
        verdict,
        reading_level,
        show_judge,
        max_customizations=3,
    )

    print("\n🌟 Final story selected. Sweet dreams! 🌟")


if __name__ == "__main__":
    main()