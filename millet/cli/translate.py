"""millet translate command."""
from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.argument("session_dir", type=click.Path(exists=True))
@click.option(
    "--to",
    "target_lang",
    type=str,
    default="en",
    help="Target language for translation (default: en)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Ollama model to use (default: qwen3.5:9b)",
)
def translate(session_dir, target_lang, summary_model):
    """Translate a session's transcript to another language.

    \b
    SESSION_DIR is the path to a meet recording session directory.

    The translated transcript is saved as <basename>.translation.<lang>.txt
    in the same session directory.

    \b
    Examples:
        millet translate ~/meet-recordings/meeting-20260313-231509
        millet translate ~/meet-recordings/meeting-20260313-231509 --to de
    """
    import requests as req

    session_path = Path(session_dir)
    if not session_path.is_dir():
        click.echo(f"Error: {session_path} is not a directory", err=True)
        raise SystemExit(1)

    # Find the .txt transcript file
    txt_files = sorted(session_path.glob("*.txt"))
    # Exclude any existing translation files
    txt_files = [f for f in txt_files if ".translation." not in f.name]
    if not txt_files:
        click.echo(f"Error: no .txt transcript found in {session_path}", err=True)
        raise SystemExit(1)

    txt_file = txt_files[0]
    transcript_text = txt_file.read_text(encoding="utf-8").strip()
    if not transcript_text:
        click.echo(f"Error: transcript file is empty: {txt_file}", err=True)
        raise SystemExit(1)

    basename = txt_file.stem

    from millet.summarize import DEFAULT_MODEL, OLLAMA_BASE_URL, is_ollama_available

    ollama_url = OLLAMA_BASE_URL
    model_name = summary_model or DEFAULT_MODEL

    if not is_ollama_available(ollama_url):
        click.echo("Error: Ollama is not running. Start with: ollama serve", err=True)
        raise SystemExit(1)

    from millet.languages import LANG_NAMES

    target_name = LANG_NAMES.get(target_lang, target_lang)

    click.echo(f"Translating: {txt_file}")
    click.echo(f"  Target language: {target_name}")
    click.echo(f"  Model: {model_name}")
    click.echo()

    # Free GPU memory from Ollama models that might be loaded
    from millet.transcribe import ensure_gpu_available

    ensure_gpu_available()

    import time as _time

    t0 = _time.time()

    system_content = (
        f"You are a professional translator. Translate the following "
        f"meeting transcript to {target_name}. "
        f"Preserve the exact formatting: keep the timestamp markers "
        f"like [HH:MM:SS --> HH:MM:SS] and speaker labels (YOU, REMOTE, etc.) "
        f"unchanged. Only translate the spoken text. "
        f"Be accurate and natural — do not add or remove information."
    )
    user_content = f"Translate this transcript to {target_name}:\n\n{transcript_text}"

    # Size the context window to fit the whole transcript PLUS room for the
    # translation output (roughly as long as the input).  A fixed num_ctx=8192
    # silently truncated the prompt for typical 45-105 min meetings.
    from millet.summarize import _dynamic_num_ctx, _estimate_tokens

    output_reserve = _estimate_tokens(transcript_text) + 1024
    num_ctx = _dynamic_num_ctx(
        system_content, user_content, output_reserve=output_reserve,
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = req.post(f"{ollama_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
    except req.Timeout:
        click.echo("Error: Ollama timed out. Try a smaller model.", err=True)
        raise SystemExit(1) from None
    except req.HTTPError as e:
        click.echo(f"Error: Ollama API error: {e}", err=True)
        raise SystemExit(1) from None

    elapsed = _time.time() - t0
    data = resp.json()
    translated = data.get("message", {}).get("content", "").strip()

    if not translated:
        click.echo("Error: Ollama returned an empty translation.", err=True)
        raise SystemExit(1)

    # Save translation
    out_path = session_path / f"{basename}.translation.{target_lang}.txt"
    out_path.write_text(translated, encoding="utf-8")

    click.echo(f"Translation complete in {elapsed:.1f}s")
    click.echo(f"  Saved to: {out_path}")
    click.echo()
    click.echo("--- Translation ---")
    click.echo()
    click.echo(translated)
