"""millet ingest command."""
from __future__ import annotations

from pathlib import Path

import click


def _has_frontmatter(summary_meta_path: Path) -> bool:
    """Return True if the session's existing summary already carries
    structured frontmatter (i.e. was produced by meetscribe >= 0.7.0 /
    millet-pipeline >= 0.9.0)."""
    if not summary_meta_path.exists():
        return False
    try:
        import json as _json

        meta = _json.loads(summary_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(meta.get("data_extracted"))


def _ingest_one_session(
    session_dir: Path,
    *,
    summary_preset: str | None = None,
    summary_backend: str | None,
    summary_model: str | None,
    ollama_singlepass: bool,
    re_pdf: bool,
    force: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    """Re-extract structured frontmatter for a single session.

    Returns (ok, message) where ``ok`` is True on success or skip and
    False on failure.
    """
    from millet.frontmatter import context_from_transcript
    from millet.label import _find_session_files, _load_transcript
    from millet.summarize import (
        SummaryConfig,
    )
    from millet.summarize import (
        summarize as do_summarize,
    )

    files = _find_session_files(session_dir)
    if "json" not in files:
        return False, f"{session_dir.name}: no transcript JSON"

    basename = files["json"].stem
    meta_path = session_dir / f"{basename}.summary.meta.json"
    if not force and _has_frontmatter(meta_path):
        return True, f"{session_dir.name}: already has frontmatter (skip)"

    if dry_run:
        return True, f"{session_dir.name}: would re-extract"

    transcript = _load_transcript(files["json"])

    cfg_kwargs: dict = {}
    if summary_preset:
        cfg_kwargs["preset"] = summary_preset
    if summary_backend:
        cfg_kwargs["backend"] = summary_backend
    if summary_model:
        cfg_kwargs["model"] = summary_model
    if ollama_singlepass:
        cfg_kwargs["ollama_singlepass"] = True
    summary_config = SummaryConfig(**cfg_kwargs)

    def _progress(msg: str) -> None:
        click.echo(f"    {msg}")

    try:
        result = do_summarize(
            transcript.to_text(),
            summary_config,
            language=transcript.language,
            progress_callback=_progress,
        )
    except Exception as exc:
        return False, f"{session_dir.name}: summary failed: {exc}"

    fm_ctx = context_from_transcript(transcript, session_dir)
    try:
        result.save(session_dir, basename, frontmatter_context=fm_ctx)
    except Exception as exc:
        return False, f"{session_dir.name}: save failed: {exc}"

    if re_pdf:
        try:
            from millet.pdf import generate_pdf

            pdf_path = session_dir / f"{basename}.pdf"
            generate_pdf(
                transcript,
                pdf_path,
                summary=result,
                language=getattr(transcript, "language", "en"),
            )
        except Exception as exc:
            # PDF regen is best-effort; the frontmatter is still written.
            return True, (
                f"{session_dir.name}: frontmatter written, PDF regen failed: {exc}"
            )

    note = ""
    if result.data_error:
        note = f" (data_error: {result.data_error})"
    elif result.data is None:
        note = " (no JSON block emitted)"
    return True, f"{session_dir.name}: ok in {result.elapsed_seconds:.1f}s{note}"


@click.command()
@click.argument(
    "session_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--re-llm/--no-re-llm",
    default=True,
    help="Re-run the LLM to extract structured frontmatter from transcripts. "
    "Currently the only supported mode (regex-only parse is intentionally not "
    "implemented); --no-re-llm is reserved and currently rejected.",
)
@click.option(
    "--summary-preset",
    type=click.Choice(["high-quality", "confidential", "alternative"], case_sensitive=False),
    default=None,
    help="DEPRECATED (removed in 0.21.0): all presets now resolve to the default private backend. Kept so existing scripts keep working.",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["tinfoil", "ollama", "bedrock"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: tinfoil, or MILLET_SUMMARY_BACKEND env var). bedrock sends the transcript to AWS.",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MILLET_SUMMARY_MODEL env var)",
)
@click.option(
    "--ollama-singlepass",
    is_flag=True,
    default=False,
    help="Use the legacy single-pass Ollama flow instead of the default two-pass.",
)
@click.option(
    "--no-pdf",
    is_flag=True,
    default=False,
    help="Skip regenerating the PDF after writing frontmatter.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-extract even when the session already has structured frontmatter.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List sessions that would be processed without invoking the LLM.",
)
def ingest(
    session_dirs,
    re_llm,
    summary_preset,
    summary_backend,
    summary_model,
    ollama_singlepass,
    no_pdf,
    force,
    dry_run,
):
    """Re-extract structured YAML frontmatter for existing sessions.

    \b
    SESSION_DIRS one or more meeting session directories. Each directory
    must contain a transcript JSON file produced by `millet transcribe`.

    For every session, the transcript is re-summarized with the configured
    backend so that the LLM emits the schema_version 1 JSON data block.
    The new frontmatter is written into <basename>.summary.md and a
    matching <basename>.frontmatter.json sidecar is produced. The PDF
    is regenerated by default to reflect the cleaned summary body.

    Sessions whose .summary.meta.json already records data_extracted=true
    are skipped unless --force is passed.

    \b
    Examples:
        millet ingest ~/meet-recordings/meeting-2026*
        millet ingest ./session-dir --dry-run
        millet ingest ./session-dir --force --no-pdf
    """
    # --re-llm is currently the only mode; the flag is reserved for a
    # future regex-only parser. Reject the contradictory case explicitly.
    if not re_llm:
        click.echo(
            "Error: --no-re-llm is not supported. The regex-only parse "
            "path was intentionally deferred; pass --re-llm (default) to "
            "re-run the summarizer.",
            err=True,
        )
        raise SystemExit(2)

    # Free GPU memory before kicking off summaries (matches behavior of
    # `meet transcribe`) — only affects the ollama backend.
    try:
        from millet.transcribe import ensure_gpu_available

        if (summary_backend or "").lower() in ("", "ollama"):
            ensure_gpu_available()
    except Exception:
        pass

    targets = [Path(d) for d in session_dirs]
    click.echo(f"Ingesting {len(targets)} session(s)...")
    if dry_run:
        click.echo("(dry run — no LLM calls will be made)")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for sd in targets:
        click.echo(f"  • {sd}")
        ok, msg = _ingest_one_session(
            sd,
            summary_preset=summary_preset,
            summary_backend=summary_backend,
            summary_model=summary_model,
            ollama_singlepass=ollama_singlepass,
            re_pdf=not no_pdf,
            force=force,
            dry_run=dry_run,
        )
        if ok:
            if "skip" in msg:
                n_skip += 1
            else:
                n_ok += 1
            click.echo(f"    {msg}")
        else:
            n_fail += 1
            click.echo(click.style(f"    {msg}", fg="red"), err=True)

    click.echo()
    click.echo(
        f"Done: {n_ok} processed, {n_skip} skipped, {n_fail} failed."
    )
    if n_fail:
        raise SystemExit(1)
