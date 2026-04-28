import click
import json
import os
import sys
import typing as t

from gbcli.utils.utils import custom_parse_markdown_str, parse_markdown_str


class FileOrStringParamType(click.ParamType):
    name = "fileOrString"

    def convert(self, value, param, ctx):
        try:
            if value == "-":
                value = t.cast("os.PathLike[str]", value)

                return sys.stdin.readline(1024).rstrip("\n")

            elif isinstance(value, str):
                return value

        except ValueError:
            self.fail(f"{value} must be a single string.", param, ctx)


def validation_formatting(
    callback_args,
    verbose_validation,
    quiet=False,
    format="simple",
    json_to_stderr=False,
):
    build_path = callback_args.get("build_path", "")
    validations = callback_args.get("validations", [])
    reformatted_original_yaml = callback_args.get("reformatted_original_yaml", "")
    number_errors = 0
    number_warnings = 0
    error_warning_text = ""
    for validation in validations:
        if validation.get("warning"):
            number_warnings += 1
            validation["status_display_text"] = (
                f"⚠️  WARNING #{number_warnings} ({validation.get('type', '')}): {validation.get('summary', '')}"
            )
        if validation.get("error"):
            number_errors += 1
            validation["status_display_text"] = (
                f"❌ ERROR #{number_errors} ({validation.get('type', '')}): {validation.get('summary', '')}"
            )

        error_warning_text += f"{validation.get('status_display_text', '')}\n"

    if format == "json":
        errors = [
            {
                "type": v.get("type", ""),
                "summary": v.get("summary", ""),
                "detail": v.get("detail", ""),
                "solution": v.get("solution", ""),
            }
            for v in validations
            if v.get("error")
        ]
        warnings = [
            {
                "type": v.get("type", ""),
                "summary": v.get("summary", ""),
                "detail": v.get("detail", ""),
                "solution": v.get("solution", ""),
            }
            for v in validations
            if v.get("warning")
        ]
        click.echo(
            json.dumps(
                {
                    "validated": number_errors == 0,
                    "build_path": build_path,
                    "errors": errors,
                    "warnings": warnings,
                }
            ),
            err=json_to_stderr,
        )
        return

    option_text = (
        "Use '--verbose-validation` to see more details."
        if not verbose_validation
        else ""
    )
    if number_errors > 0:
        click.echo(
            f"\n❌ Build validation failed with {number_errors} errors and {number_warnings} warnings for build definition '{build_path}'. {option_text}",
            err=True,
        )
    elif number_warnings > 0:
        if not quiet:
            click.echo(
                f"\n⚠️ Build validation has {number_warnings} warnings for build definition '{build_path}'. Use '--verbose-validation` to see more details.",
                err=True,
            )
    if error_warning_text and len(error_warning_text) > 0:
        if not quiet:
            click.echo(error_warning_text, err=True)
    if verbose_validation and not quiet:
        click.echo("Validation Details:", err=True)
        for validation in validations:
            detail = validation.get("detail")
            solution = validation.get("solution")
            status_display_text = validation.get("status_display_text")
            updated_yaml = validation.get("updated_yaml")

            if status_display_text and len(status_display_text) > 0:
                click.echo(custom_parse_markdown_str("---"), err=True)
                click.echo(status_display_text, err=True)
                if detail and len(detail) > 0:
                    click.echo(parse_markdown_str(f"```\n{detail}\n```"), err=True)
                if updated_yaml:
                    click.echo("Build definition before:", err=True)
                    click.echo(
                        parse_markdown_str(f"```\n{reformatted_original_yaml}\n ```"),
                        err=True,
                    )
                    click.echo(
                        "Build definition after the suggested correction:", err=True
                    )
                    click.echo(
                        parse_markdown_str(f"```\n{updated_yaml}\n```"), err=True
                    )
                else:
                    if solution and len(solution) > 0:
                        click.echo(parse_markdown_str("***Solution:***"), err=True)
                        click.echo(
                            parse_markdown_str(f"```\n{solution}\n```"), err=True
                        )
