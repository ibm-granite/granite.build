import json
import os
import random
import re
import string
import time
import unicodedata
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import humanize
import yaml
from git import RemoteProgress
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import (
    ConsoleOptions,
    Heading,
    HorizontalRule,
    ListItem,
    Markdown,
    Paragraph,
    RenderResult,
    Rule,
    Segment,
    Style,
    Table,
    TableElement,
    Text,
    box,
    loop_first,
)
from rich.panel import Panel
from tqdm import tqdm

from gbcli.utils.cli_config import get_local_build_cache
from gbcli.utils.gbconstants import (
    BUILD_LOG_DEFAULT_QUERY_RANGE,
    DMF_URL,
    GBSERVER_ARTIFACT_API,
    SPACE_REPO_NAME,
    SPACE_REPO_ORG,
    gb_environment_config,
)
from gbcli.utils.gbserver import get_artifacts, make_gbserver_call
from gbcli.utils.gh_auth import get_user
from gbcli.utils.spaceutil import resolve_space
from gbcommon.uri.lh import LhURI
from gbcommon.uri.uri import URI


def normalize_to_filename(value: str, allow_unicode: bool = False) -> str:
    """
    https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    assert isinstance(value, str)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]*", "-", value).strip("-_")


class CustomHeading(Heading):
    """Custom heading."""

    def __init__(self, tag: str) -> None:
        super().__init__(tag)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text
        text.justify = "center"
        if self.tag == "h1":
            # Draw a border around h1s
            yield Panel(
                text,
                box=box.HEAVY,
                style="markdown.h1.border",
            )
            yield Text("")
        else:
            # Styled text for h2 and beyond
            if self.tag == "h2":
                yield Text("")
                yield Text(text.plain, justify="center", style=Style(bold=True))
            else:
                text.justify = "left"
                yield text


class CustomTable(TableElement):
    """MarkdownElement corresponding to `table_open`."""

    def __init__(self) -> None:
        super().__init__()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        table = Table(box=None, header_style=Style(bold=False), padding=(0, 1, 1, 0))

        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                table.add_column(column.content, overflow="fold")

        if self.body is not None:
            for row in self.body.rows:
                row_content = [element.content for element in row.cells]
                table.add_row(*row_content)

        yield table


class CustomListItem(ListItem):
    """An item in a list."""

    def __init__(self) -> None:
        super().__init__()

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 3)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet_style = console.get_style("markdown.item.bullet", default="none")

        bullet = Segment(" • ", Style(color="cyan", bold=True))
        padding = Segment(" " * 3, bullet_style)
        new_line = Segment("\n")
        for first, line in loop_first(lines):
            yield bullet if first else padding
            yield from line
            yield new_line


class CustomHorizontalRule(HorizontalRule):
    """A horizontal rule to divide sections."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(style=Style(color="cyan"))


class CustomParagraph(Paragraph):
    """use rich text to replace paragraph"""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text
        yield Text(text.plain, style=Style(bold=True))


class CloneProgress(RemoteProgress):
    def __init__(self, update_bar):
        super().__init__()
        self.update_bar = update_bar
        if not update_bar:
            self.pbar = tqdm(leave=False)

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self.update_bar:
            # convert to step size for specified total
            step_size = 100 / max_count
            self.update_bar(
                callback_event="preparing_contents", callback_args={"steps": step_size}
            )
        else:
            self.pbar.total = max_count
            self.pbar.n = cur_count
            self.pbar.refresh()


def remove_prefix(prefix: str, full_text: str) -> str:
    return full_text.removeprefix(prefix)


def remove_suffix(full_text: str, suffix: str) -> str:
    return full_text.removesuffix(suffix)


def generate_unique_id():
    random_string = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(8)
    )
    return random_string


def humanize_iso_date(date: str) -> str:
    format = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in date else "%Y-%m-%dT%H:%M:%SZ"
    time_interval = datetime.now(timezone.utc).replace(tzinfo=None) - datetime.strptime(
        date, format
    ).replace(tzinfo=None)
    if time_interval.days > 1:
        return humanize.naturaldate(datetime.strptime(date, format))
    else:
        return humanize.naturaltime(time_interval)


def datetime_to_string(date: str) -> str:
    original_format = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in date else "%Y-%m-%dT%H:%M:%S%z"
    parsed_format = "%Y-%m-%d %H:%M:%S%z"
    return datetime.strptime(date.replace("Z", "+0000"), original_format).strftime(
        parsed_format
    )


def epoch_to_iso_date(epoch):
    return datetime.fromtimestamp(epoch).astimezone().replace(microsecond=0).isoformat()


def parse_build_parameters(params: List[str]) -> str:
    dict_params = {}
    for param in params:
        param_key, param_value = param.split("=")
        dict_params[param_key] = param_value

    return json.dumps(dict_params)


def resolve_canonical_expression_to_url(expression: str, addSuffix: bool):
    """
    accept an expression such as granite-dot-build/gb-test or gb-test
    {SPACE_REPO_ORG}/{SPACE_REPO_NAME} or {SPACE_REPO_NAME}

    return a full git URL
    """

    # clean up '/' occurence at front or end of string
    if expression.startswith("/"):
        expression = expression[1:]
    if expression.endswith("/"):
        expression = expression[:-1]
    if expression.startswith("https://github.ibm.com/"):
        expression = expression.replace("https://github.ibm.com/", "")
    if expression.startswith("github.ibm.com/"):
        expression = expression.replace("github.ibm.com/", "")

    slash_count = expression.count("/")

    if slash_count == 0:
        space_repo_name = expression
        space_org = SPACE_REPO_ORG
    else:
        space_org, space_repo_name = expression.split("/", 1)

    # default values if not provided in expression
    space_org = space_org if space_org else SPACE_REPO_ORG
    space_repo_name = space_repo_name if space_repo_name else SPACE_REPO_NAME

    suffix = ""
    if addSuffix:
        suffix = ".git"

    return f"https://github.ibm.com/{space_org}/{space_repo_name}{suffix}"


def resolve_url_to_canonical_expression(url: str, removeSuffix=True):
    """
    accept a github url
    return canonical expression
    """

    expression = remove_prefix("https://github.ibm.com/", url)

    if removeSuffix:
        expression = remove_suffix(expression, ".git")

    if expression.startswith(f"{SPACE_REPO_ORG}/"):
        expression = expression.replace(f"{SPACE_REPO_ORG}/", "")

    return expression


def resolve_to_space_key(expression: str):
    canonical_expression = resolve_url_to_canonical_expression(expression, True)
    repo = None
    if "/" in canonical_expression:
        org, repo = canonical_expression.split("/")

    return repo if repo else canonical_expression


class DecodedURIResponse(BaseModel):
    uri: str
    namespace: str
    table_name: str
    type: str
    model_label: Optional[str] = None
    model_revision: Optional[str] = None
    fileset_label: Optional[str] = None
    fileset_version: Optional[str] = None
    dataset_name: Optional[str] = None


def __get_lh_decoded_uri_response(uri: LhURI) -> DecodedURIResponse:
    metadata = uri.get_metadata()
    response = DecodedURIResponse(**metadata)
    return response


def decode_uri(
    uri_input: str,
) -> DecodedURIResponse:
    uri = URI.get_uri(uri_input)

    if not isinstance(uri, LhURI):
        raise Exception(f"Error: Artifact URI formatted incorrectly.")

    assert isinstance(uri, LhURI)
    response = __get_lh_decoded_uri_response(uri)
    return response


def compare_env_uri(uri: str):
    return (
        uri.split("/")[2],
        str(gb_environment_config()["lakehouse_environment"]).lower(),
    )


def format_artifact_tags(artifacts: list):
    formatted_artifacts = []

    for artifact in artifacts:
        if artifact["tags"]:
            artifact["tags"] = [
                json.loads(a) if a[0] == "{" else a for a in artifact["tags"]
            ]
        else:
            artifact["tags"] = []

        formatted_artifacts.append(artifact)

    return formatted_artifacts


def is_official_artifact(artifact):
    # if any tag objects exists
    if len(artifact["tags"]) > 0:
        # check each item in 'tags' list
        for tag_obj in artifact["tags"]:
            if "sys-official" in tag_obj:
                return True
            else:
                return False
    return False


def get_artifact_formatted_name(decoded_artifact: DecodedURIResponse):
    # table : <namespace_name>.<table_name>
    # dataset : <dataset_name>|<namespace_name>.<table_name>
    # model : <model_label>.<revision>|<namespace_name>.<table_name>
    # fileset: <label>.<version>|<table_name>

    type = decoded_artifact.type
    namespace = decoded_artifact.namespace
    table = decoded_artifact.table_name

    match type:
        case "dataset":
            dataset = decoded_artifact.dataset_name
            return f"{dataset}|{namespace}.{table}"
        case "model":
            model_name = decoded_artifact.model_label
            revision = decoded_artifact.model_revision
            return f"{model_name}.{revision}|{namespace}.{table}"
        case "fileset":
            label = decoded_artifact.fileset_label
            version = decoded_artifact.fileset_version
            return f"{label}.{version}|{namespace}.{table}"
        case "table":
            return f"{namespace}.{table}"
        case _:
            return None


def get_artifact_lineage_url(decoded_artifact, artifact_id):
    type = decoded_artifact.type
    namespace = decoded_artifact.namespace
    table = decoded_artifact.table_name

    match type:
        case "model":
            model_label = decoded_artifact.model_label
            revision = decoded_artifact.model_revision
            return f"{DMF_URL}/v2/models/detail/{namespace}/{table}/{model_label}/{revision}"
        case "fileset":
            return f"{DMF_URL}/gb/artifacts/{artifact_id}"
        case "dataset":
            dataset = decoded_artifact.dataset_name
            return f"{DMF_URL}/v2/datasets/detail/{namespace}/{table}/{dataset}"
        case "table":
            return f"{DMF_URL}/v2/lakehouse/{namespace}/{table}/details"

    return None


def parse_artifact_identifier(identifier: str):
    uuid_format = re.compile(
        r"^[a-z0-9]*-[a-z0-9]*-[a-z0-9]*-[a-z0-9]*-[a-z0-9]*$",
        re.IGNORECASE,
    )
    table_format = re.compile(r"^[A-Za-z0-9-_.]*\.[A-Za-z0-9-_.]*$", re.IGNORECASE)
    dataset_format = re.compile(
        r"^[A-Za-z0-9-_.]*\|[A-Za-z0-9-_.]*\.[A-Za-z0-9-_.]*$", re.IGNORECASE
    )
    model_format = re.compile(
        r"^[A-Za-z0-9-_.]*\.[A-Za-z0-9-]*\|[A-Za-z0-9_.]*\.[A-Za-z0-9_.]*$",
        re.IGNORECASE,
    )
    fileset_format = re.compile(
        r"^[A-Za-z0-9-_.]*\.[A-Za-z0-9]*\|[A-Za-z0-9_.]*$",
        re.IGNORECASE,
    )

    if "lh://" in identifier:
        return "uri"
    if bool(uuid_format.match(identifier)):
        return "uuid"
    if bool(model_format.match(identifier)):
        return "model"
    if bool(dataset_format.match(identifier)):
        return "dataset"
    if bool(fileset_format.match(identifier)):
        return "fileset"
    if bool(table_format.match(identifier)):
        return "table"
    if "hf://" in identifier:
        return "uri"

    return None


def parse_table(s):
    namespace_name, table_name = s.rsplit(".", 1)
    return namespace_name, table_name


def parse_dataset(s):
    dataset_part, table_part = s.split("|")
    namespace_name, table_name = table_part.rsplit(".", 1)
    return dataset_part, namespace_name, table_name


def parse_model(s):
    model_part, table_part = s.split("|")
    model_label, revision = model_part.split(".")
    namespace_name, table_name = table_part.rsplit(".", 1)

    return model_label, revision, namespace_name, table_name


def parse_fileset(s):
    label_version, table_name = s.split("|")
    fileset_label, fileset_version = label_version.split(".")
    return fileset_label, fileset_version, table_name


def parse_build_identifier(identifier: str):
    if identifier:
        uuid_format = re.compile(
            r"^[a-z0-9]*-[a-z0-9]*-[a-z0-9]*-[a-z0-9]*-[a-z0-9]*$",
            re.IGNORECASE,
        )

        if "https://" in identifier:
            return "url"
        if bool(uuid_format.match(identifier)):
            return "uuid"
        if ".yaml" in identifier or ".yml" in identifier:
            return "filename"

        return None
    return None


def find_space_by_name(space_list: List, name: str):
    for space in space_list:
        if space["name"] == name:
            return space

    # could not find a valid space
    return None


def map_build_spaces_to_user_spaces(user_spaces: List, profile: dict):
    """
    input: all spaces from config, profile set spaces

    maps the space names defined in a build to all the space information that is stored
    in config cached user spaces
    """
    build_spaces_mapped = []
    # the default space and space aliases come from profile
    for key, value in profile.items():
        user_space = find_space_by_name(user_spaces, value)
        if user_space:
            if key == "default":
                # add default space to the front of the list for dispay
                build_spaces_mapped.insert(
                    0,
                    {
                        "name": "<default>",
                        "git_repo_uri": user_space.get("git_repo_uri"),
                        "lakehouse_namespace": user_space.get("lakehouse_namespace"),
                        "is_admin": user_space.get("is_admin"),
                    },
                )
            else:
                build_spaces_mapped.append(
                    {
                        "name": key,
                        "git_repo_uri": user_space.get("git_repo_uri"),
                        "lakehouse_namespace": user_space.get("lakehouse_namespace"),
                        "is_admin": user_space.get("is_admin"),
                    }
                )
        else:
            # could not lookup build space in cached user spaces
            build_spaces_mapped.append(
                {
                    "name": key,
                    "git_repo_uri": "<unknown>",
                    "lakehouse_namespace": "<unknown>",
                    "is_admin": "<unknown>",
                }
            )
    # the rest comes from remote spaces
    for space in user_spaces:
        build_spaces_mapped.append(
            {
                "name": space.get("name"),
                "git_repo_uri": space.get("git_repo_uri"),
                "lakehouse_namespace": space.get("lakehouse_namespace"),
                "is_admin": space.get("is_admin"),
            }
        )

    return build_spaces_mapped


def retry_function(func, retries=3, delay=2, *args, **kwargs):
    """
    Retries a function upon failure and returns the last exception if all attempts fail.
    :param func: The function to execute.
    :param retries: Number of times to retry.
    :param delay: Delay (in seconds) between retries.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :return: The function's return value if successful, else raises the last exception.
    """
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)  # Execute the function with arguments
        except Exception as e:
            last_exception = e
            time.sleep(delay)

    raise last_exception  # Raise the last encountered exception


def get_artifact_uuid(github_token: str, uri: str, callback=None):
    username = get_user(github_token).login

    gbserver_artifacts = make_gbserver_call(
        lambda: get_artifacts(
            github_token, GBSERVER_ARTIFACT_API, username, None, None
        )["artifacts"],
        callback,
    )

    matches = [a for a in gbserver_artifacts if a["uri"] == uri]

    if len(matches) == 1:
        return matches[0]["uuid"]
    elif len(matches) > 1:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Found multiple artifacts with matching uri. Please try again with uuid."
                },
            )
        return None
    else:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Matching artifact with specified uri not found."
                },
            )
        return None


def get_build_lineage_url(build_id: str):
    url = f"{DMF_URL}/gb/builds/{build_id}"

    if gb_environment_config()["env"] == "DEV":
        url = url + "?gb_environment=DEV"

    return url


def custom_parse_markdown_str(markdown_str: str) -> str:
    console = Console()
    Markdown.elements["heading_open"] = CustomHeading
    Markdown.elements["table_open"] = CustomTable
    Markdown.elements["list_item_open"] = CustomListItem
    Markdown.elements["hr"] = CustomHorizontalRule
    Markdown.elements["paragraph_open"] = CustomParagraph

    markdown = Markdown(markdown_str)
    with console.capture() as markdown_output:
        console.print(markdown)

    return markdown_output.get()


def parse_markdown_str(markdown_str: str) -> str:
    console = Console()

    markdown = Markdown(markdown_str)
    with console.capture() as markdown_output:
        console.print(markdown)

    return markdown_output.get()


def parse_markdown_file(path: str) -> str:
    console = Console()

    with open(path, "r", encoding="utf-8") as f:
        markdown = Markdown(f.read())

    with console.capture() as markdown_output:
        console.print(markdown)

    return markdown_output.get()


def read_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def find_duplicates(values):
    seen = set()
    duplicates = set()
    for s in values:
        if s in seen:
            duplicates.add(s)
        else:
            seen.add(s)

    return list(duplicates)


def origins_from_local(from_local: str) -> list:
    # Convert to Path and get the directory if it's a file
    start_path = Path(from_local)
    # If it's a file, it is dataset or fileset, only search in the same folder (non-recursive)
    if start_path.is_file():
        custom_origin_path = Path(from_local + ".origin")
        origins = [custom_origin_path] if custom_origin_path.exists() else None
    else:
        # If it's a directory, it means, it is fileset or model, so search recursively
        origins = [
            p
            for p in start_path.rglob("artifact.origin")
            if p.name == "artifact.origin"
        ]

    if origins and len(origins) > 0:
        path = origins[0]  # Take the first match
        try:
            with open(path, "r") as file:
                content = yaml.safe_load(file)

                items = []

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and {
                            "artifact_uri",
                            "artifact_id",
                        }.issubset(item):
                            items.append(item)

                elif isinstance(content, dict) and {
                    "artifact_uri",
                    "artifact_id",
                }.issubset(content):
                    items.append(content)

                return items

        except Exception as e:
            raise (f"Error reading or parsing {path}: {e}")

    return origins


def get_standard_model_prompt() -> list:
    # It's useful to include the date in the system prompt. System prompt can be overridden in chat subcommand.
    today = datetime.now().strftime("%A, %Y-%m-%d")
    return [
        {
            "role": "system",
            "content": f"You are an advanced chatbot and assistant. Today is {today}.",
        }
    ]


def is_valid_name(name_string: str, validation_type=None):
    # returns is_valid boolean and any invalid characters
    #
    # table_name: only alphanumeric and underscores
    # artifact_name and label_name: alphanumeric, underscores, dashes, and periods

    standard_chars = "abcdefghijklmnopqrstuvwxyz0123456789_"

    if validation_type == "table_name":
        allowed = standard_chars
    else:  # artifact_name, label_name
        allowed = standard_chars + "-."

    invalid_chars = []
    for char in name_string.lower():
        if char not in allowed:
            invalid_chars.append(char)

    return len(invalid_chars) == 0, list(set(invalid_chars))


def is_valid_checksum(checksum: str, expected_length):
    # returns is_valid boolean, any invalid characters and checksum length
    #
    # checksum: alphanumeric, 0-9, a-f
    # for 128bit checksum, length should be 32

    standard_chars = "abcdef0123456789"

    invalid_chars = []
    for char in checksum.lower():
        if char not in standard_chars:
            invalid_chars.append(char)

    checksum_length = len(checksum)

    return (
        len(invalid_chars) == 0 and checksum_length == expected_length,
        list(set(invalid_chars)),
        checksum_length,
    )


def step_uri_notation(step_name: str) -> str:
    return f"space://steps/{step_name}"


def check_current_timestamp(date, is_start_date: bool = False):
    is_current_timestamp = False
    current_time = round(time.time())
    if date < current_time:
        timestamp = date
    elif is_start_date:
        timestamp = change_timestamp_by_days(
            current_time, BUILD_LOG_DEFAULT_QUERY_RANGE
        )
    else:
        timestamp = current_time
        is_current_timestamp = True
    return timestamp, is_current_timestamp


def convert_seconds_to_milliseconds(timestamp):
    return timestamp * 1000


def convert_milliseconds_to_seconds(timestamp):
    return int(timestamp / 1000)


def get_current_epoch(nDaysAgo: int = None):
    return round(time.time())


def change_timestamp_by_days(timestamp, days: int, add: bool = False):
    if add:
        return timestamp + (86400 * days)
    return timestamp - (86400 * days)


def create_if_not_dir_local_build_cache() -> str:
    cache_path = get_local_build_cache()
    if not os.path.isdir(cache_path):
        cache_path.mkdir(mode=0o777, parents=True, exist_ok=False)

    return cache_path


def validate_tag_name(value: str, is_admin: bool, callback=None):
    """
    Validate the tag name.
    """
    if not is_admin and value is not None and value.startswith("sys-"):

        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": "Tag names starting with 'sys-' are not allowed unless you are a system admin.",
                },
            )
            return
        else:
            raise Exception(
                "Tag names starting with 'sys-' are not allowed unless you are a system admin."
            )
    value = value.strip() if value is not None else value
    if value is not None and not all(
        c.isalnum() or c == "_" or c == "-" for c in value.strip()
    ):
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "steps": 1,
                    "reason": "Tag name can only contain alphanumeric characters, underscores, and hyphens.",
                },
            )
        else:
            raise Exception(
                "Tag name can only contain alphanumeric characters, underscores, and hyphens."
            )
        return
    return value


def combine_tags(tags_str: str, tags_tuple: tuple):
    list_from_string = [s.strip() for s in tags_str.split(",")] if tags_str else []
    tuple_as_list = list(tags_tuple) if tags_tuple else []

    combined_list = list_from_string + tuple_as_list
    # Filter out empty strings after stripping
    all_tags = list(set(tag.strip() for tag in combined_list if tag.strip()))
    return all_tags


def check_runnable_browser():
    runnable_browser = True
    try:
        webbrowser.get()
    except Exception as e:
        runnable_browser = False

    return runnable_browser


def validate_tags(
    github_token: str,
    tags_as_tuple: tuple,
    tags_str: str,
    space: Optional[str] = None,
    callback=None,
) -> list:
    """
    Validate a tuple of tags and a string of comma-separated tags.
    Returns a list of unique validated tag names.
    """
    global_space = resolve_space(github_token, space, callback=callback)
    is_admin = global_space.get("is_admin") == True

    all_tags = combine_tags(tags_str=tags_str, tags_tuple=tags_as_tuple)
    all_validated_tags = [
        validate_tag_name(tag, is_admin, callback) for tag in all_tags
    ]

    return list(all_validated_tags)


def pagination_range(total_items: int, page_index: int, page_size: int):
    if total_items == 0:
        return 0, 0
    page_index = max(page_index, 0)
    page_size = max(page_size, 1)

    start = page_index * page_size + 1
    end = min((page_index + 1) * page_size, total_items)

    return start, end
