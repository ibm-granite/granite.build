import logging
import requests
from openai import OpenAI
from typing import Any, List, Optional

from gbcli.utils.gbconstants import (
    RITS_BASE_URL,
    RITS_LIST_URL,
)
from gbcli.utils.utils import get_standard_model_prompt

logger = logging.getLogger(__name__)


def lookup_model_url(rits_api_key: str, model: str, callback=None):
    all_models = get_rits_models(rits_api_key)

    # Allow the user to specify part of a model name, if unique.
    matching_models = []

    for endpt, model_id in all_models.items():
        if model in endpt:
            # The "full" name we use in the dictionary to get a unique endpt+model combo.
            matching_models.append(endpt)

    # Found exactly one unique match, only non-error response from this call.
    if len(matching_models) == 1:
        return all_models[matching_models[0]], matching_models[0]

    if len(matching_models) == 0:
        raise Exception(
            f"No models matching {model} found in RITS. Use 'llmb model list' to view available models."
        )

    return matching_models


def get_rits_models(rits_api_key: str, callback=None):
    try:
        if callback:
            callback(
                callback_event="listing_models",
                callback_args={
                    "steps": 1,
                },
            )

        response = requests.get(
            RITS_LIST_URL,
            headers={"RITS_API_KEY": rits_api_key},
            timeout=10,
        )

        if response.status_code == 200:
            if callback:
                callback(
                    callback_event="set_total",
                    callback_args={"total": len(response.json())},  # TODO
                )
            results = {}

            for m in response.json():
                results[
                    f'{m["endpoint"].removeprefix(RITS_BASE_URL)}:{m["model_name"]}'
                ] = m["endpoint"]
                if callback:
                    callback(
                        callback_event="listing_models", callback_args={"steps": 1}
                    )
            if callback:
                callback(callback_event="listed_models", callback_args={})

            return results

        else:
            raise Exception(f"Failed getting RITS model list:\n\n{response.text}")

    except requests.exceptions.Timeout:
        raise Exception(
            f"Failed getting RITS model list: network timeout.  Check your VPN connection."
        )


def prompt_model(
    rits_api_key: str,
    prompt: str,
    url: str,
    model_id: str,
    temp: float,
    max: int,
    top_p: float,
    callback=None,
):
    client = OpenAI(
        api_key=rits_api_key,
        base_url=f"{url}/v1",
        default_headers={"RITS_API_KEY": rits_api_key},
    )

    model = model_id.split(":")[1]

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temp,
            top_p=top_p,
            max_tokens=max,
        )

        return response

    except Exception as e:
        if callback:
            callback(
                callback_event="error",
                callback_args={"reason": f"'{e.status_code} {e.detail}'."},
            )


def model_chat(
    rits_api_key: str,
    url: str,
    model_id: str,
    messages: List[Any],
    temp: float,
    max: int,
    top_p: float,
    chat_template: Optional[Any] = None,
    callback=None,
):
    client = OpenAI(
        api_key=rits_api_key,
        base_url=f"{url}/v1",
        default_headers={"RITS_API_KEY": rits_api_key},
    )

    prompt = messages if messages else get_standard_model_prompt()
    model = model_id.split(":")[1]

    try:
        if chat_template:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temp,
                top_p=top_p,
                max_tokens=max,
                extra_body={"chat_template": chat_template},
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temp,
                top_p=top_p,
                max_tokens=max,
            )

        return response.choices[0].message.content.strip()

    except Exception as e:
        if callback:
            callback(
                callback_event="error",
                callback_args={"reason": f"'{e.status_code} {e.detail}'."},
            )
        return ""
