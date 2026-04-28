import sys
from typing import Any, List, Optional, Tuple

from gbcli.services.service_admin import server_log
from gbcli.services.service_artifact import (
    ArtifactCopyResult,
    artifact_archive,
    artifact_copy,
    artifact_lineage,
    artifact_lineage_hf,
    artifact_list,
    check_artifact_existence,
    check_fileset_lh,
    download_fileset_lh,
    download_hf_artifact,
    download_model_lh,
    download_table_lh,
    get_artifact,
    get_artifact_uri,
    get_dataset_lh,
    get_model_lh,
    get_table_lh,
    register_artifact_gbserver,
    register_artifact_gbserver_hf,
    save_origin,
    update_artifact,
    upload_to_hf,
    upload_to_lh,
    validate_origins,
)
from gbcli.services.service_auth import (
    gbserver_login,
    gh_login,
    gh_token,
    gh_token_verify,
    ibmid_login,
    lakehouse_token_for_space,
    lh_artifact_token,
    lh_user_token,
    rits_user_api_key,
)
from gbcli.services.service_build import (
    build_cancel,
    build_describe,
    build_diff,
    build_init,
    build_lineage_gbserver,
    build_lineage_lh,
    build_list,
    build_log,
    build_monitor,
    build_notification,
    build_start,
    build_status,
    build_validate,
    fetch_build,
    update_build,
)
from gbcli.services.service_cleanup import (
    remove_config,
    remove_credentials,
    remove_local_cache,
    remove_user_fork_from_default,
)
from gbcli.services.service_model import (
    get_rits_models,
    lookup_model_url,
    model_chat,
    prompt_model,
)
from gbcli.services.service_secret import (
    create_secret,
    delete_secret,
    get_secret,
    list_secrets,
    update_secret,
)
from gbcli.services.service_space import (
    add_space_member,
    delete_space_member,
    list_space_members,
    list_spaces,
    set_space,
    update_space_member,
)
from gbcli.services.service_step import describe_step, list_steps
from gbcli.services.service_tag import artifact_tag_list, build_tag_list
from gbcli.services.service_template import describe_template, list_templates
from gbcli.services.service_version import get_gbserver_version
from gbcli.utils.gbconstants import (
    USER_NOT_LOGGED_IN_ERROR_MESSAGE,
    hf_token,
    is_standalone,
)
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.gh_auth import get_user


class GBClient:
    def __init__(self):
        pass

    class Admin:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def server_log(
            self,
            module,
            id_format: str,
            start_epoch: Optional[int] = None,
            end_epoch: Optional[int] = None,
            page_size: Optional[int] = None,
            page_index: Optional[int] = None,
            stream: Optional[str] = None,
            text: Optional[str] = None,
            sort: Optional[str] = None,
            build_id: Optional[str] = None,
            build_step_id: Optional[str] = None,
            build_step_name: Optional[str] = None,
            follow: Optional[bool] = False,
            all: Optional[bool] = False,
            callback=None,
        ):
            return server_log(
                self.github_token,
                module,
                id_format,
                start_epoch,
                end_epoch,
                page_size,
                page_index,
                stream,
                text,
                sort,
                build_id,
                build_step_id,
                build_step_name,
                follow,
                all,
                callback,
            )

        def list_space_members(self, space=None, callback=None):
            return list_space_members(self.github_token, space, callback)

        def add_space_member(self, space=None, username=None, role=None, callback=None):
            return add_space_member(self.github_token, space, username, role, callback)

        def update_space_member(
            self, space=None, username=None, role=None, callback=None
        ):
            return update_space_member(
                self.github_token, space, username, role, callback
            )

        def delete_space_member(self, space=None, username=None, callback=None):
            return delete_space_member(self.github_token, space, username, callback)

    class Artifact:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def validate_origins(
            self,
            artifact_name: str,
            local_origins: list,
            origin: tuple,
            origin_list: str,
            certify_no_restrictions: bool,
            callback=None,
        ):
            origin_as_list = list(origin) if origin != None else origin
            return validate_origins(
                github_token=self.github_token,
                artifact_name=artifact_name,
                local_origins=local_origins,
                origin=origin_as_list,
                origin_list=origin_list,
                certify_no_restrictions=certify_no_restrictions,
                callback=callback,
            )

        def push(
            self,
            lh_token: str,
            from_local: str,
            type: str,
            label: str,
            artifact_name: str,
            size: str,
            variant: str,
            model_type: str,
            version: str,
            space: Optional[str],
            table: Optional[str],
            namespace: Optional[str],
            store: str = "lh",
            hf_token: Optional[str] = None,
            hf_organization: Optional[str] = None,
            resource_group_id: Optional[str] = None,
            private: bool = False,
            callback=None,
        ):
            if store == "hf":
                return upload_to_hf(
                    hf_token=hf_token,
                    path_name=from_local,
                    artifact_name=artifact_name,
                    type=type,
                    hf_organization=hf_organization,
                    resource_group_id=resource_group_id,
                    private=private,
                    callback=callback,
                )
            else:
                if type in ["model", "fileset"]:
                    table = None
                elif type == "table":
                    label = None

                return upload_to_lh(
                    self.github_token,
                    lh_token=lh_token,
                    path_name=from_local,
                    type=type,
                    label=label,
                    artifact_name=artifact_name,
                    size=size,
                    variant=variant,
                    model_type=model_type,
                    version=version,
                    space=space,
                    table_name=table,
                    namespace=namespace,
                    callback=callback,
                )

        def check_existence(
            self,
            lh_token: str,
            type: str,
            space: Optional[str] = None,
            namespace: Optional[str] = None,
            table: Optional[str] = None,
            dataset: Optional[str] = None,
            label: Optional[str] = None,
            revision: Optional[str] = None,
            version: Optional[str] = None,
        ):
            return check_artifact_existence(
                self.github_token,
                lh_token,
                type,
                space,
                namespace,
                table,
                dataset,
                label,
                revision,
                version,
            )

        def register_artifact(
            self,
            artifact_name: str,
            description: str,
            checksum: str,
            tags: list[str],
            status: str,
            type: str,
            label: str = None,
            revision: str = None,
            version: str = None,
            space: Optional[str] = None,
            namespace: Optional[str] = None,
            table: Optional[str] = None,
            dataset_name: Optional[str] = None,
            lh_env: Optional[str] = None,
            origin_uris: Optional[list[str]] = None,
            certified_no_restrictions: bool = False,
            hf_organization: Optional[str] = None,
            resource_group_id: Optional[str] = None,
            store: str = "lh",
            callback=None,
        ):
            return (
                register_artifact_gbserver(
                    self.github_token,
                    artifact_name=artifact_name,
                    type=type,
                    label=label,
                    revision=revision,
                    version=version,
                    space=space,
                    namespace=namespace,
                    table=table,
                    dataset_name=dataset_name,
                    description=description,
                    checksum=checksum,
                    tags=tags,
                    status=status,
                    lh_env=lh_env,
                    origin_uris=origin_uris,
                    certified_no_restrictions=certified_no_restrictions,
                    store=store,
                    callback=callback,
                )
                if store == "lh"
                else register_artifact_gbserver_hf(
                    self.github_token,
                    artifact_name=artifact_name,
                    type=type,
                    description=description,
                    checksum=checksum,
                    tags=tags,
                    status=status,
                    revision=revision,
                    env=lh_env,
                    origin_uris=origin_uris,
                    certified_no_restrictions=certified_no_restrictions,
                    hf_organization=hf_organization,
                    resource_group_id=resource_group_id,
                )
            )

        def artifact_lineage(self, token: str, artifact_name: str):
            return artifact_lineage(token, artifact_name)

        def artifact_lineage_hf(self, repo_id: str):
            return artifact_lineage_hf(self.github_token, repo_id)

        def artifact_list(
            self,
            list_all: bool,
            show_archived: bool,
            show_pending: bool,
            build_id: str,
            id_format: str,
            space: str,
            all_spaces: bool,
            username: str | None,
            checksum: str | None,
            tags: list[str] | None,
            callback=None,
        ):
            return artifact_list(
                self.github_token,
                list_all,
                show_archived,
                show_pending,
                build_id,
                id_format,
                space,
                all_spaces,
                username,
                checksum,
                tags,
                callback=callback,
            )

        def existing_checksum_artifacts(self, space: str, checksum: str):
            results = artifact_list(
                self.github_token,
                list_all=True,
                show_pending=True,
                space=space,
                checksum=checksum,
            )
            if results:
                a = results[0]
                a["user_is_owner"] = get_user(self.github_token).login == a["username"]
                return a
            else:
                return None

        def fetch_artifact(self, artifact_id: str, callback=None):
            return get_artifact(self.github_token, artifact_id, callback=callback)

        def fetch_artifact_uri(self, artifact_uri: str, callback=None):
            return get_artifact_uri(self.github_token, artifact_uri, callback=callback)

        def get_dataset(
            self,
            lh_token: str,
            dataset_name: str,
            namespace: str,
            space: str,
            callback=None,
        ):
            return get_dataset_lh(
                self.github_token,
                lh_token,
                dataset_name,
                namespace,
                space,
                callback=callback,
            )

        def download_table(
            self,
            lh_token: str,
            namespace: str,
            table_name: str,
            directory: str,
            format: str,
            space: Optional[str],
            callback=None,
        ):
            return download_table_lh(
                self.github_token,
                lh_token,
                namespace,
                table_name,
                directory,
                format,
                space,
                callback,
            )

        def get_table(
            self,
            lh_token: str,
            namespace: str,
            table_name: str,
            space: Optional[str],
            callback=None,
        ):
            return get_table_lh(
                self.github_token, lh_token, namespace, table_name, space, callback
            )

        def download_model(
            self,
            lh_token: str,
            namespace: str,
            table_name: str,
            model_label: str,
            revision: str,
            directory: str,
            space: Optional[str],
            callback=None,
        ):
            return download_model_lh(
                self.github_token,
                lh_token,
                namespace,
                table_name,
                model_label,
                revision,
                directory,
                space,
                callback,
            )

        def save_origin(self, file_path, artifact):
            return save_origin(file_path, artifact)

        def save_origin(self, file_path, artifact):
            return save_origin(file_path, artifact)

        def check_fileset(
            self,
            lh_token: str,
            namespace: str,
            fileset_name: str,
            table_name: str,
            version: str,
            space: Optional[str],
            callback=None,
        ):
            return check_fileset_lh(
                self.github_token,
                lh_token,
                namespace,
                fileset_name,
                table_name,
                version,
                space,
                callback,
            )

        def download_fileset(
            self,
            lh_token: str,
            namespace: str,
            table_name: str,
            label: str,
            version: str,
            directory: str,
            space: Optional[str],
            callback=None,
        ):
            return download_fileset_lh(
                self.github_token,
                lh_token,
                namespace,
                table_name,
                label,
                version,
                directory,
                space,
                callback,
            )

        def download_hf_artifact(
            self,
            hf_token: str,
            repo_id: str,
            artifact_type: str,
            directory: str,
            revision: str = "main",
            callback=None,
        ):
            return download_hf_artifact(
                hf_token,
                repo_id,
                artifact_type,
                directory,
                revision,
                callback,
            )

        def get_model(
            self,
            lh_token: str,
            namespace: str,
            table_name: str,
            model_label: str,
            revision: str,
            space: Optional[str] = None,
            callback=None,
        ):
            return get_model_lh(
                self.github_token,
                lh_token,
                namespace,
                table_name,
                model_label,
                revision,
                space,
                callback,
            )

        def archive_artifact(self, artifact_id: str, archive: bool, callback=None):
            return artifact_archive(self.github_token, artifact_id, archive, callback)

        def artifact_copy(
            self,
            lh_token: str,
            source_namespace: str,
            source_table: str,
            space_to: str,
            artifact_name: str,
            revision: str,
            callback=None,
        ) -> ArtifactCopyResult:
            return artifact_copy(
                self.github_token,
                lh_token,
                source_namespace,
                source_table,
                space_to,
                artifact_name,
                revision,
                callback,
            )

        def update_artifact(
            self,
            artifact_id: str,
            tags: Optional[list[str]] = None,
            description: str = None,
            status: str = None,
            append: bool = False,
            isUpdate: bool = False,
            callback=None,
        ):

            return update_artifact(
                self.github_token,
                artifact_id=artifact_id,
                tags=tags,
                description=description,
                status=status,
                append=append,
                isUpdate=isUpdate,
                callback=callback,
            )

    class Auth:
        def __init__(self):
            pass

        def github_token(self):
            return gh_token()

        def login_github_with_token(self, gh_access_token: str):
            return gh_login(gh_access_token)

        def login_github(self, device_code: str):
            token_obj = gh_token_verify(device_code)
            gh_access_token = token_obj.access_token[0]
            return gh_login(gh_access_token)

        def lakehouse_token_for_space(
            github_token: str, space: str = None, callback=None
        ) -> str:
            return lakehouse_token_for_space(
                github_token, space=space, callback=callback
            )

        def lakehouse_token(github_token: str, callback=None) -> str:
            return lh_artifact_token(github_token, callback=callback)

        def lakehouse_user_token(github_token: str, callback=None) -> str:
            return lh_user_token(github_token, callback=callback)

        def rits_user_api_key(self):
            return rits_user_api_key()

        def hf_token(self) -> str:
            import os

            return os.environ.get("HF_TOKEN", None)

        def login_gbserver(self, api_user: str, api_key: str):
            return gbserver_login(api_user, api_key)

        def login_ibmid(self, open_browser=None):
            return ibmid_login(open_browser=open_browser)

    class Build:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def build_init(
            self,
            build_name: Optional[str] = None,
            filename: Optional[str] = "",
            space: Optional[str] = None,
            from_build: Optional[str] = None,
            from_template: Optional[str] = None,
            template_repo: Optional[str] = None,
            id_format: Optional[str] = None,
            callback=None,
        ):
            return build_init(
                self.github_token,
                build_name,
                filename,
                space,
                from_build,
                from_template,
                template_repo,
                id_format,
                callback,
            )

        def build_start(
            self,
            quiet: bool,
            filename: str,
            space: Optional[str] = None,
            params: Optional[List[str]] = [],
            skip_validation=False,
            parameters_path: Optional[str] = None,
            targets: Optional[tuple[str, ...]] = (),
            description: str = "",
            tags: list[str] = [],
            callback=None,
            validation_type: str = "static",
        ) -> str:
            return build_start(
                self.github_token,
                quiet,
                filename,
                space,
                params,
                skip_validation,
                parameters_path,
                targets,
                description=description,
                tags=tags,
                callback=callback,
                validation_type=validation_type,
            )

        def build_cancel(
            self,
            build_id: str,
            id_format: str,
            space: Optional[str] = None,
            callback=None,
        ) -> Any:
            return build_cancel(
                self.github_token, build_id, id_format, space, callback=callback
            )

        def build_lineage_lh(
            self, token: str, build_id: str, id_format: str, callback=None
        ):
            return build_lineage_lh(
                self.github_token, token, build_id, id_format, callback
            )

        def build_lineage(self, build_id: str, id_format: str, callback=None):
            return build_lineage_gbserver(
                self.github_token, build_id, id_format, callback
            )

        def build_list(
            self,
            list_all: bool,
            show_done: bool,
            show_all: bool,
            all_spaces: bool,
            space: Optional[str] = None,
            username: Optional[str] = None,
            tags: list[str] = None,
            page_index: Optional[int] = None,
            page_size: Optional[int] = None,
            callback=None,
        ):
            return build_list(
                self.github_token,
                list_all,
                show_done,
                show_all,
                all_spaces,
                space,
                username,
                tags,
                page_index,
                page_size,
                callback,
            )

        def build_log(
            self,
            id_format: str,
            start_epoch: Optional[int] = None,
            end_epoch: Optional[int] = None,
            page_size: Optional[int] = None,
            page_index: Optional[int] = None,
            stream: Optional[str] = None,
            text: Optional[str] = None,
            sort: Optional[str] = None,
            build_id: Optional[str] = None,
            build_step_id: Optional[str] = None,
            build_step_name: Optional[str] = None,
            runner: Optional[bool] = False,
            follow: Optional[bool] = False,
            all: Optional[bool] = False,
            skip_id_check: Optional[bool] = False,
            callback=None,
        ):
            return build_log(
                self.github_token,
                id_format,
                start_epoch,
                end_epoch,
                page_size,
                page_index,
                stream,
                text,
                sort,
                build_id,
                build_step_id,
                build_step_name,
                runner,
                follow,
                all,
                skip_id_check,
                callback,
            )

        def build_status(
            self,
            build_id: str,
            quiet: bool,
            id_format: str,
            show_events: bool,
            fetch_pr: bool,
            result_format: str,
            callback=None,
        ) -> str | List[Any]:
            return build_status(
                self.github_token,
                build_id,
                quiet,
                id_format,
                show_events,
                fetch_pr,
                result_format,
                callback=callback,
            )

        def build_describe(
            self,
            filename: str,
            format: str,
            raw: bool,
            build_id: Optional[str] = None,
            id_format: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> List[Any]:
            return build_describe(
                self.github_token,
                filename,
                format,
                raw,
                build_id,
                id_format,
                space,
                callback=callback,
            )

        def build_diff(
            self,
            build_id_1: str,
            id_format_1: str,
            build_id_2: Optional[str] = None,
            id_format_2: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> Tuple[str, str, List[Any]]:
            return build_diff(
                self.github_token,
                build_id_1,
                id_format_1,
                build_id_2,
                id_format_2,
                space,
                callback=callback,
            )

        def build_monitor(
            self,
            build_id: str,
            show_events: bool,
            fetch_pr: bool,
            id_format: Optional[str] = None,
            callback=None,
        ) -> Tuple[Any, List[Any], List[Any], List[Any]]:
            return build_monitor(
                self.github_token,
                build_id,
                show_events,
                fetch_pr,
                id_format,
                callback=callback,
            )

        def build_validate(
            self,
            quiet: bool,
            filename: Optional[str] = "",
            space: Optional[str] = None,
            params: Optional[List[str]] = [],
            parameters_path: Optional[str] = None,
            targets: Optional[tuple[str, ...]] = (),
            callback=None,
            validation_type: str = "static",
        ) -> Optional[Tuple]:
            return build_validate(
                self.github_token,
                quiet,
                filename,
                space,
                params,
                parameters_path,
                targets,
                callback=callback,
                validation_type=validation_type,
            )

        def update_build(
            self,
            build_id: str,
            tags: Optional[list[str]] = None,
            description: str = None,
            append: bool = False,
            callback=None,
        ):

            return update_build(
                self.github_token,
                build_id=build_id,
                tags=tags,
                description=description,
                append=append,
                callback=callback,
            )

        def build_notification(
            self,
            status: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> Tuple[str, str]:
            return build_notification(self.github_token, status, space, callback)

        def fetch_build(
            self,
            build_id: str,
            id_format: str,
            callback=None,
        ):
            return fetch_build(
                self.github_token,
                build_id=build_id,
                id_format=id_format,
                callback=callback,
            )

    class Cleanup:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def remove_config(self):
            return remove_config()

        def remove_credentials(self):
            return remove_credentials()

        def remove_local_cache(self):
            return remove_local_cache()

        def remove_default_fork(self, callback=None):
            return remove_user_fork_from_default(self.github_token, callback)

    class Model:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def lookup_model(self, rits_api_key: str, model_url: str, callback=None):
            return lookup_model_url(rits_api_key, model_url, callback)

        def get_rits_models(self, rits_api_key: str, callback=None):
            return get_rits_models(rits_api_key, callback)

        def prompt_model(
            self,
            rits_api_key: str,
            prompt: str,
            url: str,
            model_id: str,
            temp: float,
            max: int,
            top_p: float,
            callback=None,
        ):
            return prompt_model(
                rits_api_key, prompt, url, model_id, temp, max, top_p, callback
            )

        def model_chat(
            self,
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
            return model_chat(
                rits_api_key,
                url,
                model_id,
                messages,
                temp,
                max,
                top_p,
                chat_template,
                callback,
            )

    class Space:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def set_space(self, space_name: str, default: bool, callback=None, name=None):
            return set_space(self.github_token, space_name, default, callback, name)

        def list_spaces(self, all: bool, refresh: bool, callback=None) -> List[Any]:
            return list_spaces(self.github_token, all, refresh, callback)

    class Secret:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def list_secrets(
            self,
            personal: bool,
            space: Optional[str] = None,
            callback=None,
        ) -> Any:
            return list_secrets(self.github_token, personal, space, callback)

        def get_secret(
            self,
            secret_name: str,
            personal: bool,
            space: Optional[str] = None,
            callback=None,
        ) -> Any:
            return get_secret(self.github_token, secret_name, personal, space, callback)

        def create_secret(
            self,
            secret_name: str,
            personal: bool,
            secret_value: Optional[str] = None,
            space: Optional[str] = None,
            path_name: Optional[str] = None,
            callback=None,
        ) -> Tuple[Any, str, str]:
            return create_secret(
                self.github_token,
                secret_name,
                personal,
                secret_value,
                space,
                path_name,
                callback,
            )

        def update_secret(
            self,
            secret_name: str,
            personal: bool,
            secret_value: Optional[str] = None,
            space: Optional[str] = None,
            path_name: Optional[str] = None,
            callback=None,
        ) -> Tuple[Any, str, str]:
            return update_secret(
                self.github_token,
                secret_name,
                personal,
                secret_value,
                space,
                path_name,
                callback,
            )

        def delete_secret(
            self,
            secret_name: str,
            personal: bool,
            space: Optional[str] = None,
            callback=None,
        ) -> Tuple[Any, str, str]:
            return delete_secret(
                self.github_token, secret_name, personal, space, callback
            )

    class Step:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def list_steps(
            self,
            step_repo: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> List[Any]:
            return list_steps(self.github_token, step_repo, space, callback)

        def describe_step(
            self,
            step_name: str,
            step_repo: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> Any:
            return describe_step(
                self.github_token, step_name, step_repo, space, callback
            )

    class Template:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def list_templates(
            self,
            space: Optional[str] = None,
            template_repo: Optional[str] = None,
            callback=None,
        ) -> List[Any]:
            return list_templates(self.github_token, space, template_repo, callback)

        def describe_template(
            self,
            template_name: str,
            format: str,
            space: Optional[str] = None,
            template_repo: Optional[str] = None,
            callback=None,
        ) -> List[Any]:
            return describe_template(
                self.github_token, template_name, format, space, template_repo, callback
            )

    class Version:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def get_gbserver_version(self, quiet: bool, callback=None) -> str:
            return get_gbserver_version(self.github_token, quiet, callback)

    class Tag:
        def __init__(self, github_token: str):
            self.github_token = github_token

        def artifact_tag_list(
            self,
            username: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> list[str]:
            return artifact_tag_list(self.github_token, username, space, callback)

        def build_tag_list(
            self,
            username: Optional[str] = None,
            space: Optional[str] = None,
            callback=None,
        ) -> list[str]:
            return build_tag_list(self.github_token, username, space, callback)
