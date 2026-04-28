from gbserver.types.constants import (
    GRANITE_DOT_BUILD_PARENT_NAMESPACE,
    LAKEHOUSE_ENVIRONMENT,
    PUBLIC_SPACE_LH_NAMESPACE,
)


def lakehouse_list_user_namespaces(token):
    """Get lakehouse namespaces with write permission"""
    from lakehouse.api import AccessPermission, ConfigMap

    from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

    try:
        lh = create_lakehouse_iceberg(
            config="map",
            conf_map=ConfigMap(
                token=token,
                environment=LAKEHOUSE_ENVIRONMENT,
            ),
        )
        namespaces = lh.list_children_namespaces(GRANITE_DOT_BUILD_PARENT_NAMESPACE)

        access_namespaces = []

        for namespace in namespaces:
            try:
                if (
                    namespace == PUBLIC_SPACE_LH_NAMESPACE
                    or lh.get_lakehouse_api().has_current_user_permission_on_namespace(
                        namespace, AccessPermission.WRITE
                    )
                ):
                    access_namespaces.append(namespace)
            except Exception as e:
                print(f"Failed to get a Lakehouse namespace {namespace} access: {e}")
                pass
        return access_namespaces
    except Exception as e:
        print(f"Error accessing Lakehouse namespace list: {e}")
        return {"error": str(e)}


def lakehouse_user_namespaces_admin_details(token, namespaces):
    """Get lakehouse namespaces user has access"""
    from lakehouse import Environment, LakehouseApi
    from lakehouse.api import AccessPermission, ConfigMap

    from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

    try:
        lh = create_lakehouse_iceberg(
            config="map",
            conf_map=ConfigMap(
                token=token,
                environment=LAKEHOUSE_ENVIRONMENT,
            ),
        )

        user_namespaces = []

        user_email = lh.get_lakehouse_api().get_token_details().email.lower()

        api = LakehouseApi(
            host=Environment.build_from(LAKEHOUSE_ENVIRONMENT).value, token=token
        )
        result = api.make_api_call(
            url=f"{api.host}/authz/namespace/team_with_owners", params={}
        )

        namespaces_owner = []

        for val in result:
            if user_email in val["owners"]:
                namespaces_owner.append(val["namespace"])

        for namespace in namespaces:
            try:
                if (
                    namespace == PUBLIC_SPACE_LH_NAMESPACE
                    or namespace in namespaces_owner
                    or lh.get_lakehouse_api().has_current_user_permission_on_namespace(
                        namespace, AccessPermission.WRITE
                    )
                ):
                    namespace_details = {
                        "namespace": namespace,
                        "is_admin": True if namespace in namespaces_owner else False,
                    }
                    user_namespaces.append(namespace_details)
            except Exception as e:
                print(f"Failed to get a Lakehouse namespace {namespace} access: {e}")
                pass
        return user_namespaces
    except Exception as e:
        print(f"Error accessing Lakehouse team owner namespace list: {e}")
        return []


def has_access_to_lakehouse_namespace(lh_token: str, namespace: str):
    """Check if the user has write access to namespace.

    Args:
        lh_token: Lakehouse token (already obtained from GitHub token exchange).
        namespace: Lakehouse namespace to check.
    """
    from lakehouse.api import AccessPermission, ConfigMap

    from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

    try:
        hasAccess = False
        if lh_token:
            lh = create_lakehouse_iceberg(
                config="map",
                conf_map=ConfigMap(
                    token=lh_token,
                    environment=LAKEHOUSE_ENVIRONMENT,
                ),
            )
            hasAccess = lh.get_lakehouse_api().has_current_user_permission_on_namespace(
                namespace, AccessPermission.WRITE
            )
        return hasAccess
    except Exception as e:
        print(f"Error accessing Lakehouse : {e}")
        return []
