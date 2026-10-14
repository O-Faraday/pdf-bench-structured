"""Look around a Databricks workspace so the app can offer a SQL warehouse and a volume to pick.

Credentials come from ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` (``.env``): the SDK reads them from
the environment. Nothing here returns or prints the token.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass
class Workspace:
    host: str
    user: str
    warehouses: list[dict]  # id, name, state, type, size


def credentials_status() -> dict[str, bool]:
    return {name: bool(config.get(name)) for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")}


def client():
    from databricks.sdk import WorkspaceClient

    config.load_env()
    return WorkspaceClient()


def _value(x) -> str:
    return str(getattr(x, "value", x) or "")


def connect(w=None) -> Workspace:
    """Authenticate and list the SQL warehouses. Raises with a readable message on failure."""
    w = w or client()
    try:
        user = w.current_user.me().user_name or "?"
        warehouses = [{"id": e.id, "name": e.name, "state": _value(e.state), "type": _value(e.warehouse_type),
                       "size": e.cluster_size or ""} for e in w.warehouses.list()]
    except Exception as exc:  # noqa: BLE001 - show the reason, never a traceback with the token
        raise RuntimeError(f"Databricks connection failed: {type(exc).__name__}: {exc}") from exc
    return Workspace(w.config.host if getattr(w, "config", None) else (config.get("DATABRICKS_HOST") or ""),
                     user, warehouses)


def catalogs(w) -> list[str]:
    return sorted(c.name for c in w.catalogs.list())


def schemas(w, catalog: str) -> list[str]:
    return sorted(s.name for s in w.schemas.list(catalog_name=catalog))


def volumes(w, catalog: str, schema: str) -> list[str]:
    """Volume paths ``/Volumes/<catalog>/<schema>/<volume>``."""
    return sorted(f"/Volumes/{catalog}/{schema}/{v.name}" for v in w.volumes.list(catalog_name=catalog, schema_name=schema))
